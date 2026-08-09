import asyncio

import httpx
import pytest

from backend.main import ConnectionManager, poll_loop, poll_once, snapshot_to_message
from backend.poller import VLLMMetrics
from backend.store import MetricsSnapshot, MetricsStore
from tests.test_poller import SAMPLE_METRICS_TEXT


class FakeConnection:
    def __init__(self, fail: bool = False):
        self.sent: list[dict] = []
        self.fail = fail

    async def send_json(self, message: dict) -> None:
        if self.fail:
            raise RuntimeError("connection closed")
        self.sent.append(message)


def _metrics() -> VLLMMetrics:
    return VLLMMetrics(
        kv_cache_usage_perc=0.5,
        num_requests_running=2.0,
        num_requests_waiting=1.0,
        prompt_tokens_total=10.0,
        generation_tokens_total=20.0,
        ttft_avg_seconds=0.2,
        request_success={"stop": 3.0},
    )


def test_snapshot_to_message_includes_timestamp_and_metrics_fields():
    snapshot = MetricsSnapshot(timestamp=123.0, metrics=_metrics())

    message = snapshot_to_message(snapshot)

    assert message["timestamp"] == 123.0
    assert message["kv_cache_usage_perc"] == 0.5
    assert message["request_success"] == {"stop": 3.0}


async def test_connection_manager_broadcasts_to_all_registered_connections():
    manager = ConnectionManager()
    a, b = FakeConnection(), FakeConnection()
    manager.register(a)
    manager.register(b)

    await manager.broadcast({"hello": "world"})

    assert a.sent == [{"hello": "world"}]
    assert b.sent == [{"hello": "world"}]


async def test_connection_manager_drops_connections_that_fail_to_send():
    manager = ConnectionManager()
    ok, broken = FakeConnection(), FakeConnection(fail=True)
    manager.register(ok)
    manager.register(broken)

    await manager.broadcast({"x": 1})

    assert ok.sent == [{"x": 1}]
    assert broken not in manager._connections
    assert ok in manager._connections


async def test_poll_once_fetches_parses_stores_and_broadcasts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SAMPLE_METRICS_TEXT)

    transport = httpx.MockTransport(handler)
    store = MetricsStore(retention_seconds=60.0)
    manager = ConnectionManager()
    conn = FakeConnection()
    manager.register(conn)

    async with httpx.AsyncClient(transport=transport) as client:
        snapshot = await poll_once(
            store=store, manager=manager, client=client, url="http://mock/metrics", timestamp=100.0
        )

    assert snapshot.timestamp == 100.0
    stored = store.get_recent()
    assert len(stored) == 1
    assert stored[0].metrics.kv_cache_usage_perc == pytest.approx(0.42)
    assert len(conn.sent) == 1
    assert conn.sent[0]["kv_cache_usage_perc"] == pytest.approx(0.42)


async def test_poll_loop_keeps_running_after_a_fetch_error():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, text=SAMPLE_METRICS_TEXT)

    transport = httpx.MockTransport(handler)
    store = MetricsStore(retention_seconds=60.0)
    manager = ConnectionManager()

    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            poll_loop(store=store, manager=manager, client=client, url="http://mock/metrics", interval=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert calls["count"] >= 2
    assert len(store.get_recent()) >= 1
