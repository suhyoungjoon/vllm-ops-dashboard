"""FastAPI 앱 — vLLM /metrics를 폴링해서 WebSocket으로 프론트에 실시간 push.

프론트 연결 시 최근 보존된 스냅샷(store.get_recent())을 먼저 전송한 뒤,
이후 폴링될 때마다 새 스냅샷을 push한다.
"""

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from backend.diagnostics import Diagnosis, diagnose
from backend.poller import fetch_metrics, parse_metrics
from backend.store import DEFAULT_RETENTION_SECONDS, MetricsSnapshot, MetricsStore

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8000/metrics")
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
RETENTION_SECONDS = float(os.environ.get("RETENTION_SECONDS", str(DEFAULT_RETENTION_SECONDS)))


class SupportsSendJSON(Protocol):
    async def send_json(self, message: dict) -> None: ...


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[Any] = set()

    def register(self, connection: SupportsSendJSON) -> None:
        self._connections.add(connection)

    def unregister(self, connection: SupportsSendJSON) -> None:
        self._connections.discard(connection)

    async def broadcast(self, message: dict) -> None:
        stale = []
        for connection in self._connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.unregister(connection)


def snapshot_to_message(snapshot: MetricsSnapshot, diagnosis: list[Diagnosis] | None = None) -> dict:
    return {
        "timestamp": snapshot.timestamp,
        **asdict(snapshot.metrics),
        "diagnosis": [asdict(d) for d in (diagnosis or [])],
    }


async def poll_once(
    store: MetricsStore,
    manager: ConnectionManager,
    client: httpx.AsyncClient,
    url: str,
    timestamp: float | None = None,
) -> MetricsSnapshot:
    text = await fetch_metrics(client, url)
    metrics = parse_metrics(text)
    ts = timestamp if timestamp is not None else time.time()

    previous = store.get_recent()
    prev_preemptions = previous[-1].metrics.num_preemptions_total if previous else metrics.num_preemptions_total
    preemptions_delta = max(0.0, metrics.num_preemptions_total - prev_preemptions)

    snapshot = MetricsSnapshot(timestamp=ts, metrics=metrics)
    store.add(metrics, ts)
    diagnosis = diagnose(metrics, preemptions_delta)
    await manager.broadcast(snapshot_to_message(snapshot, diagnosis))
    return snapshot


async def poll_loop(
    store: MetricsStore,
    manager: ConnectionManager,
    client: httpx.AsyncClient,
    url: str,
    interval: float,
) -> None:
    while True:
        with contextlib.suppress(Exception):
            await poll_once(store, manager, client, url)
        await asyncio.sleep(interval)


store = MetricsStore(retention_seconds=RETENTION_SECONDS)
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        task = asyncio.create_task(poll_loop(store, manager, client, VLLM_METRICS_URL, POLL_INTERVAL_SECONDS))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    manager.register(websocket)
    try:
        for snapshot in store.get_recent():
            await websocket.send_json(snapshot_to_message(snapshot))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.unregister(websocket)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
