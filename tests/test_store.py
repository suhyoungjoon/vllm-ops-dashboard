from backend.poller import VLLMMetrics
from backend.store import MetricsStore


def _metrics(kv: float = 0.5) -> VLLMMetrics:
    return VLLMMetrics(
        kv_cache_usage_perc=kv,
        num_requests_running=1.0,
        num_requests_waiting=0.0,
        prompt_tokens_total=10.0,
        generation_tokens_total=20.0,
        ttft_avg_seconds=0.1,
        request_success={"stop": 1.0},
    )


def test_empty_store_returns_no_snapshots():
    store = MetricsStore(retention_seconds=60.0)
    assert store.get_recent() == []


def test_get_recent_returns_added_snapshot():
    store = MetricsStore(retention_seconds=60.0)
    store.add(_metrics(kv=0.7), timestamp=100.0)

    snapshots = store.get_recent()

    assert len(snapshots) == 1
    assert snapshots[0].timestamp == 100.0
    assert snapshots[0].metrics.kv_cache_usage_perc == 0.7


def test_get_recent_returns_snapshots_in_insertion_order():
    store = MetricsStore(retention_seconds=60.0)
    store.add(_metrics(kv=0.1), timestamp=100.0)
    store.add(_metrics(kv=0.2), timestamp=101.0)
    store.add(_metrics(kv=0.3), timestamp=102.0)

    snapshots = store.get_recent()

    assert [s.metrics.kv_cache_usage_perc for s in snapshots] == [0.1, 0.2, 0.3]


def test_snapshots_older_than_retention_window_are_pruned():
    store = MetricsStore(retention_seconds=60.0)
    store.add(_metrics(kv=0.1), timestamp=0.0)
    store.add(_metrics(kv=0.2), timestamp=30.0)
    store.add(_metrics(kv=0.3), timestamp=61.0)  # cutoff is 1.0, so timestamp=0.0 falls out

    snapshots = store.get_recent()

    assert [s.timestamp for s in snapshots] == [30.0, 61.0]


def test_snapshot_exactly_at_cutoff_is_kept():
    store = MetricsStore(retention_seconds=60.0)
    store.add(_metrics(), timestamp=40.0)
    store.add(_metrics(), timestamp=100.0)  # cutoff = 40.0, boundary item kept

    snapshots = store.get_recent()

    assert [s.timestamp for s in snapshots] == [40.0, 100.0]
