import pytest

from backend.capacity import estimate_capacity
from backend.poller import VLLMMetrics
from backend.store import MetricsSnapshot


def _metrics(kv: float = 0.5, running: float = 5.0, cache_config: dict | None = None) -> VLLMMetrics:
    return VLLMMetrics(
        kv_cache_usage_perc=kv,
        num_requests_running=running,
        num_requests_waiting=0.0,
        prompt_tokens_total=10.0,
        generation_tokens_total=20.0,
        ttft_avg_seconds=0.1,
        request_success={},
        cache_config=cache_config,
    )


def _snapshot(timestamp: float, kv: float) -> MetricsSnapshot:
    return MetricsSnapshot(timestamp=timestamp, metrics=_metrics(kv=kv))


def test_headroom_computed_from_cache_config_max_concurrency():
    metrics = _metrics(running=5.0, cache_config={"kv_cache_max_concurrency": "20.0"})
    result = estimate_capacity(metrics, recent=[], latest_timestamp=0.0)
    assert result.max_concurrency == 20.0
    assert result.headroom == 15.0
    assert result.headroom_ratio == 0.25


def test_headroom_none_when_cache_config_missing():
    metrics = _metrics(running=5.0, cache_config=None)
    result = estimate_capacity(metrics, recent=[], latest_timestamp=0.0)
    assert result.max_concurrency is None
    assert result.headroom is None
    assert result.headroom_ratio is None


def test_headroom_none_when_max_concurrency_unparseable():
    metrics = _metrics(running=5.0, cache_config={"kv_cache_max_concurrency": "n/a"})
    result = estimate_capacity(metrics, recent=[], latest_timestamp=0.0)
    assert result.max_concurrency is None


def test_trend_none_with_fewer_than_two_points_in_window():
    metrics = _metrics(kv=0.5)
    recent = [_snapshot(100.0, 0.5)]
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=100.0)
    assert result.kv_trend_per_minute is None
    assert result.minutes_to_saturation is None


def test_trend_none_when_span_below_minimum():
    metrics = _metrics(kv=0.5)
    recent = [_snapshot(100.0, 0.3), _snapshot(110.0, 0.5)]  # 10s span < 20s minimum
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=110.0)
    assert result.kv_trend_per_minute is None


def test_trend_excludes_points_outside_window():
    # 첫 지점이 TREND_WINDOW_SECONDS(180s)보다 오래돼서 제외되어야 함
    recent = [_snapshot(0.0, 0.0), _snapshot(190.0, 0.3), _snapshot(220.0, 0.4)]
    metrics = _metrics(kv=0.4)
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=220.0)
    # window = [190:0.3, 220:0.4] -> span 30s, delta 0.1 -> per minute = 0.1/30*60=0.2
    assert result.kv_trend_per_minute is not None
    assert round(result.kv_trend_per_minute, 4) == 0.2


def test_minutes_to_saturation_none_when_trend_not_increasing():
    recent = [_snapshot(0.0, 0.6), _snapshot(60.0, 0.5)]
    metrics = _metrics(kv=0.5)
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=60.0)
    assert result.kv_trend_per_minute is not None
    assert result.kv_trend_per_minute < 0
    assert result.minutes_to_saturation is None


def test_minutes_to_saturation_computed_for_positive_trend():
    # 60초 동안 0.5 -> 0.6, 분당 0.1 증가. 목표 0.95까지 (0.95-0.6)/0.1 = 3.5분
    recent = [_snapshot(0.0, 0.5), _snapshot(60.0, 0.6)]
    metrics = _metrics(kv=0.6)
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=60.0)
    assert result.minutes_to_saturation == pytest.approx(3.5)


def test_minutes_to_saturation_zero_when_already_at_target():
    recent = [_snapshot(0.0, 0.9), _snapshot(60.0, 0.96)]
    metrics = _metrics(kv=0.96)
    result = estimate_capacity(metrics, recent=recent, latest_timestamp=60.0)
    assert result.minutes_to_saturation == 0.0
