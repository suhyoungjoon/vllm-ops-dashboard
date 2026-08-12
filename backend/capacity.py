"""KV 캐시 사용률 추세로 용량 여유와 포화 임박 시점을 추정한다.

정밀한 예측이 아니라 "이 추세가 계속되면 대략 언제쯤" 수준의 방향성 지표다.
최근 시계열 중 첫 지점과 마지막 지점 두 점만으로 기울기를 구하는 단순한
근사이며, 노이즈에 민감할 수 있다는 점을 감안하고 사용한다.
"""

from dataclasses import dataclass

from backend.poller import VLLMMetrics
from backend.store import MetricsSnapshot

SATURATION_TARGET = 0.95
TREND_WINDOW_SECONDS = 180.0
MIN_TREND_SPAN_SECONDS = 20.0


@dataclass
class CapacityEstimate:
    max_concurrency: float | None
    current_concurrency: float
    headroom: float | None
    headroom_ratio: float | None
    kv_trend_per_minute: float | None
    minutes_to_saturation: float | None


def _max_concurrency(cache_config: dict[str, str] | None) -> float | None:
    if not cache_config or "kv_cache_max_concurrency" not in cache_config:
        return None
    try:
        return float(cache_config["kv_cache_max_concurrency"])
    except ValueError:
        return None


def _kv_trend_per_minute(recent: list[MetricsSnapshot], latest_timestamp: float) -> float | None:
    window = [s for s in recent if latest_timestamp - s.timestamp <= TREND_WINDOW_SECONDS]
    if len(window) < 2:
        return None
    first, last = window[0], window[-1]
    span = last.timestamp - first.timestamp
    if span < MIN_TREND_SPAN_SECONDS:
        return None
    return (last.metrics.kv_cache_usage_perc - first.metrics.kv_cache_usage_perc) / span * 60.0


def estimate_capacity(
    metrics: VLLMMetrics, recent: list[MetricsSnapshot], latest_timestamp: float
) -> CapacityEstimate:
    max_concurrency = _max_concurrency(metrics.cache_config)
    headroom = max_concurrency - metrics.num_requests_running if max_concurrency is not None else None
    headroom_ratio = metrics.num_requests_running / max_concurrency if max_concurrency else None

    trend = _kv_trend_per_minute(recent, latest_timestamp)
    minutes_to_saturation = None
    if trend is not None and trend > 0:
        if metrics.kv_cache_usage_perc >= SATURATION_TARGET:
            minutes_to_saturation = 0.0
        else:
            minutes_to_saturation = (SATURATION_TARGET - metrics.kv_cache_usage_perc) / trend

    return CapacityEstimate(
        max_concurrency=max_concurrency,
        current_concurrency=metrics.num_requests_running,
        headroom=headroom,
        headroom_ratio=headroom_ratio,
        kv_trend_per_minute=trend,
        minutes_to_saturation=minutes_to_saturation,
    )
