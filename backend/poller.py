"""vLLM `/metrics` 폴링 + Prometheus 텍스트 포맷 파싱.

여러 engine 라벨(데이터 병렬 등)이 존재할 수 있다는 가정 하에, gauge/counter는
모든 샘플을 합산하고 히스토그램은 sum/count/bucket을 각각 합산한 뒤 평균 또는
분위수(quantile)를 계산한다.
"""

import math
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

GAUGE_NAMES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")
COUNTER_NAMES = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)
REQUEST_SUCCESS_NAME = "vllm:request_success_total"
CACHE_CONFIG_INFO_NAME = "vllm:cache_config_info"

# sum/count/bucket을 모두 사용해 p50/p90/p99까지 계산하는 히스토그램
QUANTILE_HISTOGRAM_BASE_NAMES = ("vllm:time_to_first_token_seconds", "vllm:e2e_request_latency_seconds")
# sum/count만 사용해 평균만 계산하는 히스토그램 (버킷 경계가 안 쓰임)
AVERAGE_HISTOGRAM_BASE_NAMES = ("vllm:request_queue_time_seconds", "vllm:request_inference_time_seconds")


@dataclass
class VLLMMetrics:
    kv_cache_usage_perc: float
    num_requests_running: float
    num_requests_waiting: float
    prompt_tokens_total: float
    generation_tokens_total: float
    ttft_avg_seconds: float | None
    request_success: dict[str, float]
    cache_config: dict[str, str] | None = None
    ttft_p50_seconds: float | None = None
    ttft_p90_seconds: float | None = None
    ttft_p99_seconds: float | None = None
    e2e_p50_seconds: float | None = None
    e2e_p90_seconds: float | None = None
    e2e_p99_seconds: float | None = None
    queue_time_avg_seconds: float | None = None
    inference_time_avg_seconds: float | None = None
    num_preemptions_total: float = 0.0
    prefix_cache_hit_rate: float | None = None


def _parse_le(raw: str) -> float:
    return math.inf if raw == "+Inf" else float(raw)


def _histogram_quantile(buckets: dict[float, float], total_count: float, q: float) -> float | None:
    """누적 버킷(le -> 누적 카운트)에서 분위수를 선형보간으로 근사한다 (PromQL histogram_quantile과 동일 방식)."""
    if not buckets or total_count <= 0:
        return None

    target = q * total_count
    prev_bound, prev_count = 0.0, 0.0
    for bound in sorted(buckets):
        count = buckets[bound]
        if count >= target:
            if bound == math.inf or count == prev_count:
                return prev_bound
            frac = (target - prev_count) / (count - prev_count)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound, prev_count = bound, count
    return prev_bound


def parse_metrics(text: str) -> VLLMMetrics:
    values: dict[str, float] = {}
    request_success: dict[str, float] = {}
    histogram_buckets: dict[str, dict[float, float]] = {name: {} for name in QUANTILE_HISTOGRAM_BASE_NAMES}
    cache_config: dict[str, str] | None = None

    histogram_sum_count_names = {
        f"{base}_{suffix}" for base in (*QUANTILE_HISTOGRAM_BASE_NAMES, *AVERAGE_HISTOGRAM_BASE_NAMES) for suffix in ("sum", "count")
    }
    bucket_name_to_base = {f"{base}_bucket": base for base in QUANTILE_HISTOGRAM_BASE_NAMES}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            if name in GAUGE_NAMES or name in COUNTER_NAMES or name in histogram_sum_count_names:
                values[name] = values.get(name, 0.0) + sample.value
            elif name == REQUEST_SUCCESS_NAME:
                reason = sample.labels.get("finished_reason", "unknown")
                request_success[reason] = request_success.get(reason, 0.0) + sample.value
            elif name == CACHE_CONFIG_INFO_NAME:
                cache_config = dict(sample.labels)
            elif name in bucket_name_to_base:
                base = bucket_name_to_base[name]
                le = _parse_le(sample.labels.get("le", "+Inf"))
                histogram_buckets[base][le] = histogram_buckets[base].get(le, 0.0) + sample.value

    def _avg(base: str) -> float | None:
        count = values.get(f"{base}_count", 0.0)
        return values.get(f"{base}_sum", 0.0) / count if count > 0 else None

    def _percentiles(base: str) -> tuple[float | None, float | None, float | None]:
        count = values.get(f"{base}_count", 0.0)
        buckets = histogram_buckets[base]
        return (
            _histogram_quantile(buckets, count, 0.5),
            _histogram_quantile(buckets, count, 0.9),
            _histogram_quantile(buckets, count, 0.99),
        )

    ttft_p50, ttft_p90, ttft_p99 = _percentiles("vllm:time_to_first_token_seconds")
    e2e_p50, e2e_p90, e2e_p99 = _percentiles("vllm:e2e_request_latency_seconds")

    prefix_hits = values.get("vllm:prefix_cache_hits_total", 0.0)
    prefix_queries = values.get("vllm:prefix_cache_queries_total", 0.0)
    prefix_cache_hit_rate = prefix_hits / prefix_queries if prefix_queries > 0 else None

    return VLLMMetrics(
        kv_cache_usage_perc=values.get("vllm:kv_cache_usage_perc", 0.0),
        num_requests_running=values.get("vllm:num_requests_running", 0.0),
        num_requests_waiting=values.get("vllm:num_requests_waiting", 0.0),
        prompt_tokens_total=values.get("vllm:prompt_tokens_total", 0.0),
        generation_tokens_total=values.get("vllm:generation_tokens_total", 0.0),
        ttft_avg_seconds=_avg("vllm:time_to_first_token_seconds"),
        request_success=request_success,
        cache_config=cache_config,
        ttft_p50_seconds=ttft_p50,
        ttft_p90_seconds=ttft_p90,
        ttft_p99_seconds=ttft_p99,
        e2e_p50_seconds=e2e_p50,
        e2e_p90_seconds=e2e_p90,
        e2e_p99_seconds=e2e_p99,
        queue_time_avg_seconds=_avg("vllm:request_queue_time_seconds"),
        inference_time_avg_seconds=_avg("vllm:request_inference_time_seconds"),
        num_preemptions_total=values.get("vllm:num_preemptions_total", 0.0),
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )


async def fetch_metrics(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text
