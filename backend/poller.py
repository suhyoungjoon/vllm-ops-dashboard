"""vLLM `/metrics` 폴링 + Prometheus 텍스트 포맷 파싱.

여러 engine 라벨(데이터 병렬 등)이 존재할 수 있다는 가정 하에, gauge/counter는
모든 샘플을 합산하고 히스토그램은 sum/count를 각각 합산한 뒤 나눠서 평균을 낸다.
"""

from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

GAUGE_NAMES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")
COUNTER_NAMES = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")
TTFT_SUM_NAME = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT_NAME = "vllm:time_to_first_token_seconds_count"
REQUEST_SUCCESS_NAME = "vllm:request_success_total"


@dataclass
class VLLMMetrics:
    kv_cache_usage_perc: float
    num_requests_running: float
    num_requests_waiting: float
    prompt_tokens_total: float
    generation_tokens_total: float
    ttft_avg_seconds: float | None
    request_success: dict[str, float]


def parse_metrics(text: str) -> VLLMMetrics:
    values: dict[str, float] = {}
    request_success: dict[str, float] = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in GAUGE_NAMES or sample.name in COUNTER_NAMES:
                values[sample.name] = values.get(sample.name, 0.0) + sample.value
            elif sample.name in (TTFT_SUM_NAME, TTFT_COUNT_NAME):
                values[sample.name] = values.get(sample.name, 0.0) + sample.value
            elif sample.name == REQUEST_SUCCESS_NAME:
                reason = sample.labels.get("finished_reason", "unknown")
                request_success[reason] = request_success.get(reason, 0.0) + sample.value

    ttft_count = values.get(TTFT_COUNT_NAME, 0.0)
    ttft_avg_seconds = values.get(TTFT_SUM_NAME, 0.0) / ttft_count if ttft_count > 0 else None

    return VLLMMetrics(
        kv_cache_usage_perc=values.get("vllm:kv_cache_usage_perc", 0.0),
        num_requests_running=values.get("vllm:num_requests_running", 0.0),
        num_requests_waiting=values.get("vllm:num_requests_waiting", 0.0),
        prompt_tokens_total=values.get("vllm:prompt_tokens_total", 0.0),
        generation_tokens_total=values.get("vllm:generation_tokens_total", 0.0),
        ttft_avg_seconds=ttft_avg_seconds,
        request_success=request_success,
    )


async def fetch_metrics(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text
