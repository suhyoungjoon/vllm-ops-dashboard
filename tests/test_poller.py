import httpx
import pytest

from backend.poller import fetch_metrics, parse_metrics

SAMPLE_METRICS_TEXT = """
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.42
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2000.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 10.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 10.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.5
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{engine="0",finished_reason="stop",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 7.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
vllm:request_success_total{engine="0",finished_reason="error",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0
"""


def test_parse_metrics_extracts_kv_cache_usage():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.kv_cache_usage_perc == pytest.approx(0.42)


def test_parse_metrics_extracts_running_and_waiting():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.num_requests_running == pytest.approx(3.0)
    assert metrics.num_requests_waiting == pytest.approx(1.0)


def test_parse_metrics_extracts_token_counters():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.prompt_tokens_total == pytest.approx(1000.0)
    assert metrics.generation_tokens_total == pytest.approx(2000.0)


def test_parse_metrics_computes_ttft_average():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.ttft_avg_seconds == pytest.approx(1.5 / 10.0)


def test_parse_metrics_ttft_average_none_when_no_samples():
    text_without_ttft = SAMPLE_METRICS_TEXT.replace(
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 10.0',
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0',
    )
    metrics = parse_metrics(text_without_ttft)
    assert metrics.ttft_avg_seconds is None


def test_parse_metrics_extracts_request_success_by_reason():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.request_success == {
        "stop": 7.0,
        "length": 2.0,
        "abort": 1.0,
        "error": 0.0,
    }


async def test_fetch_metrics_gets_text_from_url():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=SAMPLE_METRICS_TEXT)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        text = await fetch_metrics(client, "http://mock/metrics")

    metrics = parse_metrics(text)
    assert metrics.kv_cache_usage_perc == pytest.approx(0.42)
