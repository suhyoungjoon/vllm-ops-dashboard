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

EXTENDED_SAMPLE_METRICS_TEXT = (
    SAMPLE_METRICS_TEXT
    + """
# HELP vllm:e2e_request_latency_seconds Histogram of end to end request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="5.0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 8.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 10.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 10.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 20.0
# HELP vllm:request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 8.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 4.0
# HELP vllm:request_inference_time_seconds Histogram of time spent in RUNNING phase for request.
# TYPE vllm:request_inference_time_seconds histogram
vllm:request_inference_time_seconds_count{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 8.0
vllm:request_inference_time_seconds_sum{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 16.0
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 3.0
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 100.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 80.0
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",enable_prefix_caching="True",gpu_memory_utilization="0.92",kv_cache_max_concurrency="39.78",model_name="Qwen/Qwen2.5-0.5B-Instruct",num_gpu_blocks="81469",prefix_caching_hash_algo="sha256"} 1.0
"""
)


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


def test_parse_metrics_computes_ttft_percentiles():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.ttft_p50_seconds == pytest.approx(0.1)
    assert metrics.ttft_p90_seconds == pytest.approx(0.1)
    assert metrics.ttft_p99_seconds == pytest.approx(0.1)


def test_parse_metrics_computes_e2e_latency_percentiles():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.e2e_p50_seconds == pytest.approx(3.0)
    assert metrics.e2e_p90_seconds == pytest.approx(5.0)
    assert metrics.e2e_p99_seconds == pytest.approx(5.0)


def test_parse_metrics_percentiles_none_without_histogram_samples():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.e2e_p50_seconds is None
    assert metrics.e2e_p90_seconds is None
    assert metrics.e2e_p99_seconds is None


def test_parse_metrics_computes_queue_and_inference_time_averages():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.queue_time_avg_seconds == pytest.approx(4.0 / 8.0)
    assert metrics.inference_time_avg_seconds == pytest.approx(16.0 / 8.0)


def test_parse_metrics_extracts_preemptions():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.num_preemptions_total == pytest.approx(3.0)


def test_parse_metrics_computes_prefix_cache_hit_rate():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.prefix_cache_hit_rate == pytest.approx(0.8)


def test_parse_metrics_prefix_cache_hit_rate_none_without_queries():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.prefix_cache_hit_rate is None


def test_parse_metrics_extracts_cache_config_labels():
    metrics = parse_metrics(EXTENDED_SAMPLE_METRICS_TEXT)
    assert metrics.cache_config["block_size"] == "16"
    assert metrics.cache_config["gpu_memory_utilization"] == "0.92"
    assert metrics.cache_config["num_gpu_blocks"] == "81469"


def test_parse_metrics_cache_config_none_when_absent():
    metrics = parse_metrics(SAMPLE_METRICS_TEXT)
    assert metrics.cache_config is None


async def test_fetch_metrics_gets_text_from_url():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=SAMPLE_METRICS_TEXT)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        text = await fetch_metrics(client, "http://mock/metrics")

    metrics = parse_metrics(text)
    assert metrics.kv_cache_usage_perc == pytest.approx(0.42)
