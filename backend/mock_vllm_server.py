"""개발용 가짜 vLLM /metrics 서버.

실제 vLLM(CUDA GPU) 없이 대시보드를 개발하기 위해, vllm_metrics_sample.txt에서
확인한 실제 지표 이름/라벨/HELP 문구를 그대로 흉내내는 Prometheus 텍스트 포맷
/metrics 엔드포인트를 제공한다. 값 자체의 정확도는 중요하지 않고, 파서 호환성과
UI 시나리오(정상 상태 / KV 캐시 경고 스파이크) 재현이 목적이다.
"""

import asyncio
import math
import random
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

ENGINE = "0"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = {"engine": ENGINE, "model_name": MODEL_NAME}

UPDATE_INTERVAL_SECONDS = 1.0
MAX_CONCURRENCY = 32
KV_CACHE_SPIKE_PROBABILITY = 0.05

# vllm_metrics_sample.txt의 time_to_first_token_seconds 버킷 경계값을 그대로 사용
TTFT_BUCKETS = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
)

registry = CollectorRegistry()

kv_cache_usage_perc = Gauge(
    "vllm:kv_cache_usage_perc",
    "KV-cache usage. 1 means 100 percent usage.",
    ["engine", "model_name"],
    registry=registry,
)
num_requests_running = Gauge(
    "vllm:num_requests_running",
    "Number of requests in model execution batches.",
    ["engine", "model_name"],
    registry=registry,
)
num_requests_waiting = Gauge(
    "vllm:num_requests_waiting",
    "Number of requests waiting to be processed.",
    ["engine", "model_name"],
    registry=registry,
)
prompt_tokens_total = Counter(
    "vllm:prompt_tokens_total",
    "Number of prefill tokens processed.",
    ["engine", "model_name"],
    registry=registry,
)
generation_tokens_total = Counter(
    "vllm:generation_tokens_total",
    "Number of generation tokens processed.",
    ["engine", "model_name"],
    registry=registry,
)
time_to_first_token_seconds = Histogram(
    "vllm:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    ["engine", "model_name"],
    buckets=TTFT_BUCKETS,
    registry=registry,
)
request_success_total = Counter(
    "vllm:request_success_total",
    "Count of successfully processed requests.",
    ["engine", "model_name", "finished_reason"],
    registry=registry,
)

_kv_gauge = kv_cache_usage_perc.labels(**LABELS)
_running_gauge = num_requests_running.labels(**LABELS)
_waiting_gauge = num_requests_waiting.labels(**LABELS)
_prompt_counter = prompt_tokens_total.labels(**LABELS)
_generation_counter = generation_tokens_total.labels(**LABELS)
_ttft_histogram = time_to_first_token_seconds.labels(**LABELS)
_success_counters = {
    reason: request_success_total.labels(**LABELS, finished_reason=reason)
    for reason in ("stop", "length", "abort", "error")
}
_finished_reason_weights = {"stop": 0.85, "length": 0.1, "abort": 0.03, "error": 0.02}

_start_time = time.monotonic()


def _tick() -> None:
    """한 폴링 주기만큼 지표를 진행시킨다 (사인파 + 노이즈 기반 시뮬레이션)."""
    elapsed = time.monotonic() - _start_time

    if random.random() < KV_CACHE_SPIKE_PROBABILITY:
        kv_usage = random.uniform(0.9, 1.0)
    else:
        wave = (math.sin(elapsed / 30.0) + 1) / 2  # 60초 주기로 0~1 오가는 사인파
        noise = random.uniform(-0.05, 0.05)
        kv_usage = min(max(wave + noise, 0.0), 1.0)
    _kv_gauge.set(kv_usage)

    running = round(kv_usage * MAX_CONCURRENCY + random.uniform(-2, 2))
    running = min(max(running, 0), MAX_CONCURRENCY)
    _running_gauge.set(running)

    waiting = max(0, running - MAX_CONCURRENCY + random.randint(0, 5)) if running >= MAX_CONCURRENCY - 3 else 0
    _waiting_gauge.set(waiting)

    active_requests = max(running, 1)
    _prompt_counter.inc(random.uniform(0, 20) * active_requests)
    _generation_counter.inc(random.uniform(5, 15) * active_requests)

    completed = max(0, round(random.gauss(active_requests * 0.3, 1)))
    for _ in range(completed):
        _ttft_histogram.observe(max(0.001, random.lognormvariate(math.log(0.15), 0.6)))
        reason = random.choices(
            list(_finished_reason_weights.keys()),
            weights=list(_finished_reason_weights.values()),
        )[0]
        _success_counters[reason].inc()


async def _update_loop() -> None:
    while True:
        _tick()
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_update_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
