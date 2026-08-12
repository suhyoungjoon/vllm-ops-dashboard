"""수집된 지표를 규칙 기반으로 해석해 지금 무엇이 병목인지 텍스트로 진단한다.

액션 제안(튜닝 방법 등)은 포함하지 않는다 — "지금 뭐가 문제인가"만 말하는 순수 진단 단계.
"""

from dataclasses import dataclass

from backend.capacity import CapacityEstimate
from backend.poller import VLLMMetrics

KV_CACHE_PRESSURE_THRESHOLD = 0.85
QUEUE_BOTTLENECK_MIN_SECONDS = 0.05
LOW_PREFIX_CACHE_HIT_RATE = 0.5
SATURATION_WARNING_MINUTES = 5.0


@dataclass
class Diagnosis:
    level: str  # "warning" | "info" | "ok"
    message: str
    recommendation: str | None = None


def diagnose(
    metrics: VLLMMetrics,
    preemptions_delta: float = 0.0,
    capacity: CapacityEstimate | None = None,
) -> list[Diagnosis]:
    findings: list[Diagnosis] = []

    if (
        metrics.queue_time_avg_seconds is not None
        and metrics.inference_time_avg_seconds is not None
        and metrics.queue_time_avg_seconds > metrics.inference_time_avg_seconds
        and metrics.queue_time_avg_seconds > QUEUE_BOTTLENECK_MIN_SECONDS
    ):
        findings.append(
            Diagnosis(
                level="warning",
                message=(
                    f"대기열 병목 — 평균 대기시간({metrics.queue_time_avg_seconds * 1000:.0f}ms)이 "
                    f"평균 처리시간({metrics.inference_time_avg_seconds * 1000:.0f}ms)보다 깁니다."
                ),
                recommendation="처리 용량 대비 요청이 밀리고 있습니다. max_num_seqs 상향이나 replica 추가를 검토하세요.",
            )
        )

    if metrics.kv_cache_usage_perc > KV_CACHE_PRESSURE_THRESHOLD:
        findings.append(
            Diagnosis(
                level="warning",
                message=f"메모리 압박 — KV 캐시 사용률이 {metrics.kv_cache_usage_perc * 100:.0f}%로 높습니다.",
                recommendation="gpu_memory_utilization 여유를 늘리거나 동시 요청 수 제한을 검토하세요.",
            )
        )

    if preemptions_delta > 0:
        findings.append(
            Diagnosis(
                level="warning",
                message=f"선점 발생 — 최근 폴링 사이 {preemptions_delta:.0f}건의 요청이 선점되었습니다.",
                recommendation="gpu_memory_utilization을 낮추거나 max_model_len을 줄이는 것을 검토하세요.",
            )
        )

    if metrics.prefix_cache_hit_rate is not None and metrics.prefix_cache_hit_rate < LOW_PREFIX_CACHE_HIT_RATE:
        findings.append(
            Diagnosis(
                level="info",
                message=f"프리픽스 캐시 적중률 낮음 — 현재 {metrics.prefix_cache_hit_rate * 100:.0f}%입니다.",
                recommendation="프롬프트 앞부분을 공통 템플릿화하면 캐시 적중률을 높일 수 있습니다.",
            )
        )

    if (
        capacity is not None
        and capacity.minutes_to_saturation is not None
        and capacity.minutes_to_saturation <= SATURATION_WARNING_MINUTES
    ):
        findings.append(
            Diagnosis(
                level="warning",
                message=(
                    f"포화 임박 — 현재 추세면 약 {capacity.minutes_to_saturation:.1f}분 후 "
                    "KV 캐시가 포화 상태에 도달할 수 있습니다."
                ),
                recommendation="트래픽이 몰리는 시간대라면 동시 처리 용량을 미리 늘리는 것을 검토하세요.",
            )
        )

    if not findings:
        findings.append(Diagnosis(level="ok", message="특이 신호 없음 — 현재 병목 징후가 감지되지 않았습니다."))

    return findings
