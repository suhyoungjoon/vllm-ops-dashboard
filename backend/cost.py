"""토큰 사용량을 상용 API 요금과 비교해 환산 비용을 계산한다.

실제 절감액이나 청구액이 아니라 "이만큼 썼다면 상용 API였을 때 대략 이 정도
비용이었을 것"이라는 참고용 방향성 수치다. 단가를 모르면(환경변수 미설정)
비활성화된 상태로 반환한다.
"""

from dataclasses import dataclass


@dataclass
class CostEstimate:
    enabled: bool
    prompt_price_per_1m: float | None
    completion_price_per_1m: float | None
    estimated_cost_usd: float | None


def estimate_cost(
    prompt_tokens_total: float,
    generation_tokens_total: float,
    prompt_price_per_1m: float | None,
    completion_price_per_1m: float | None,
) -> CostEstimate:
    if prompt_price_per_1m is None or completion_price_per_1m is None:
        return CostEstimate(
            enabled=False,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            estimated_cost_usd=None,
        )

    estimated_cost_usd = (
        prompt_tokens_total / 1_000_000 * prompt_price_per_1m
        + generation_tokens_total / 1_000_000 * completion_price_per_1m
    )
    return CostEstimate(
        enabled=True,
        prompt_price_per_1m=prompt_price_per_1m,
        completion_price_per_1m=completion_price_per_1m,
        estimated_cost_usd=estimated_cost_usd,
    )
