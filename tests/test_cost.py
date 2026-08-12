import pytest

from backend.cost import estimate_cost


def test_disabled_when_prompt_price_missing():
    result = estimate_cost(1_000_000, 1_000_000, None, 1.0)
    assert result.enabled is False
    assert result.estimated_cost_usd is None


def test_disabled_when_completion_price_missing():
    result = estimate_cost(1_000_000, 1_000_000, 1.0, None)
    assert result.enabled is False
    assert result.estimated_cost_usd is None


def test_disabled_when_both_prices_missing():
    result = estimate_cost(1_000_000, 1_000_000, None, None)
    assert result.enabled is False


def test_computes_cost_from_token_totals_and_prices():
    # 2M prompt tokens @ $0.15/1M + 1M generation tokens @ $0.60/1M = 0.30 + 0.60 = 0.90
    result = estimate_cost(2_000_000, 1_000_000, 0.15, 0.60)
    assert result.enabled is True
    assert result.estimated_cost_usd == pytest.approx(0.90)


def test_zero_tokens_yields_zero_cost_when_enabled():
    result = estimate_cost(0, 0, 0.15, 0.60)
    assert result.enabled is True
    assert result.estimated_cost_usd == pytest.approx(0.0)


def test_prices_echoed_back_when_enabled():
    result = estimate_cost(0, 0, 0.15, 0.60)
    assert result.prompt_price_per_1m == 0.15
    assert result.completion_price_per_1m == 0.60
