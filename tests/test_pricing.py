from llm_cache.pricing import Prices, cost_micro, estimate_cost_micro, usd


def test_cost_is_integer_micro_dollars():
    prices = Prices(1.0, 5.0)
    assert cost_micro(prices, 1_000_000, 0) == 1_000_000
    assert cost_micro(prices, 0, 1_000_000) == 5_000_000
    assert usd(1_500_000) == 1.5


def test_estimate_is_a_worst_case_that_covers_a_full_answer():
    prices = Prices(1.0, 5.0)
    estimate = estimate_cost_micro(prices, "sys", "hello world", 1000)
    assert estimate >= cost_micro(prices, 0, 1000)
