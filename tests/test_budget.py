import asyncio

from llm_cache.budget import DailyBudget


async def test_reserve_within_cap_and_refuse_over_it(redis_client):
    budget = DailyBudget(redis_client, 1000, today=lambda: "2026-01-01")
    assert await budget.reserve(600)
    assert not await budget.reserve(600)
    assert await budget.spent() == 600  # the refused reservation was rolled back
    assert await budget.reserve(400)


async def test_settle_adjusts_to_the_real_cost(redis_client):
    budget = DailyBudget(redis_client, 1000, today=lambda: "2026-01-01")
    await budget.reserve(500)
    await budget.settle(500, 120)
    assert await budget.spent() == 120
    await budget.reserve(300)
    await budget.settle(300, 350)  # the call cost more than reserved
    assert await budget.spent() == 470


async def test_release_returns_the_reservation(redis_client):
    budget = DailyBudget(redis_client, 1000, today=lambda: "2026-01-01")
    await budget.reserve(500)
    await budget.release(500)
    assert await budget.spent() == 0


async def test_a_new_day_starts_at_zero(redis_client):
    day = ["2026-01-01"]
    budget = DailyBudget(redis_client, 1000, today=lambda: day[0])
    await budget.reserve(900)
    day[0] = "2026-01-02"
    assert await budget.spent() == 0
    assert await budget.reserve(900)


async def test_concurrent_reservations_never_overshoot_the_cap(redis_client):
    budget = DailyBudget(redis_client, 1000, today=lambda: "2026-01-01")
    results = await asyncio.gather(*[budget.reserve(100) for _ in range(40)])
    assert sum(results) == 10
    assert await budget.spent() == 1000


async def test_key_has_an_expiry(redis_client):
    budget = DailyBudget(redis_client, 1000, today=lambda: "2026-01-01")
    await budget.reserve(1)
    assert await redis_client.ttl("budget:2026-01-01") > 0
