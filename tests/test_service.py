import asyncio

import pytest

from llm_cache.providers import FakeProvider, ProviderError
from llm_cache.service import (
    BudgetExhausted,
    ChatRequest,
    ChatService,
    RateLimited,
    StorageUnavailable,
)
from llm_cache.settings import Settings


def make(redis, provider=None, **overrides):
    settings = Settings(**{"rate_limit_requests": 100, **overrides})
    provider = provider or FakeProvider()
    return ChatService(redis, provider, settings, today=lambda: "2026-01-01"), provider


async def test_second_identical_request_is_a_hit_and_costs_nothing(redis_client):
    service, provider = make(redis_client)
    first = await service.chat("c", ChatRequest("hello"))
    second = await service.chat("c", ChatRequest("hello"))
    assert (first.source, second.source) == ("miss", "hit")
    assert provider.calls == 1
    assert first.cost_usd > 0 and first.saved_usd == 0
    assert second.cost_usd == 0 and second.saved_usd == first.cost_usd
    assert second.text == first.text


async def test_cache_false_bypasses_the_cache(redis_client):
    service, provider = make(redis_client)
    await service.chat("c", ChatRequest("hello", cache=False))
    again = await service.chat("c", ChatRequest("hello", cache=False))
    assert again.source == "bypass" and provider.calls == 2


async def test_different_system_prompt_is_a_different_entry(redis_client):
    service, provider = make(redis_client)
    await service.chat("c", ChatRequest("hello", system="a"))
    other = await service.chat("c", ChatRequest("hello", system="b"))
    assert other.source == "miss" and provider.calls == 2


async def test_rate_limit_applies_to_cache_hits_too(redis_client):
    service, _ = make(redis_client, rate_limit_requests=2)
    await service.chat("c", ChatRequest("hello"))
    await service.chat("c", ChatRequest("hello"))
    with pytest.raises(RateLimited) as info:
        await service.chat("c", ChatRequest("hello"))
    assert info.value.retry_after >= 1


async def test_budget_stops_model_calls_but_cached_answers_still_work(redis_client):
    service, provider = make(redis_client, daily_budget_usd=0.0005)
    first = await service.chat("c", ChatRequest("hello", max_tokens=10))
    assert first.source == "miss"
    with pytest.raises(BudgetExhausted):
        await service.chat("c", ChatRequest("something else " * 200, max_tokens=1000))
    again = await service.chat("c", ChatRequest("hello", max_tokens=10))
    assert again.source == "hit"
    assert provider.calls == 1


async def test_provider_failure_releases_the_reservation(redis_client):
    service, _ = make(redis_client, provider=FakeProvider(fail_times=1))
    with pytest.raises(ProviderError):
        await service.chat("c", ChatRequest("hello"))
    assert await service.budget.spent() == 0
    ok = await service.chat("c", ChatRequest("hello"))
    assert ok.source == "miss"


async def test_concurrent_identical_requests_make_one_model_call(redis_client):
    service, provider = make(redis_client, provider=FakeProvider(latency=0.1))
    service.cache.poll = 0.005
    results = await asyncio.gather(*[service.chat("c", ChatRequest("same")) for _ in range(15)])
    assert provider.calls == 1
    assert sorted(r.source for r in results).count("coalesced") == 14
    stats = await service.stats()
    assert stats["cache"]["misses"] == 1 and stats["cache"]["coalesced"] == 14


async def test_stats_add_up(redis_client):
    service, _ = make(redis_client)
    await service.chat("c", ChatRequest("a"))
    await service.chat("c", ChatRequest("a"))
    await service.chat("c", ChatRequest("a"))
    stats = await service.stats()
    assert stats["requests"] == 3
    assert stats["cache"]["hits"] == 2 and stats["cache"]["misses"] == 1
    assert stats["cache"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert stats["cache"]["saved_usd"] == pytest.approx(2 * stats["spent_usd_total"])
    assert stats["spent_usd_today"] == stats["spent_usd_total"]


async def test_redis_down_refuses_by_default(broken_redis):
    service, provider = make(broken_redis)
    with pytest.raises(StorageUnavailable):
        await service.chat("c", ChatRequest("hello"))
    assert provider.calls == 0


async def test_redis_down_with_fail_open_still_answers(broken_redis):
    service, provider = make(broken_redis, fail_open=True)
    result = await service.chat("c", ChatRequest("hello"))
    assert result.source == "miss" and result.rate_remaining is None and provider.calls == 1
