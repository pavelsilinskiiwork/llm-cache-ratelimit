import asyncio
import time

from llm_cache.cache import CacheEntry, LLMCache, cache_key


def entry(text="x"):
    return CacheEntry(text, 5, 7, 123, "m", "p", time.time())


def test_key_changes_with_every_input():
    base = cache_key("p", "m", "sys", "hi", 100)
    assert base == cache_key("p", "m", "sys", "hi", 100)
    for other in [("q", "m", "sys", "hi", 100), ("p", "n", "sys", "hi", 100), ("p", "m", "s2", "hi", 100),
                  ("p", "m", "sys", "ho", 100), ("p", "m", "sys", "hi", 101)]:  # fmt: skip
        assert cache_key(*other) != base


def test_key_does_not_confuse_field_boundaries():
    assert cache_key("p", "m", "ab", "c", 1) != cache_key("p", "m", "a", "bc", 1)


async def test_miss_then_hit(redis_client):
    cache = LLMCache(redis_client, 60)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return entry("answer")

    first, source1 = await cache.get_or_compute("k", compute)
    second, source2 = await cache.get_or_compute("k", compute)
    assert (source1, source2, calls) == ("miss", "hit", 1)
    assert first.text == second.text == "answer"


async def test_entry_expires_with_ttl(redis_client):
    cache = LLMCache(redis_client, 60)
    await cache.set("k", entry())
    ttl = await redis_client.ttl("k")
    assert 0 < ttl <= 60


async def test_single_flight_one_model_call_for_many_identical_requests(redis_client):
    cache = LLMCache(redis_client, 60, poll=0.005)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return entry("shared")

    results = await asyncio.gather(*[cache.get_or_compute("k", compute) for _ in range(20)])
    sources = sorted(s for _, s in results)
    assert calls == 1
    assert sources.count("miss") == 1 and sources.count("coalesced") == 19
    assert {e.text for e, _ in results} == {"shared"}


async def test_lock_is_released_when_the_leader_fails(redis_client):
    cache = LLMCache(redis_client, 60)

    async def boom():
        raise RuntimeError("model down")

    try:
        await cache.get_or_compute("k", boom)
    except RuntimeError:
        pass
    assert not await redis_client.exists("k:lock")

    async def ok():
        return entry("recovered")

    result, source = await cache.get_or_compute("k", ok)
    assert (result.text, source) == ("recovered", "miss")


async def test_follower_computes_itself_when_the_leader_dies_without_a_result(redis_client):
    cache = LLMCache(redis_client, 60, poll=0.005, wait_timeout=1)
    await redis_client.set("k:lock", "ghost", ex=1)  # a leader that will never deliver

    async def compute():
        return entry("mine")

    task = asyncio.create_task(cache.get_or_compute("k", compute))
    await asyncio.sleep(0.02)
    await redis_client.delete("k:lock")  # the ghost's lock vanishes (expired)
    result, source = await asyncio.wait_for(task, 2)
    assert (result.text, source) == ("mine", "miss")


async def test_corrupt_entry_is_treated_as_a_miss(redis_client):
    cache = LLMCache(redis_client, 60)
    await redis_client.set("k", "not json")

    async def compute():
        return entry("fresh")

    result, source = await cache.get_or_compute("k", compute)
    assert (result.text, source) == ("fresh", "miss")


async def test_redis_down_means_miss_not_error(broken_redis):
    cache = LLMCache(broken_redis, 60)

    async def compute():
        return entry("direct")

    result, source = await cache.get_or_compute("k", compute)
    assert (result.text, source) == ("direct", "miss")
