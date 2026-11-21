import asyncio

from llm_cache.ratelimit import SlidingWindowLimiter


class Clock:
    def __init__(self):
        self.ms = 1_000_000

    def __call__(self):
        return self.ms


async def test_allows_up_to_the_limit_then_rejects(redis_client):
    clock = Clock()
    limiter = SlidingWindowLimiter(redis_client, 3, 60, clock=clock)
    decisions = [await limiter.allow("a") for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0]


async def test_rejected_requests_do_not_use_quota(redis_client):
    clock = Clock()
    limiter = SlidingWindowLimiter(redis_client, 2, 60, clock=clock)
    for _ in range(10):
        await limiter.allow("a")
    assert await redis_client.zcard("rl:a") == 2


async def test_retry_after_counts_down_to_the_oldest_request_leaving_the_window(redis_client):
    clock = Clock()
    limiter = SlidingWindowLimiter(redis_client, 1, 60, clock=clock)
    await limiter.allow("a")
    clock.ms += 20_000
    denied = await limiter.allow("a")
    assert not denied.allowed and denied.retry_after == 40


async def test_window_slides(redis_client):
    clock = Clock()
    limiter = SlidingWindowLimiter(redis_client, 2, 60, clock=clock)
    await limiter.allow("a")
    clock.ms += 30_000
    await limiter.allow("a")
    assert not (await limiter.allow("a")).allowed
    clock.ms += 31_000  # the first request is now 61 s old, the second 31 s
    assert (await limiter.allow("a")).allowed
    assert not (await limiter.allow("a")).allowed


async def test_clients_are_independent(redis_client):
    clock = Clock()
    limiter = SlidingWindowLimiter(redis_client, 1, 60, clock=clock)
    assert (await limiter.allow("a")).allowed
    assert (await limiter.allow("b")).allowed
    assert not (await limiter.allow("a")).allowed


async def test_concurrent_requests_never_exceed_the_limit(redis_client):
    limiter = SlidingWindowLimiter(redis_client, 5, 60)  # real Redis clock
    decisions = await asyncio.gather(*[limiter.allow("a") for _ in range(30)])
    assert sum(d.allowed for d in decisions) == 5


async def test_key_expires_when_idle(redis_client):
    limiter = SlidingWindowLimiter(redis_client, 5, 60, clock=Clock())
    await limiter.allow("a")
    assert 0 < await redis_client.pttl("rl:a") <= 60_000
