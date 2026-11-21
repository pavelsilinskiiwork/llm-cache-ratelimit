import os

import fakeredis
import pytest
import pytest_asyncio
from redis.asyncio import Redis

SETTING_VARS = [
    "REDIS_URL", "PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "CACHE_TTL_SECONDS", "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS", "DAILY_BUDGET_USD", "PRICE_INPUT_PER_MTOK", "PRICE_OUTPUT_PER_MTOK",
    "FAIL_OPEN", "TRUST_PROXY",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in SETTING_VARS:
        monkeypatch.delenv(name, raising=False)


def _backends():
    backends = ["fakeredis"]
    if os.getenv("REDIS_TEST_URL"):
        backends.append("redis")
    return backends


@pytest_asyncio.fixture(params=_backends())
async def redis_client(request):
    """Every Redis-facing test runs on fakeredis, and on a real Redis when REDIS_TEST_URL is set."""
    if request.param == "fakeredis":
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
    else:
        client = Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


class BrokenRedis:
    """Every operation raises, like a Redis that is down."""

    def __getattr__(self, name):
        from redis.exceptions import ConnectionError as RedisConnectionError

        async def fail(*args, **kwargs):
            raise RedisConnectionError("down")

        if name == "pipeline":

            def pipeline(*a, **k):
                raise RedisConnectionError("down")

            return pipeline
        return fail


@pytest.fixture
def broken_redis():
    return BrokenRedis()
