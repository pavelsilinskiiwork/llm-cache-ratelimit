import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_cache.app import create_app
from llm_cache.providers import FakeProvider
from llm_cache.settings import Settings


@pytest.fixture
def make_client():
    def make(provider=None, redis=None, **overrides):
        settings = Settings(**overrides)
        redis = redis or fakeredis.FakeAsyncRedis(decode_responses=True)
        return TestClient(create_app(settings, redis, provider or FakeProvider()))

    return make


def test_chat_miss_then_hit(make_client):
    client = make_client()
    a = client.post("/chat", json={"message": "hello"}).json()
    b = client.post("/chat", json={"message": "hello"}).json()
    assert (a["source"], a["cached"]) == ("miss", False)
    assert (b["source"], b["cached"]) == ("hit", True)
    assert b["saved_usd"] == a["cost_usd"] > 0
    assert b["rate_limit"]["remaining"] == 8


def test_rate_limit_returns_429_with_retry_after(make_client):
    client = make_client(rate_limit_requests=1)
    assert client.post("/chat", json={"message": "x"}).status_code == 200
    r = client.post("/chat", json={"message": "x"})
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limited"
    assert int(r.headers["Retry-After"]) >= 1


def test_budget_exhausted_returns_429(make_client):
    client = make_client(daily_budget_usd=0.0)
    r = client.post("/chat", json={"message": "x"})
    assert r.status_code == 429 and r.json()["error"] == "daily_budget_exhausted"


def test_provider_failure_is_502(make_client):
    client = make_client(provider=FakeProvider(fail_times=1))
    assert client.post("/chat", json={"message": "x"}).status_code == 502


def test_validation(make_client):
    client = make_client()
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x", "max_tokens": 5000}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_clients_have_separate_limits_behind_a_trusted_proxy(make_client):
    client = make_client(rate_limit_requests=1, trust_proxy=True)
    a = {"X-Forwarded-For": "1.1.1.1"}
    b = {"X-Forwarded-For": "2.2.2.2"}
    assert client.post("/chat", json={"message": "x"}, headers=a).status_code == 200
    assert client.post("/chat", json={"message": "x"}, headers=b).status_code == 200
    assert client.post("/chat", json={"message": "x"}, headers=a).status_code == 429


def test_forwarded_header_is_ignored_unless_trusted(make_client):
    client = make_client(rate_limit_requests=1)
    assert (
        client.post("/chat", json={"message": "x"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    )
    assert (
        client.post("/chat", json={"message": "x"}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    )


def test_stats_and_health(make_client):
    client = make_client()
    client.post("/chat", json={"message": "x"})
    assert client.get("/stats").json()["requests"] == 1
    assert client.get("/health").json() == {"status": "ok", "redis": True}


def test_redis_down_is_503_and_health_degraded(make_client, broken_redis):
    client = make_client(redis=broken_redis)
    assert client.post("/chat", json={"message": "x"}).status_code == 503
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["redis"] is False
