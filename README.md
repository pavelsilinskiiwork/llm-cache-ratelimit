# LLM cache, rate limit and budget on Redis

[![tests](https://github.com/pavelsilinskiiwork/llm-cache-ratelimit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/pavelsilinskiiwork/llm-cache-ratelimit/actions/workflows/tests.yml)

A small FastAPI service that puts three Redis-backed controls in front of an LLM call:

1. **Response cache** with single-flight: the same question costs one model call, even when 20 arrive at once.
2. **Sliding-window rate limit** per client, correct across several API processes.
3. **Daily spending cap** shared by all clients, enforced with reserve-then-settle so concurrent requests cannot overshoot it.

```
POST /chat
  │
  ├─ rate limit   (every request, cache hits included)       → 429 + Retry-After
  ├─ cache lookup (key = hash of provider, model, system, message, max_tokens)
  │     hit        → answer from Redis, no model call, no budget used
  │     miss       → take a short lock (SET NX EX); others wait for this result
  │         ├─ reserve worst-case cost against the daily budget → 429 if over the cap
  │         ├─ call the model
  │         └─ settle to the real cost, store the answer with a TTL
  └─ response: text, source (hit | miss | coalesced | bypass), cost_usd, saved_usd, rate_limit
```

The default provider is a free, deterministic fake, so everything runs and tests without an API key.
`PROVIDER=groq` calls Groq's OpenAI-compatible API.

## Design decisions

| Decision | Why |
| --- | --- |
| Rate limit applies to cache hits | it protects the service, not only the model bill |
| Budget is touched only on a real model call | cached answers keep working after the day's budget is spent |
| Reserve worst-case cost first, settle after | "check spent < cap, then call" lets many in-flight requests pass together and overshoot |
| Money as integer micro-dollars | `INCRBY` is atomic and exact; floats drift |
| Limiter uses Redis `TIME`, one `MULTI/EXEC` | API servers with skewed clocks agree; no race between count and add |
| Rejected requests are removed from the window | being refused does not use up quota |
| Cache key includes provider, model, system prompt and `max_tokens`, plus a version prefix | a different prompt or model never gets someone else's answer; bump `CACHE_VERSION` to invalidate everything |
| Cache failure is a miss; limiter/budget failure is **503 by default** | cost protection first. `FAIL_OPEN=true` serves without limits instead |
| `X-Forwarded-For` ignored unless `TRUST_PROXY=true` | otherwise any client can pick its own identity and dodge the limit |

## Run it

```bash
docker compose up --build            # Redis + API on http://127.0.0.1:8000/docs
```

```bash
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is Redis?"}'
curl -s localhost:8000/stats
```

Without Docker: start any Redis, `pip install -r requirements-dev.txt`, `uvicorn llm_cache.app:app`.
Settings are environment variables, listed in `.env.example`.

Sample session with `RATE_LIMIT_REQUESTS=3` (real Redis, fake provider):

```
200 200 200 429        # the fourth request in the window is refused
retry-after: 60
{"error":"rate_limited","retry_after_seconds":60,"limit":3}

GET /stats
{"requests":5,"cache":{"hits":2,"misses":1,"hit_rate":0.6667,"saved_usd":5.8e-05},
 "rate_limited":2,"spent_usd_total":2.9e-05,"daily_budget_usd":1.0, ...}
```

## Tests

```bash
python -m pytest --cov=llm_cache                                   # fakeredis
REDIS_TEST_URL=redis://localhost:6379/15 python -m pytest          # also on a real Redis
python bench.py --redis-url redis://localhost:6379/15              # benchmark
```

78 tests, 96% coverage. Every Redis test runs on fakeredis **and** on a real Redis when `REDIS_TEST_URL` is
set (CI does both). Highlights: 20 identical concurrent requests make one model call; 30 concurrent requests
against a limit of 5 allow exactly 5; 40 concurrent reservations against a cap of 10 units grant exactly 10;
a failed model call releases its reservation; a dead Redis gives 503 (or serves, with `FAIL_OPEN`).

## Benchmark

The model is **simulated**: `FakeProvider` sleeps 300 ms per call. The numbers show what the cache removes,
not how fast a real model is. 300 requests over 40 prompts with skewed popularity, 20 concurrent, real
Redis 7 on one machine; costs use illustrative prices ($1 / $5 per million tokens).

| Scenario | Requests | Model calls | Wall time | p50 | Cost |
| --- | --- | --- | --- | --- | --- |
| No cache | 300 | 300 | 4.78 s | 315 ms | $0.00735 |
| With cache | 300 | 37 | 1.30 s | 5.7 ms | $0.00099 |
| 20 identical requests at once, no cache | 20 | 20 | 0.32 s | 316 ms | $0.00032 |
| 20 identical requests at once, single-flight | 20 | **1** | 0.33 s | 333 ms | $0.000016 |

Cache hit rate in the second row: 87.7%. Single-flight does not make the first answer faster; it stops 19
other callers paying for the same call.

## Limitations

- Exact match only. "What is Redis?" and "what's redis" are different keys. Semantic caching (embeddings) is a
  different tool with its own false-positive risk.
- Token counts for the budget *estimate* are about 4 characters per token; the settled cost uses the provider's
  real usage numbers.
- The lock release is compare-then-delete, not a Lua script. The lock has a TTL, so the worst case is one
  answer computed twice.
- A single Redis is a single point of failure; the fail-open/closed switch decides what that means for you.
- Budget day is UTC.
- The client identity is the IP address; add authentication for real per-user limits.
