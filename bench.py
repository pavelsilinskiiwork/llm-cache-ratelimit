"""Benchmark the cache against a *simulated* model (FakeProvider with fixed latency).

    python bench.py --redis-url redis://localhost:6379/15

The model is fake: its latency is a number you pass in, so the absolute times only show what the
cache removes, not how fast a real model is. Costs use the illustrative prices in Settings.
"""

import argparse
import asyncio
import json
import random
import time

from redis.asyncio import Redis

from llm_cache.providers import FakeProvider
from llm_cache.service import ChatRequest, ChatService
from llm_cache.settings import Settings


async def run(redis, label, prompts, concurrency, latency, use_cache):
    await redis.flushdb()
    settings = Settings(rate_limit_requests=1_000_000, daily_budget_usd=1000.0)
    provider = FakeProvider(latency=latency)
    service = ChatService(redis, provider, settings)
    gate = asyncio.Semaphore(concurrency)
    durations = []

    async def one(prompt):
        async with gate:
            started = time.perf_counter()
            await service.chat("bench", ChatRequest(prompt, cache=use_cache))
            durations.append(time.perf_counter() - started)

    started = time.perf_counter()
    await asyncio.gather(*[one(p) for p in prompts])
    wall = time.perf_counter() - started
    stats = await service.stats()
    durations.sort()
    return {
        "scenario": label,
        "requests": len(prompts),
        "model_calls": provider.calls,
        "wall_seconds": round(wall, 2),
        "p50_ms": round(durations[len(durations) // 2] * 1000, 1),
        "p95_ms": round(durations[int(len(durations) * 0.95) - 1] * 1000, 1),
        "hit_rate": stats["cache"]["hit_rate"],
        "spent_usd": stats["spent_usd_total"],
        "saved_usd": stats["cache"]["saved_usd"],
    }


def workload(requests, distinct, seed=7):
    """Skewed traffic: a few popular questions asked often, many asked once or twice."""
    rng = random.Random(seed)
    weights = [1 / (rank + 1) for rank in range(distinct)]
    return [f"question {i}" for i in rng.choices(range(distinct), weights, k=requests)]


async def main(args):
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    try:
        prompts = workload(args.requests, args.distinct)
        rows = [
            await run(redis, "no cache", prompts, args.concurrency, args.latency, use_cache=False),
            await run(redis, "with cache", prompts, args.concurrency, args.latency, use_cache=True),
            await run(
                redis,
                "stampede: same prompt, no cache",
                ["same"] * args.concurrency,
                args.concurrency,
                args.latency,
                use_cache=False,
            ),  # fmt: skip
            await run(
                redis,
                "stampede: same prompt, single-flight",
                ["same"] * args.concurrency,
                args.concurrency,
                args.latency,
                use_cache=True,
            ),  # fmt: skip
        ]
    finally:
        await redis.flushdb()
        await redis.aclose()
    if args.json:
        print(json.dumps(rows, indent=2))
        return rows
    print(f"simulated model latency {args.latency * 1000:.0f} ms, concurrency {args.concurrency}\n")
    header = [
        "scenario",
        "requests",
        "model_calls",
        "wall_seconds",
        "p50_ms",
        "p95_ms",
        "hit_rate",
        "spent_usd",
    ]
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(str(r[h]) for h in header))
    return rows


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--distinct", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--latency", type=float, default=0.3, help="simulated model latency, seconds")
    parser.add_argument("--json", action="store_true")
    return asyncio.run(main(parser.parse_args(argv)))


if __name__ == "__main__":
    cli()
