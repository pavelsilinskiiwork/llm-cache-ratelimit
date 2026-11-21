"""Counters in a Redis hash, so every API process contributes to the same numbers."""

import logging

from .cache import REDIS_ERRORS

log = logging.getLogger("llm_cache")
KEY = "llm_cache:stats"


class Metrics:
    def __init__(self, redis):
        self.redis = redis

    async def incr(self, **fields: int) -> None:
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                for name, amount in fields.items():
                    if amount:
                        pipe.hincrby(KEY, name, amount)
                await pipe.execute()
        except REDIS_ERRORS as e:
            log.warning("metrics_failed %s", type(e).__name__)  # losing a counter must never fail a request

    async def snapshot(self) -> dict[str, int]:
        try:
            return {k: int(v) for k, v in (await self.redis.hgetall(KEY)).items()}
        except REDIS_ERRORS:
            return {}
