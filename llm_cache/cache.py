"""Exact-match response cache for LLM calls, with single-flight.

Key: a hash of everything that changes the answer (provider, model, system prompt, message,
max_tokens). The same question asked twice costs one model call.

Single-flight: when many identical requests arrive while the first is still running, only one
(the leader) calls the model. The others wait for its result instead of each paying for the same
call (a "cache stampede"). The leader takes a short-lived lock with SET NX EX.

Redis trouble never breaks a request here: a failing cache is just a miss (fail open).
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from redis.exceptions import RedisError

log = logging.getLogger("llm_cache")

CACHE_VERSION = "v1"  # bump to invalidate every cached answer at once
REDIS_ERRORS = (RedisError, OSError)


def cache_key(provider: str, model: str, system: str, message: str, max_tokens: int) -> str:
    payload = json.dumps(
        [provider, model, system, message, max_tokens], ensure_ascii=False, separators=(",", ":")
    )
    return f"llmcache:{CACHE_VERSION}:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    text: str
    input_tokens: int
    output_tokens: int
    cost_micro: int  # what the original model call cost, so a hit can report what it saved
    model: str
    provider: str
    created_at: float

    def dumps(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def loads(cls, raw: str) -> "CacheEntry":
        return cls(**json.loads(raw))


class LLMCache:
    def __init__(
        self, redis, ttl_seconds: int, lock_ttl: int = 30, wait_timeout: float = 30, poll: float = 0.05
    ):
        self.redis = redis
        self.ttl = ttl_seconds
        self.lock_ttl = lock_ttl
        self.wait_timeout = wait_timeout
        self.poll = poll

    async def get(self, key: str) -> CacheEntry | None:
        try:
            raw = await self.redis.get(key)
        except REDIS_ERRORS as e:
            log.warning("cache_get_failed %s", type(e).__name__)
            return None
        if raw is None:
            return None
        try:
            return CacheEntry.loads(raw)
        except (ValueError, TypeError, KeyError):
            log.warning("cache_entry_corrupt, treating as a miss")
            return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        try:
            await self.redis.set(key, entry.dumps(), ex=self.ttl)
        except REDIS_ERRORS as e:
            log.warning("cache_set_failed %s", type(e).__name__)

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[CacheEntry]]
    ) -> tuple[CacheEntry, str]:
        """Returns (entry, source) where source is "hit", "miss" (we called the model) or "coalesced"
        (another request called it and we reused the result)."""
        cached = await self.get(key)
        if cached is not None:
            return cached, "hit"

        lock_key, token = f"{key}:lock", uuid.uuid4().hex
        try:
            is_leader = bool(await self.redis.set(lock_key, token, nx=True, ex=self.lock_ttl))
        except REDIS_ERRORS as e:
            log.warning("cache_lock_failed %s", type(e).__name__)
            return await compute(), "miss"

        if is_leader:
            try:
                entry = await compute()
                await self.set(key, entry)
                return entry, "miss"
            finally:
                await self._release(lock_key, token)

        # Someone else is already computing this answer: wait for it.
        deadline = time.monotonic() + self.wait_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll)
            cached = await self.get(key)
            if cached is not None:
                return cached, "coalesced"
            try:
                if not await self.redis.exists(lock_key):
                    break  # the leader finished without a result (it failed): do the work ourselves
            except REDIS_ERRORS:
                break
        entry = await compute()
        await self.set(key, entry)
        return entry, "miss"

    async def _release(self, lock_key: str, token: str) -> None:
        # Delete the lock only if it is still ours. Not atomic (a Lua script would be), but the lock
        # has a TTL and the worst case is a second leader recomputing one answer.
        try:
            if await self.redis.get(lock_key) == token:
                await self.redis.delete(lock_key)
        except REDIS_ERRORS:
            pass
