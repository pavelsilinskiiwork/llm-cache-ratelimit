"""Sliding-window rate limiter on a Redis sorted set.

Each request is a member scored by its timestamp. One MULTI/EXEC transaction drops members older
than the window, adds this request and counts. If the count is over the limit the request is removed
again, so rejected requests do not use up quota. Because the state lives in Redis, the limit holds
across several API processes, and the clock is Redis's own (TIME), so servers with skewed clocks
agree.
"""

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    remaining: int
    retry_after: int  # seconds until a request would be allowed; 0 when allowed
    limit: int


class SlidingWindowLimiter:
    def __init__(self, redis, limit: int, window_seconds: int, clock: Callable[[], int] | None = None):
        self.redis = redis
        self.limit = limit
        self.window_ms = window_seconds * 1000
        self.clock = clock  # tests inject a fake clock returning milliseconds

    async def _now_ms(self) -> int:
        if self.clock is not None:
            return self.clock()
        seconds, microseconds = await self.redis.time()
        return int(seconds) * 1000 + int(microseconds) // 1000

    async def allow(self, client: str) -> Decision:
        key = f"rl:{client}"
        now = await self._now_ms()
        member = f"{now}:{uuid.uuid4().hex}"
        async with self.redis.pipeline(transaction=True) as pipe:
            _, _, count, oldest, _ = await (
                pipe.zremrangebyscore(key, 0, now - self.window_ms)
                .zadd(key, {member: now})
                .zcard(key)
                .zrange(key, 0, 0, withscores=True)
                .pexpire(key, self.window_ms)
                .execute()
            )
        if count <= self.limit:
            return Decision(True, self.limit - count, 0, self.limit)
        await self.redis.zrem(key, member)  # a rejected request must not count against the client
        oldest_score = int(oldest[0][1]) if oldest else now
        retry_after = max(1, math.ceil((oldest_score + self.window_ms - now) / 1000))
        return Decision(False, 0, retry_after, self.limit)
