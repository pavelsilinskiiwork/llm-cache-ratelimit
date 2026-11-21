"""A daily spending cap shared by all clients, kept in Redis as integer micro-dollars.

Reserve-then-settle: before calling the model we reserve its worst-case cost (input plus a full
max_tokens answer). If the day's total would go over the cap the reservation is rolled back and the
request is refused. After the call we settle to the real cost. Without the reservation, many
requests in flight at once could each pass a "spent < cap" check and overshoot the cap together.
"""

from collections.abc import Callable
from datetime import UTC, datetime

DAY_SECONDS = 86_400


def utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class DailyBudget:
    def __init__(self, redis, cap_micro: int, today: Callable[[], str] = utc_day):
        self.redis = redis
        self.cap = cap_micro
        self.today = today

    def _key(self) -> str:
        return f"budget:{self.today()}"

    async def reserve(self, amount: int) -> bool:
        key = self._key()
        total = await self.redis.incrby(key, amount)
        await self.redis.expire(key, 2 * DAY_SECONDS)
        if total > self.cap:
            await self.redis.decrby(key, amount)
            return False
        return True

    async def settle(self, reserved: int, actual: int) -> None:
        if actual != reserved:
            await self.redis.incrby(self._key(), actual - reserved)

    async def release(self, reserved: int) -> None:
        await self.redis.decrby(self._key(), reserved)

    async def spent(self) -> int:
        return int(await self.redis.get(self._key()) or 0)
