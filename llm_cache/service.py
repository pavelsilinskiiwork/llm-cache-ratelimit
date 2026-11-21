"""One request through the whole stack.

    rate limit -> cache lookup (single-flight) -> reserve budget -> model call -> settle -> cache

Order matters. The rate limit applies to every request, hits included, because it protects the
service itself. The budget is only touched on a real model call, so cached answers stay available
after the day's budget is gone. If Redis is down, `fail_open` decides whether requests are refused
(default: cost protection first) or served without limits.
"""

import logging
import time
from dataclasses import dataclass

from redis.exceptions import RedisError

from .budget import DailyBudget
from .cache import REDIS_ERRORS, CacheEntry, LLMCache, cache_key
from .metrics import Metrics
from .pricing import Prices, cost_micro, estimate_cost_micro, usd
from .providers import Provider, ProviderError
from .ratelimit import SlidingWindowLimiter
from .settings import Settings

log = logging.getLogger("llm_cache")


class RateLimited(Exception):
    def __init__(self, retry_after: int, limit: int):
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.limit = limit


class BudgetExhausted(Exception):
    pass


class StorageUnavailable(Exception):
    pass


@dataclass
class ChatRequest:
    message: str
    system: str = ""
    max_tokens: int = 300
    cache: bool = True


@dataclass
class ChatResult:
    text: str
    source: str  # hit | miss | coalesced | bypass
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # what this request cost us (0 when answered from cache)
    saved_usd: float  # what the cache saved (0 when the model was called)
    rate_remaining: int | None
    rate_limit: int


class ChatService:
    def __init__(self, redis, provider: Provider, settings: Settings, clock=None, today=None):
        self.redis = redis
        self.provider = provider
        self.settings = settings
        self.prices = Prices(settings.price_input_per_mtok, settings.price_output_per_mtok)
        self.cache = LLMCache(redis, settings.cache_ttl_seconds)
        self.limiter = SlidingWindowLimiter(
            redis, settings.rate_limit_requests, settings.rate_limit_window_seconds, clock=clock
        )
        budget_args = {"today": today} if today else {}
        self.budget = DailyBudget(redis, round(settings.daily_budget_usd * 1_000_000), **budget_args)
        self.metrics = Metrics(redis)

    async def chat(self, client_id: str, req: ChatRequest) -> ChatResult:
        await self.metrics.incr(requests=1)
        remaining = await self._check_rate(client_id)
        key = cache_key(self.provider.name, self.provider.model, req.system, req.message, req.max_tokens)

        async def compute() -> CacheEntry:
            return await self._call_model(req)

        if req.cache:
            entry, source = await self.cache.get_or_compute(key, compute)
        else:
            entry, source = await compute(), "bypass"

        counter = {"hit": "hits", "coalesced": "coalesced", "miss": "misses", "bypass": "bypass"}[source]
        saved = entry.cost_micro if source in ("hit", "coalesced") else 0
        await self.metrics.incr(**{counter: 1, "saved_micro": saved})
        called_model = source in ("miss", "bypass")
        return ChatResult(
            text=entry.text,
            source=source,
            model=entry.model,
            provider=entry.provider,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cost_usd=usd(entry.cost_micro) if called_model else 0.0,
            saved_usd=0.0 if called_model else usd(entry.cost_micro),
            rate_remaining=remaining,
            rate_limit=self.settings.rate_limit_requests,
        )

    async def _check_rate(self, client_id: str) -> int | None:
        try:
            decision = await self.limiter.allow(client_id)
        except REDIS_ERRORS as e:
            return self._storage_down("rate limiter", e)
        if not decision.allowed:
            await self.metrics.incr(rate_limited=1)
            raise RateLimited(decision.retry_after, decision.limit)
        return decision.remaining

    def _storage_down(self, what: str, error: Exception) -> None:
        log.error("storage_unavailable %s %s", what, type(error).__name__)
        if self.settings.fail_open:
            return
        raise StorageUnavailable(f"{what} is unavailable") from error

    async def _call_model(self, req: ChatRequest) -> CacheEntry:
        reserved = estimate_cost_micro(self.prices, req.system, req.message, req.max_tokens)
        reservation_held = True
        try:
            if not await self.budget.reserve(reserved):
                await self.metrics.incr(budget_rejected=1)
                raise BudgetExhausted("daily budget exhausted")
        except RedisError as e:
            self._storage_down("budget", e)
            reservation_held = False  # fail open: no reservation was made

        try:
            result = await self.provider.generate(req.system, req.message, req.max_tokens)
        except ProviderError:
            await self.metrics.incr(provider_errors=1)
            if reservation_held:
                await self._safely(self.budget.release(reserved))
            raise
        actual = cost_micro(self.prices, result.input_tokens, result.output_tokens)
        if reservation_held:
            await self._safely(self.budget.settle(reserved, actual))
        await self.metrics.incr(spent_micro=actual)
        return CacheEntry(
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_micro=actual,
            model=result.model,
            provider=result.provider,
            created_at=time.time(),
        )

    @staticmethod
    async def _safely(awaitable) -> None:
        try:
            await awaitable
        except REDIS_ERRORS as e:
            log.warning("budget_update_failed %s", type(e).__name__)

    async def stats(self) -> dict:
        s = await self.metrics.snapshot()
        hits, coalesced, misses = s.get("hits", 0), s.get("coalesced", 0), s.get("misses", 0)
        cacheable = hits + coalesced + misses
        try:
            today = await self.budget.spent()
        except REDIS_ERRORS:
            today = 0
        return {
            "requests": s.get("requests", 0),
            "cache": {
                "hits": hits,
                "coalesced": coalesced,
                "misses": misses,
                "bypass": s.get("bypass", 0),
                "hit_rate": round((hits + coalesced) / cacheable, 4) if cacheable else None,
                "saved_usd": usd(s.get("saved_micro", 0)),
            },
            "rate_limited": s.get("rate_limited", 0),
            "budget_rejected": s.get("budget_rejected", 0),
            "provider_errors": s.get("provider_errors", 0),
            "spent_usd_total": usd(s.get("spent_micro", 0)),
            "spent_usd_today": usd(today),
            "daily_budget_usd": self.settings.daily_budget_usd,
        }
