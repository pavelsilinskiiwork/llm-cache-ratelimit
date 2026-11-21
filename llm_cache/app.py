"""FastAPI front end: POST /chat, GET /stats, GET /health."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .providers import Provider, ProviderError, build_provider
from .service import (
    BudgetExhausted,
    ChatRequest,
    ChatService,
    RateLimited,
    StorageUnavailable,
)
from .settings import Settings

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    system: str = Field(default="", max_length=2000)
    max_tokens: int = Field(default=300, ge=1, le=1000)
    cache: bool = True


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class RateInfo(BaseModel):
    limit: int
    remaining: int | None


class ChatResponse(BaseModel):
    text: str
    source: str
    cached: bool
    model: str
    provider: str
    usage: Usage
    cost_usd: float
    saved_usd: float
    rate_limit: RateInfo


def create_app(
    settings: Settings | None = None, redis_client=None, provider: Provider | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    holder: dict[str, ChatService] = {}

    def build(client) -> ChatService:
        return ChatService(client, provider or build_provider(settings), settings)

    @asynccontextmanager
    async def lifespan(app):
        client = redis_client or Redis.from_url(settings.redis_url, decode_responses=True)
        holder["service"] = build(client)
        yield
        if redis_client is None:
            await client.aclose()

    app = FastAPI(
        title="LLM cache, rate limit and budget",
        description="Redis response cache (single-flight), per-client rate limit, daily spending cap.",
        version="1.0.0",
        lifespan=lifespan,
    )
    if redis_client is not None:  # tests and scripts that hand in a client skip the lifespan
        holder["service"] = build(redis_client)

    def client_id(request: Request) -> str:
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.exception_handler(RateLimited)
    async def on_rate_limited(_, e: RateLimited):
        return JSONResponse(
            {"error": "rate_limited", "retry_after_seconds": e.retry_after, "limit": e.limit},
            status_code=429,
            headers={"Retry-After": str(e.retry_after)},
        )

    @app.exception_handler(BudgetExhausted)
    async def on_budget(_, e: BudgetExhausted):
        return JSONResponse({"error": "daily_budget_exhausted"}, status_code=429)

    @app.exception_handler(StorageUnavailable)
    async def on_storage(_, e: StorageUnavailable):
        return JSONResponse({"error": "storage_unavailable"}, status_code=503)

    @app.exception_handler(ProviderError)
    async def on_provider(_, e: ProviderError):
        status = 429 if e.rate_limited else 502
        return JSONResponse(
            {"error": "provider_rate_limited" if e.rate_limited else "provider_failed"}, status_code=status
        )

    @app.get("/health")
    async def health():
        service = holder["service"]
        try:
            await service.redis.ping()
        except (RedisError, OSError):
            return JSONResponse({"status": "degraded", "redis": False}, status_code=503)
        return {"status": "ok", "redis": True}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatBody, request: Request):
        result = await holder["service"].chat(
            client_id(request), ChatRequest(body.message, body.system, body.max_tokens, body.cache)
        )
        return ChatResponse(
            text=result.text,
            source=result.source,
            cached=result.source in ("hit", "coalesced"),
            model=result.model,
            provider=result.provider,
            usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
            cost_usd=result.cost_usd,
            saved_usd=result.saved_usd,
            rate_limit=RateInfo(limit=result.rate_limit, remaining=result.rate_remaining),
        )

    @app.get("/stats")
    async def stats():
        return await holder["service"].stats()

    return app


app = create_app()
