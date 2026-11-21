"""All configuration comes from environment variables, with safe defaults for a demo."""

import os
from dataclasses import dataclass


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    provider: str = "fake"  # fake | groq
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    cache_ttl_seconds: int = 3600
    rate_limit_requests: int = 10  # per client, per window
    rate_limit_window_seconds: int = 60
    daily_budget_usd: float = 1.0  # shared by all clients, UTC day

    # Illustrative prices in USD per 1M tokens. Set the real ones for your model.
    price_input_per_mtok: float = 1.0
    price_output_per_mtok: float = 5.0

    # When Redis is down: False = refuse requests (cost protection first), True = serve without limits.
    fail_open: bool = False
    trust_proxy: bool = False  # read the client IP from X-Forwarded-For (only behind a proxy you control)

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            redis_url=env.get("REDIS_URL", d.redis_url),
            provider=env.get("PROVIDER", d.provider).strip().lower(),
            groq_api_key=env.get("GROQ_API_KEY", d.groq_api_key).strip(),
            groq_model=env.get("GROQ_MODEL", d.groq_model),
            cache_ttl_seconds=int(env.get("CACHE_TTL_SECONDS", d.cache_ttl_seconds)),
            rate_limit_requests=int(env.get("RATE_LIMIT_REQUESTS", d.rate_limit_requests)),
            rate_limit_window_seconds=int(env.get("RATE_LIMIT_WINDOW_SECONDS", d.rate_limit_window_seconds)),
            daily_budget_usd=float(env.get("DAILY_BUDGET_USD", d.daily_budget_usd)),
            price_input_per_mtok=float(env.get("PRICE_INPUT_PER_MTOK", d.price_input_per_mtok)),
            price_output_per_mtok=float(env.get("PRICE_OUTPUT_PER_MTOK", d.price_output_per_mtok)),
            fail_open=_bool(env.get("FAIL_OPEN", str(d.fail_open))),
            trust_proxy=_bool(env.get("TRUST_PROXY", str(d.trust_proxy))),
        )
