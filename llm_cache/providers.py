"""LLM providers behind one tiny async interface.

`FakeProvider` is the default: deterministic, free, with configurable latency, so the cache, limiter
and budget can be demonstrated and benchmarked without spending anything. `GroqProvider` talks to
Groq's OpenAI-compatible API.
"""

import asyncio
from dataclasses import dataclass

import httpx

from .settings import Settings


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class ProviderError(Exception):
    def __init__(self, message: str, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


class Provider:
    name = "provider"
    model = "model"

    async def generate(self, system: str, message: str, max_tokens: int) -> LLMResult:
        raise NotImplementedError


def _tokens(*texts: str) -> int:
    return max(1, -(-sum(len(t) for t in texts) // 4))  # about 4 characters per token, rounded up


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, latency: float = 0.0, model: str = "fake-1", fail_times: int = 0):
        self.latency = latency
        self.model = model
        self.fail_times = fail_times
        self.calls = 0

    async def generate(self, system: str, message: str, max_tokens: int) -> LLMResult:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError("fake provider failure")
        text = f"Echo: {message}"
        return LLMResult(
            text, _tokens(system, message), min(max_tokens, _tokens(text)), self.model, self.name
        )


class GroqProvider(Provider):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for PROVIDER=groq")
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=60)

    async def generate(self, system: str, message: str, max_tokens: int) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": message}
        ]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            response = await self.client.post(
                self.URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"request failed: {type(e).__name__}") from e
        if response.status_code == 429:
            raise ProviderError("provider rate limit", rate_limited=True)
        if response.status_code >= 400:
            raise ProviderError(f"provider returned HTTP {response.status_code}")
        try:
            data = response.json()
            usage = data.get("usage") or {}
            return LLMResult(
                text=data["choices"][0]["message"]["content"] or "",
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                model=self.model,
                provider=self.name,
            )
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderError("unexpected response format") from e


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "fake":
        return FakeProvider()
    if settings.provider == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model)
    raise ValueError(f"Unknown PROVIDER {settings.provider!r} (use fake or groq)")
