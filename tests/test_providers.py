import httpx
import pytest

from llm_cache.providers import FakeProvider, GroqProvider, ProviderError, build_provider
from llm_cache.settings import Settings


def groq(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqProvider("secret-key", "m", client=client)


async def test_groq_parses_a_normal_response():
    def handler(request):
        assert request.headers["authorization"] == "Bearer secret-key"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })  # fmt: skip

    result = await groq(handler).generate("sys", "hello", 50)
    assert (result.text, result.input_tokens, result.output_tokens) == ("hi", 3, 2)


async def test_groq_429_is_marked_rate_limited():
    with pytest.raises(ProviderError) as e:
        await groq(lambda r: httpx.Response(429)).generate("", "hello", 50)
    assert e.value.rate_limited


async def test_groq_errors_do_not_leak_the_key():
    with pytest.raises(ProviderError) as e:
        await groq(lambda r: httpx.Response(500, text="secret-key leaked")).generate("", "hello", 50)
    assert "secret-key" not in str(e.value)


async def test_groq_bad_payload_and_network_error():
    with pytest.raises(ProviderError):
        await groq(lambda r: httpx.Response(200, json={"nope": 1})).generate("", "hello", 50)

    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError):
        await groq(boom).generate("", "hello", 50)


def test_build_provider():
    assert isinstance(build_provider(Settings()), FakeProvider)
    with pytest.raises(ValueError):
        build_provider(Settings(provider="groq"))  # no key
    with pytest.raises(ValueError):
        build_provider(Settings(provider="nope"))


async def test_fake_provider_fails_then_recovers():
    p = FakeProvider(fail_times=1)
    with pytest.raises(ProviderError):
        await p.generate("", "x", 10)
    assert (await p.generate("", "x", 10)).text == "Echo: x"
