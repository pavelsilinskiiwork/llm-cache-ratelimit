from llm_cache.settings import Settings


def test_defaults():
    s = Settings.from_env({})
    assert s.provider == "fake" and s.rate_limit_requests == 10 and s.fail_open is False


def test_reads_environment():
    s = Settings.from_env({"RATE_LIMIT_REQUESTS": "3", "FAIL_OPEN": "true", "DAILY_BUDGET_USD": "0.25",
                           "PROVIDER": " Groq ", "GROQ_API_KEY": " k "})  # fmt: skip
    assert (s.rate_limit_requests, s.fail_open, s.daily_budget_usd) == (3, True, 0.25)
    assert (s.provider, s.groq_api_key) == ("groq", "k")
