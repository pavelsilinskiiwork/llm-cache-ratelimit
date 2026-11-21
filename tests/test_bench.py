import os

import pytest

import bench


@pytest.mark.skipif(not os.getenv("REDIS_TEST_URL"), reason="bench needs a real Redis")
def test_bench_runs_and_the_cache_reduces_model_calls(capsys):
    rows = bench.cli(["--redis-url", os.environ["REDIS_TEST_URL"], "--requests", "60", "--distinct", "10",
                      "--latency", "0.01", "--json"])  # fmt: skip
    by = {r["scenario"]: r for r in rows}
    assert by["no cache"]["model_calls"] == 60
    assert by["with cache"]["model_calls"] <= 10
    assert by["stampede: same prompt, single-flight"]["model_calls"] == 1
