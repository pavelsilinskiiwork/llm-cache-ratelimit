"""Money is counted in micro-dollars (integers): $1 per 1M tokens is exactly 1 micro-dollar per token.

Integers keep Redis counters exact (INCRBY) and avoid float drift when many small costs add up.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    input_per_mtok: float
    output_per_mtok: float


def estimate_tokens(*texts: str) -> int:
    return max(1, math.ceil(sum(len(t) for t in texts) / 4))


def cost_micro(prices: Prices, input_tokens: int, output_tokens: int) -> int:
    return math.ceil(input_tokens * prices.input_per_mtok + output_tokens * prices.output_per_mtok)


def estimate_cost_micro(prices: Prices, system: str, message: str, max_tokens: int) -> int:
    """Worst case for one call: the input as sent plus a full max_tokens answer."""
    return cost_micro(prices, estimate_tokens(system, message), max_tokens)


def usd(micro: int) -> float:
    return round(micro / 1_000_000, 6)
