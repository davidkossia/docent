"""Claude prices for the answer eval's spend controls. USD per 1M tokens."""
from __future__ import annotations

from .config import CLAUDE_MODEL

# (input, output, cache read, cache write 5 min). Source: claude-api skill model table.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.50, 6.25),
}
UNKNOWN_MODEL_PRICES = PRICES["claude-opus-5"]  # the dearest known model, so an unknown one is not under-counted
ANSWER_MAX_TOKENS = 16000  # answer.py
WORST_CASE_INPUT_TOKENS = 6000


def _attempt(prices) -> float:
    return (ANSWER_MAX_TOKENS * prices[1] + WORST_CASE_INPUT_TOKENS * prices[0]) / 1e6


# Worst case for one answer call: a full attempt on the main model plus a full attempt on the dearest fallback.
WORST_CASE_CALL_USD = _attempt(PRICES[CLAUDE_MODEL]) + _attempt(UNKNOWN_MODEL_PRICES)
