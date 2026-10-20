"""Token + cost bookkeeping for LLM calls, so you can see what the app spends while you test.

Prices are USD per 1M tokens (input, output) as listed by third-party trackers -- double-check them on the Bedrock pricing
page. Unknown models are logged with tokens only. This is an estimate, not a bill.
"""
from fridge_log import log

PRICES = {  # substring of the model id -> (input $/1M, output $/1M)
    "haiku-4-5": (1.00, 5.00), "sonnet-4-6": (3.00, 15.00), "sonnet-4-5": (3.00, 15.00),
    "haiku-5-5": (0.11, 0.55), "sonnet-5-5": (2.20, 11.00),
}
_total = {"usd": 0.0, "in": 0, "out": 0, "calls": 0}


def price_for(model_id: str):
    m = (model_id or "").lower()
    return next((p for k, p in PRICES.items() if k in m), None)


def estimate(model_id: str, tokens_in: int, tokens_out: int):
    p = price_for(model_id)
    return None if p is None else (tokens_in * p[0] + tokens_out * p[1]) / 1e6


def record(tag: str, model_id: str, tokens_in: int, tokens_out: int, what: str = "") -> None:
    """Log one call's tokens and estimated cost, plus the running total for this process."""
    cost = estimate(model_id, tokens_in, tokens_out)
    _total["in"] += tokens_in; _total["out"] += tokens_out; _total["calls"] += 1
    if cost is not None:
        _total["usd"] += cost
    price = f"~${cost:.4f}" if cost is not None else "cost n/a"
    log(tag, f"{what or 'LLM call'}: {tokens_in} in / {tokens_out} out tokens, {price}  (total so far ~${_total['usd']:.3f}, {_total['calls']} calls)")


def usage_from_result(result) -> tuple[int, int]:
    """Pull (input, output) token counts out of a Strands AgentResult."""
    try:
        u = result.metrics.accumulated_usage
        return int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
    except Exception:
        return 0, 0


def totals() -> dict:
    return dict(_total)
