"""Unified cost and token tracker for the classifier subsystem.

Logs all LLM call metrics (input tokens, output tokens, thinking/reasoning tokens,
estimated cost, latency, caller tag, operation name) directly into a .txt file.
"""

from datetime import datetime
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple


# Prices in USD per 1M tokens: (input $/1M, output $/1M)
# Thinking/reasoning tokens are billed as output tokens.
PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude
    "haiku-3-5": (0.80, 4.00),
    "haiku-4-5": (1.00, 5.00),
    "haiku-5-5": (0.11, 0.55),
    "sonnet-3-5": (3.00, 15.00),
    "sonnet-3-7": (3.00, 15.00),
    "sonnet-4-5": (3.00, 15.00),
    "sonnet-4-6": (3.00, 15.00),
    "sonnet-5-5": (2.20, 11.00),
    "opus-3": (15.00, 75.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
}

DEFAULT_LOG_FILE = "classifier_llm_costs.txt"


def price_for(model_id: str) -> Optional[Tuple[float, float]]:
    """Lookup (input, output) price per 1M tokens based on model name substring."""
    m = (model_id or "").lower()
    return next((price for key, price in PRICES.items() if key in m), None)


def estimate_cost(
    model_id: str, tokens_in: int, tokens_out: int, tokens_thinking: int = 0
) -> Optional[float]:
    """Estimate call cost in USD. Returns None if model is unpriced.
    
    If tokens_thinking is supplied and not already part of tokens_out,
    it is treated at the model's output token price.
    """
    p = price_for(model_id)
    if p is None:
        return None
    # If output tokens is less than thinking tokens, thinking tokens were provided separately
    effective_out = tokens_out if tokens_out >= tokens_thinking else (tokens_out + tokens_thinking)
    return (tokens_in * p[0] + effective_out * p[1]) / 1e6


class CostTracker:
    """Thread-safe cost and token tracking logger for classifier LLM operations."""

    def __init__(self, log_file: Optional[str] = None):
        self._custom_log_file = log_file
        self._lock = threading.Lock()
        self._total = {"usd": 0.0, "in": 0, "out": 0, "thinking": 0, "calls": 0}

    @property
    def log_file_path(self) -> Path:
        """Resolve destination .txt file path."""
        file_name = (
            self._custom_log_file
            or os.environ.get("CLASSIFIER_LLM_COST_FILE")
            or DEFAULT_LOG_FILE
        )
        return Path(file_name)

    def set_log_file(self, file_path: Optional[str]) -> None:
        """Dynamically set the output .txt file path."""
        with self._lock:
            self._custom_log_file = file_path

    def record_call(
        self,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        tokens_thinking: int = 0,
        latency_ms: Optional[float] = None,
        tag: str = "CLASSIFIER:vision",
        op: str = "detect",
    ) -> Dict[str, Any]:
        """Record an LLM call's metrics, update running totals, and append to the .txt file."""
        tokens_in = max(0, int(tokens_in or 0))
        tokens_out = max(0, int(tokens_out or 0))
        tokens_thinking = max(0, int(tokens_thinking or 0))

        # Total tokens: thinking tokens are generally included in completion/output tokens
        total_tokens = tokens_in + (tokens_out if tokens_out >= tokens_thinking else (tokens_out + tokens_thinking))
        cost = estimate_cost(model_id, tokens_in, tokens_out, tokens_thinking)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        latency_val = float(latency_ms) if latency_ms is not None else 0.0

        with self._lock:
            self._total["in"] += tokens_in
            self._total["out"] += tokens_out
            self._total["thinking"] += tokens_thinking
            self._total["calls"] += 1
            if cost is not None:
                self._total["usd"] += cost

            running_usd = self._total["usd"]
            running_calls = self._total["calls"]

            # Format log entry
            cost_str = f"${cost:.4f}" if cost is not None else "cost_n/a"
            total_usd_str = f"${running_usd:.4f}"
            line = (
                f"[{now_str}] [{tag}] model={model_id} | op={op} | "
                f"tokens: in={tokens_in}, out={tokens_out}, thinking={tokens_thinking}, total={total_tokens} | "
                f"cost={cost_str} (total={total_usd_str}, calls={running_calls}) | "
                f"latency={latency_val:.1f}ms\n"
            )

            # Ensure parent directories exist and write atomically
            target_path = self.log_file_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

        return {
            "timestamp": now_str,
            "tag": tag,
            "op": op,
            "model_id": model_id,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "thinking_tokens": tokens_thinking,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "running_total_usd": running_usd,
            "running_calls": running_calls,
            "latency_ms": latency_val,
        }

    def totals(self) -> Dict[str, Any]:
        """Return a snapshot copy of running totals for this process."""
        with self._lock:
            return dict(self._total)

    def reset_totals(self) -> None:
        """Reset the running totals in memory."""
        with self._lock:
            self._total = {"usd": 0.0, "in": 0, "out": 0, "thinking": 0, "calls": 0}


# Global singleton tracker instance
_global_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Return the global CostTracker instance."""
    return _global_tracker
