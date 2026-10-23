"""Unified LLM client for the classifier subsystem.

Abstracts differences between LLM providers (Anthropic Claude, OpenAI, etc.),
normalizes multimodal (vision) payloads, extracts token breakdown (input, output,
thinking/reasoning), and automatically tracks usage and cost via CostTracker.
"""

from dataclasses import dataclass, field
import functools
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from classifier.llm.tracker import CostTracker, get_cost_tracker

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response container for all LLM calls in classifier."""

    text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int
    cost_usd: Optional[float]
    latency_ms: float
    raw_response: Any = field(repr=False, default=None)


def infer_provider(model_id: str, explicit_provider: Optional[str] = None) -> str:
    """Determine provider based on explicit configuration or model name."""
    if explicit_provider:
        return explicit_provider.lower().strip()

    m = (model_id or "").lower()
    if any(k in m for k in ("claude", "anthropic")):
        return "anthropic"
    if any(k in m for k in ("gpt", "o1", "o3", "openai")):
        return "openai"

    # Default fallback to anthropic
    return "anthropic"


class LLMClient:
    """Common LLM caller object for multimodal vision and text inference."""

    def __init__(self, tracker: Optional[CostTracker] = None):
        self.tracker = tracker or get_cost_tracker()
        self._clients: Dict[str, Any] = {}

    def _get_anthropic_client(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            # Check dotenv
            from dotenv import load_dotenv
            load_dotenv()
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set it in your environment or .env file."
            )

        client_cache_key = f"anthropic:{key[:8]}"
        if client_cache_key not in self._clients:
            from anthropic import Anthropic
            self._clients[client_cache_key] = Anthropic(api_key=key)
        return self._clients[client_cache_key]

    def _get_openai_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not key:
            from dotenv import load_dotenv
            load_dotenv()
            key = api_key or os.environ.get("OPENAI_API_KEY")
            url = base_url or os.environ.get("OPENAI_BASE_URL")

        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your environment or .env file."
            )

        client_cache_key = f"openai:{key[:8]}:{url or ''}"
        if client_cache_key not in self._clients:
            from openai import OpenAI
            self._clients[client_cache_key] = OpenAI(api_key=key, base_url=url)
        return self._clients[client_cache_key]

    def complete_vision(
        self,
        prompt: str,
        image_b64: str,
        system_prompt: str = "",
        model_id: str = "claude-sonnet-5-5",
        provider: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
        op: str = "detect",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a multimodal vision LLM call across providers and automatically track cost/tokens."""
        resolved_provider = infer_provider(model_id, provider)
        t0 = time.perf_counter()

        logger.info(
            f"[LLMClient] Executing vision call: provider='{resolved_provider}', model='{model_id}', op='{op}'"
        )

        if resolved_provider == "anthropic":
            client = self._get_anthropic_client(api_key)
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_content}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok

        elif resolved_provider == "openai":
            client = self._get_openai_client(api_key, base_url)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ]
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})

            raw_resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice = raw_resp.choices[0] if raw_resp.choices else None
            text = (choice.message.content or "") if choice and choice.message else ""

            usage = getattr(raw_resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            think_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            tot_tok = int(getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))

        else:
            raise ValueError(
                f"Unsupported LLM provider '{resolved_provider}'. Supported: 'anthropic', 'openai'."
            )

        # Log metrics to .txt file and calculate cost
        recorded = self.tracker.record_call(
            model_id=model_id,
            tokens_in=in_tok,
            tokens_out=out_tok,
            tokens_thinking=think_tok,
            latency_ms=latency_ms,
            tag=f"CLASSIFIER:{resolved_provider}",
            op=op,
        )

        return LLMResponse(
            text=text,
            model_id=model_id,
            provider=resolved_provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
            total_tokens=tot_tok,
            cost_usd=recorded.get("cost_usd"),
            latency_ms=latency_ms,
            raw_response=raw_resp,
        )

    def complete_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_id: str = "claude-sonnet-5-5",
        provider: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
        op: str = "text",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a text-only LLM call across providers and automatically track cost/tokens."""
        resolved_provider = infer_provider(model_id, provider)
        t0 = time.perf_counter()

        if resolved_provider == "anthropic":
            client = self._get_anthropic_client(api_key)
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok

        elif resolved_provider == "openai":
            client = self._get_openai_client(api_key, base_url)
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            raw_resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice = raw_resp.choices[0] if raw_resp.choices else None
            text = (choice.message.content or "") if choice and choice.message else ""

            usage = getattr(raw_resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            think_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            tot_tok = int(getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))

        else:
            raise ValueError(
                f"Unsupported LLM provider '{resolved_provider}'. Supported: 'anthropic', 'openai'."
            )

        recorded = self.tracker.record_call(
            model_id=model_id,
            tokens_in=in_tok,
            tokens_out=out_tok,
            tokens_thinking=think_tok,
            latency_ms=latency_ms,
            tag=f"CLASSIFIER:{resolved_provider}",
            op=op,
        )

        return LLMResponse(
            text=text,
            model_id=model_id,
            provider=resolved_provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=think_tok,
            total_tokens=tot_tok,
            cost_usd=recorded.get("cost_usd"),
            latency_ms=latency_ms,
            raw_response=raw_resp,
        )


def track_llm_call(op: str = "custom_call", tag: str = "CLASSIFIER:custom"):
    """Decorator for tracking cost and tokens on any arbitrary LLM invocation."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            resp = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            model_id = kwargs.get("model") or kwargs.get("model_id") or "unknown"
            usage = getattr(resp, "usage", None)
            if usage:
                in_tok = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
                out_tok = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0)
                details = getattr(usage, "completion_tokens_details", None)
                think_tok = getattr(details, "reasoning_tokens", 0) if details else (
                    getattr(usage, "thinking_tokens", 0) or 0
                )
                get_cost_tracker().record_call(
                    model_id=str(model_id),
                    tokens_in=int(in_tok or 0),
                    tokens_out=int(out_tok or 0),
                    tokens_thinking=int(think_tok or 0),
                    latency_ms=latency_ms,
                    tag=tag,
                    op=op,
                )
            return resp
        return wrapper
    return decorator


_global_client = LLMClient()


def get_llm_client() -> LLMClient:
    """Return the global LLMClient instance."""
    return _global_client
