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
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Check env config if set
    env_provider = os.environ.get("CLASSIFIER_LLM_PROVIDER") or os.environ.get("AGENT_LLM_PROVIDER")
    if env_provider:
        env_p = env_provider.lower().strip()
        if env_p in ("bedrock", "anthropic", "openai"):
            return env_p

    m = (model_id or "").lower()
    if "bedrock" in m:
        return "bedrock"
    if any(k in m for k in ("claude", "anthropic")):
        return "anthropic"
    if any(k in m for k in ("gpt", "o1", "o3", "openai")):
        return "openai"

    # Default fallback to anthropic
    return "anthropic"


def resolve_model_for_provider(model_id: str, provider: str) -> str:
    """Map canonical or cross-provider model IDs to the format expected by the target provider."""
    p = provider.lower()
    m = model_id or ""

    if p == "bedrock":
        env_classifier = os.environ.get("BEDROCK_CLASSIFIER_MODEL_ID")
        if env_classifier:
            return env_classifier
        if any(prefix in m for prefix in ("anthropic.", "us.", "eu.", "global.", "amazon.")):
            return m
        m_lower = m.lower()
        if "sonnet" in m_lower or "claude" in m_lower:
            agent_model = os.environ.get("BEDROCK_AGENT_MODEL_ID", "")
            if any(k in agent_model.lower() for k in ("claude", "sonnet")):
                return agent_model
            return "global.anthropic.claude-sonnet-4-6"
        if "haiku" in m_lower:
            sim_model = os.environ.get("BEDROCK_SIM_MODEL_ID", "")
            if "haiku" in sim_model.lower():
                return sim_model
            return "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        return os.environ.get("BEDROCK_AGENT_MODEL_ID") or m

    if p == "anthropic":
        if any(prefix in m for prefix in ("anthropic.", "us.", "eu.", "global.")):
            for part in m.split("."):
                if part.startswith("claude-"):
                    return part.split(":")[0]
            return os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-5-5")
        return m or os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-5-5")

    if p == "openai":
        if any(k in m for k in ("gpt", "o1", "o3")):
            return m
        return os.environ.get("AGENT_OPENAI_MODEL", "gpt-4o-mini")

    return m


def resolve_provider_chain(requested_provider: str) -> List[str]:
    """Build prioritized list of providers to try in order with fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    p = requested_provider.lower().strip()
    if p == "bedrock":
        chain = ["bedrock"]
        if os.environ.get("ANTHROPIC_API_KEY"):
            chain.append("anthropic")
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("AGENT_OPENAI_API_KEY"):
            chain.append("openai")
        return chain
    return [p]


class LLMClient:
    """Common LLM caller object for multimodal vision and text inference."""

    def __init__(self, tracker: Optional[CostTracker] = None):
        self.tracker = tracker or get_cost_tracker()
        self._clients: Dict[str, Any] = {}

    def _get_bedrock_client(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        region = region or os.environ.get("AWS_REGION", "us-east-1")
        bearer_token = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        timeout_val = float(timeout or os.environ.get("BEDROCK_TIMEOUT", "12.0"))

        client_cache_key = (
            f"bedrock:{region}:{bearer_token[:8] if bearer_token else aws_access_key[:8] if aws_access_key else 'default'}:{timeout_val}"
        )
        if client_cache_key not in self._clients:
            from anthropic import AnthropicBedrock
            kw: Dict[str, Any] = {"aws_region": region, "timeout": timeout_val}
            if bearer_token:
                kw["api_key"] = bearer_token
            elif aws_access_key and aws_secret_key:
                kw["aws_access_key"] = aws_access_key
                kw["aws_secret_key"] = aws_secret_key
                if aws_session_token:
                    kw["aws_session_token"] = aws_session_token
            self._clients[client_cache_key] = AnthropicBedrock(**kw)
        return self._clients[client_cache_key]

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
        if client_cache_key not in self._clients and f"openai:{key[:8]}:" in self._clients:
            return self._clients[f"openai:{key[:8]}:"]
        if client_cache_key not in self._clients:
            from openai import OpenAI
            self._clients[client_cache_key] = OpenAI(api_key=key, base_url=url)
        return self._clients[client_cache_key]

    def _execute_vision_call(
        self,
        provider: str,
        target_model: str,
        prompt: str,
        image_b64: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Tuple[str, int, int, int, int, Any]:
        if provider == "bedrock":
            client = self._get_bedrock_client(api_key)
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
                "model": target_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_content}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        elif provider == "anthropic":
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
                "model": target_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_content}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        elif provider == "openai":
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
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = raw_resp.choices[0] if raw_resp.choices else None
            text = (choice.message.content or "") if choice and choice.message else ""
            usage = getattr(raw_resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            think_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            tot_tok = int(getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        else:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Supported: 'bedrock', 'anthropic', 'openai'."
            )

    def _execute_text_call(
        self,
        provider: str,
        target_model: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Tuple[str, int, int, int, int, Any]:
        if provider == "bedrock":
            client = self._get_bedrock_client(api_key)
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        elif provider == "anthropic":
            client = self._get_anthropic_client(api_key)
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            raw_resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in raw_resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(raw_resp.usage, "input_tokens", 0)
            out_tok = getattr(raw_resp.usage, "output_tokens", 0)
            think_tok = int(
                getattr(raw_resp.usage, "thinking_tokens", 0)
                or getattr(raw_resp.usage, "reasoning_tokens", 0)
                or 0
            )
            tot_tok = in_tok + out_tok
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        elif provider == "openai":
            client = self._get_openai_client(api_key, base_url)
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            raw_resp = client.chat.completions.create(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = raw_resp.choices[0] if raw_resp.choices else None
            text = (choice.message.content or "") if choice and choice.message else ""
            usage = getattr(raw_resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            think_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            tot_tok = int(getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))
            return text, in_tok, out_tok, think_tok, tot_tok, raw_resp

        else:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Supported: 'bedrock', 'anthropic', 'openai'."
            )

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
        """Execute a multimodal vision LLM call across providers with automatic fallback and cost tracking."""
        initial_provider = infer_provider(model_id, provider)
        provider_chain = resolve_provider_chain(initial_provider)
        last_error: Optional[Exception] = None

        for idx, current_provider in enumerate(provider_chain):
            target_model = resolve_model_for_provider(model_id, current_provider)
            t0 = time.perf_counter()
            logger.info(
                f"[LLMClient] Executing vision call: provider='{current_provider}', model='{target_model}', op='{op}'"
            )
            try:
                text, in_tok, out_tok, think_tok, tot_tok, raw_resp = self._execute_vision_call(
                    provider=current_provider,
                    target_model=target_model,
                    prompt=prompt,
                    image_b64=image_b64,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                recorded = self.tracker.record_call(
                    model_id=target_model,
                    tokens_in=in_tok,
                    tokens_out=out_tok,
                    tokens_thinking=think_tok,
                    latency_ms=latency_ms,
                    tag=f"CLASSIFIER:{current_provider}",
                    op=op,
                )

                return LLMResponse(
                    text=text,
                    model_id=target_model,
                    provider=current_provider,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thinking_tokens=think_tok,
                    total_tokens=tot_tok,
                    cost_usd=recorded.get("cost_usd"),
                    latency_ms=latency_ms,
                    raw_response=raw_resp,
                )
            except Exception as e:
                last_error = e
                remaining = provider_chain[idx + 1:]
                if remaining:
                    logger.warning(
                        f"[LLMClient] Provider '{current_provider}' failed ({type(e).__name__}: {e}) -> falling back to '{remaining[0]}'"
                    )
                else:
                    logger.error(
                        f"[LLMClient] Provider '{current_provider}' failed ({type(e).__name__}: {e}) and no further fallbacks remaining."
                    )
        raise RuntimeError(
            f"All LLM providers in chain {provider_chain} failed. Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

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
        """Execute a text-only LLM call across providers with automatic fallback and cost tracking."""
        initial_provider = infer_provider(model_id, provider)
        provider_chain = resolve_provider_chain(initial_provider)
        last_error: Optional[Exception] = None

        for idx, current_provider in enumerate(provider_chain):
            target_model = resolve_model_for_provider(model_id, current_provider)
            t0 = time.perf_counter()
            logger.info(
                f"[LLMClient] Executing text call: provider='{current_provider}', model='{target_model}', op='{op}'"
            )
            try:
                text, in_tok, out_tok, think_tok, tot_tok, raw_resp = self._execute_text_call(
                    provider=current_provider,
                    target_model=target_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                recorded = self.tracker.record_call(
                    model_id=target_model,
                    tokens_in=in_tok,
                    tokens_out=out_tok,
                    tokens_thinking=think_tok,
                    latency_ms=latency_ms,
                    tag=f"CLASSIFIER:{current_provider}",
                    op=op,
                )

                return LLMResponse(
                    text=text,
                    model_id=target_model,
                    provider=current_provider,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thinking_tokens=think_tok,
                    total_tokens=tot_tok,
                    cost_usd=recorded.get("cost_usd"),
                    latency_ms=latency_ms,
                    raw_response=raw_resp,
                )
            except Exception as e:
                last_error = e
                remaining = provider_chain[idx + 1:]
                if remaining:
                    logger.warning(
                        f"[LLMClient] Provider '{current_provider}' failed ({type(e).__name__}: {e}) -> falling back to '{remaining[0]}'"
                    )
                else:
                    logger.error(
                        f"[LLMClient] Provider '{current_provider}' failed ({type(e).__name__}: {e}) and no further fallbacks remaining."
                    )
        raise RuntimeError(
            f"All LLM providers in chain {provider_chain} failed. Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error


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
