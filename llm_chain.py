"""Which LLM provider to try, in what order, and when to stop trying a broken one.

Provider setting (LLM_PROVIDER for the simulator, AGENT_LLM_PROVIDER for the dish agent):
  auto       try every provider that has credentials: bedrock -> anthropic -> openai, then the built-in fallback
  bedrock    bedrock first, then the other providers that have keys
  anthropic  anthropic first, then openai if it has a key
  openai     openai only
  demo/rules no LLM at all (built-in keyword router / rules)
The built-in fallback (simulator demo router, dish-agent rules) always runs last, so the app never goes down.

A provider that fails with an access/auth/quota error is skipped for COOLDOWN seconds, so one broken
service costs a single slow call, not a delay on every turn.
"""
import os, time

COOLDOWN = int(os.environ.get("LLM_COOLDOWN_SECONDS", "300"))
_down: dict[str, float] = {}


def has_creds(provider: str, *, agent: bool) -> bool:
    e = os.environ.get
    if provider == "bedrock":
        return bool(e("AWS_BEARER_TOKEN_BEDROCK") or e("AWS_ACCESS_KEY_ID") or e("AWS_PROFILE") or e("AWS_WEB_IDENTITY_TOKEN_FILE"))
    if provider == "anthropic":
        return bool(e("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return bool(e("AGENT_OPENAI_API_KEY") if agent else e("OPENAI_API_KEY"))
    return False


def chain(primary: str, *, agent: bool) -> list[str]:
    """Providers to try, best first, skipping ones without credentials or in cooldown."""
    primary = (primary or "").lower()
    if primary in ("demo", "rules", "", "none", "off"):
        return []
    order = ["bedrock", "anthropic", "openai"]
    if primary in order:                       # the chosen one first, then the rest as backups
        order = [primary] + [p for p in order if p != primary]
    out = []
    for p in order:
        # the chosen provider is always attempted (so a bad key shows a clear error); backups need credentials
        if (p == primary or has_creds(p, agent=agent)) and not is_down(p):
            out.append(p)
    return out


def is_down(provider: str) -> bool:
    until = _down.get(provider, 0)
    if until and time.time() < until:
        return True
    _down.pop(provider, None)
    return False


def mark_down(provider: str, err: Exception) -> None:
    """Skip this provider for a while if the error looks permanent (access, auth, quota, billing)."""
    s = f"{type(err).__name__} {err}".lower()
    permanent = any(k in s for k in ("accessdenied", "not authorized", "throttl", "too many tokens", "invalid_payment",
                                     "use case details", "unrecognizedclient", "invalidsignature", "expiredtoken",
                                     "authentication", "invalid x-api-key", "401", "403", "credit balance", "no credentials",
                                     "nocredentials", "resourcenotfound", "validationexception"))
    if permanent and COOLDOWN > 0:
        _down[provider] = time.time() + COOLDOWN


def reset() -> None:
    _down.clear()
