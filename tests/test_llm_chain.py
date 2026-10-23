"""Provider ordering, credentials detection, and the cooldown for broken providers."""
import llm_chain as lc


def _env(monkeypatch, **kw):
    for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE",
              "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AGENT_OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)
    lc.reset()


def test_auto_uses_only_providers_with_credentials(monkeypatch):
    _env(monkeypatch, ANTHROPIC_API_KEY="x")
    assert lc.chain("auto", agent=False) == ["anthropic"]
    _env(monkeypatch, AWS_ACCESS_KEY_ID="a", ANTHROPIC_API_KEY="x", OPENAI_API_KEY="y")
    assert lc.chain("auto", agent=False) == ["bedrock", "anthropic", "openai"]
    assert lc.chain("auto", agent=True) == ["bedrock", "anthropic"]          # dish agent has its own OpenAI key


def test_no_credentials_means_builtin_fallback(monkeypatch):
    _env(monkeypatch)
    assert lc.chain("auto", agent=False) == []
    assert lc.chain("demo", agent=False) == [] and lc.chain("rules", agent=True) == []


def test_chosen_provider_first_backups_need_keys(monkeypatch):
    _env(monkeypatch, ANTHROPIC_API_KEY="x")
    assert lc.chain("bedrock", agent=False) == ["bedrock", "anthropic"]       # bedrock always attempted when chosen
    assert lc.chain("anthropic", agent=False) == ["anthropic"]


def test_cooldown_skips_a_broken_provider_then_recovers(monkeypatch):
    _env(monkeypatch, AWS_ACCESS_KEY_ID="a", ANTHROPIC_API_KEY="x")
    lc.mark_down("bedrock", RuntimeError("AccessDeniedException: INVALID_PAYMENT_INSTRUMENT"))
    assert lc.chain("bedrock", agent=False) == ["anthropic"]
    monkeypatch.setitem(lc._down, "bedrock", 1.0)                              # cooldown expired
    assert lc.chain("bedrock", agent=False) == ["bedrock", "anthropic"]


def test_transient_errors_are_not_blacklisted(monkeypatch):
    _env(monkeypatch, AWS_ACCESS_KEY_ID="a")
    lc.mark_down("bedrock", TimeoutError("timed out"))
    assert lc.chain("bedrock", agent=False) == ["bedrock"]
