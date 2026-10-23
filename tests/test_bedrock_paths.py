"""Strands/Bedrock code paths, run against a scripted fake model (no AWS key, no cost)."""
import asyncio, json
from types import SimpleNamespace

import pytest

pytest.importorskip("strands")
import bedrock_util, llm_usage
from dish_agent import brain
from simulator import agent as sim
from tests.fakes import ScriptedModel

ITEMS = [{"item": "eggs", "qty": 2, "est_days_left": 2}, {"item": "onion", "qty": 1, "est_days_left": 20},
         {"item": "tomato", "qty": 2, "est_days_left": 1}]
DISHES = json.dumps({"dishes": [{"name": "Masala Omelette", "why": "Uses tomato and eggs before they spoil.", "time_min": 10,
                                 "uses_expiring": ["eggs", "tomato"], "have": ["eggs", "onion", "tomato"], "missing": [],
                                 "steps": ["Chop", "Whisk", "Cook"]}]})


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(bedrock_util, "make_model", lambda *a, **k: model)


def test_dish_agent_bedrock_uses_tools_and_returns_dishes(monkeypatch):
    model = ScriptedModel([("tool", "rank_builtin_recipes", '{"limit": 3}'), ("text", "Here you go:\n" + DISHES)])
    _patch_model(monkeypatch, model)
    monkeypatch.setattr(brain, "PROVIDER", "bedrock")
    out = asyncio.run(brain.suggest(ITEMS, "", 3))
    assert out["source"] == "llm:bedrock"
    assert out["dishes"][0]["name"] == "Masala Omelette"
    assert model.calls == 2          # one call that asked for the tool, one that answered


def test_dish_agent_falls_back_to_rules_when_bedrock_fails(monkeypatch):
    _patch_model(monkeypatch, ScriptedModel([("raise", RuntimeError("AccessDeniedException: model not available"))]))
    monkeypatch.setattr(brain, "PROVIDER", "bedrock")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = asyncio.run(brain.suggest(ITEMS, "", 3))
    assert out["source"] == "rules" and out["dishes"]
    assert "AccessDenied" in out["llm_error"]


def test_dish_agent_bad_json_falls_back(monkeypatch):
    _patch_model(monkeypatch, ScriptedModel([("text", "sorry, no idea")]))
    monkeypatch.setattr(brain, "PROVIDER", "bedrock")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert asyncio.run(brain.suggest(ITEMS, "", 3))["source"] == "rules"


def test_simulator_bedrock_brain_calls_mcp_tool_and_traces_it(monkeypatch):
    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append((name, args))
            return SimpleNamespace(structuredContent={"speech": "Yes, you have 2 eggs."}, content=[], isError=False)

    tools = [{"name": "check_item", "description": "Check one item", "input_schema": {
        "type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}]
    _patch_model(monkeypatch, ScriptedModel([("tool", "check_item", '{"item": "eggs"}'), ("text", "Yes, you have two eggs.")]))
    trace = []
    reply = asyncio.run(sim._bedrock([], "do i have eggs?", tools, FakeSession(), trace))
    assert reply == "Yes, you have two eggs."
    assert calls == [("check_item", {"item": "eggs"})]
    assert trace[0]["tool"] == "check_item" and trace[0]["error"] is False and "ms" in trace[0]


def test_cost_estimate():
    assert llm_usage.estimate("global.anthropic.claude-haiku-4-5-20251001-v1:0", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert llm_usage.estimate("global.anthropic.claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)
    assert llm_usage.estimate("some-unknown-model", 10, 10) is None


def test_simulator_bedrock_brain_stops_a_looping_model(monkeypatch):
    """A weak model that keeps calling the same tool must be cut off at SIM_MAX_TOOL_CALLS and still answer."""
    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append(name)
            return SimpleNamespace(structuredContent={"speech": "ok"}, content=[], isError=False)

    tools = [{"name": "get_fridge_inventory", "description": "inventory", "input_schema": {"type": "object", "properties": {}}}]
    monkeypatch.setattr(sim, "MAX_TOOL_CALLS", 3)
    steps = [("tool", "get_fridge_inventory", "{}")] * 8 + [("text", "Here is what I found.")]
    _patch_model(monkeypatch, ScriptedModel(steps))
    reply = asyncio.run(sim._bedrock([], "what's in my fridge?", tools, FakeSession(), []))
    assert len(calls) == 3 and reply


def test_simulator_bedrock_brain_times_out_so_the_chain_can_fall_back(monkeypatch):
    class Slow(ScriptedModel):
        pass
    monkeypatch.setattr(sim, "SIM_RUN_TIMEOUT", 0.05)

    async def never(*a, **k):
        await asyncio.sleep(5)
    import strands
    monkeypatch.setattr(strands.Agent, "invoke_async", never)
    _patch_model(monkeypatch, ScriptedModel([("text", "x")]))
    with pytest.raises(TimeoutError):
        asyncio.run(sim._bedrock([], "hi", [], object(), []))


def test_own_timeout_puts_provider_in_cooldown(monkeypatch):
    import llm_chain as lc
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a"); monkeypatch.setenv("ANTHROPIC_API_KEY", "x"); lc.reset()
    lc.mark_down("bedrock", TimeoutError(), force=True)
    assert lc.chain("bedrock", agent=True) == ["anthropic"]
    lc.reset()
