"""End-to-end: spawn the real MCP server, talk to it through the simulator's Alexa+-style client."""
import asyncio, json, os, subprocess, sys, time, shutil, tempfile
from pathlib import Path
import httpx, pytest

ROOT = Path(__file__).resolve().parent.parent
PORT = 18899


@pytest.fixture(scope="module")
def server():
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "shared/fridge_state.json", tmp / "fridge_state.json")
    env = {**os.environ, "MCP_PORT": str(PORT), "FRIDGE_STATE_PATH": str(tmp / "fridge_state.json"),
           "SHOPPING_LIST_PATH": str(tmp / "shop.json"), "MCP_API_KEY": "secret", "MCP_URL": f"http://127.0.0.1:{PORT}/mcp"}
    env.update(INGEST_API_KEY="ingest-secret", AGENT_URL="http://127.0.0.1:18897", AGENT_PORT="18897", AGENT_LLM_PROVIDER="rules")
    agent = subprocess.Popen([sys.executable, "-m", "dish_agent.service"], cwd=ROOT, env=env)
    p = subprocess.Popen([sys.executable, "-m", "mcp_server.server"], cwd=ROOT, env=env)
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health").status_code == 200: break
        except Exception: time.sleep(0.2)
    yield env
    p.terminate(); agent.terminate()


def test_auth_rejects_without_token(server):
    r = httpx.post(f"http://127.0.0.1:{PORT}/mcp", json={})
    assert r.status_code == 401 and "www-authenticate" not in r.headers


def test_alexa_style_flow(server):
    os.environ.update(server); os.environ["LLM_PROVIDER"] = "demo"
    sys.path.insert(0, str(ROOT))
    from simulator import agent
    agent.MCP_URL, agent.MCP_KEY = server["MCP_URL"], "secret"

    async def go():
        info = await agent.deploy()
        assert info["server"] == "smart-fridge" and "suggest_recipes" in info["tools"]
        out = {}
        for q in ["What's in my fridge?", "what's about to expire", "what can I make for dinner",
                  "do I have paneer?", "do I have mango?", "how do I make egg bhurji", "add rice to my shopping list",
                  "read my shopping list"]:
            out[q] = await agent.chat([], q)
        return out
    out = asyncio.run(go())
    assert "milk" in out["What's in my fridge?"]["reply"]
    assert out["do I have paneer?"]["reply"].startswith("Yes")
    assert "don't see" in out["do I have mango?"]["reply"]
    assert "rice" in out["read my shopping list"]["reply"]
    for r in out.values():
        for t in r["trace"]:
            # fridge reads must be fast; tools that call the dish agent get a looser budget (cold start on slow machines)
            limit = 2500 if t["tool"] in ("suggest_recipes", "get_recipe_steps") else 500
            assert not t["error"] and t["ms"] < limit, t
    print(json.dumps({k: v["reply"] for k, v in out.items()}, indent=1))


def test_classifier_ingest_api(server):
    base = f"http://127.0.0.1:{PORT}"
    assert httpx.post(f"{base}/api/inventory", json={"items": ["milk"]}).status_code == 401
    h = {"Authorization": "Bearer ingest-secret"}
    assert httpx.post(f"{base}/api/inventory", headers=h, json={"items": "nope"}).status_code == 400
    r = httpx.post(f"{base}/api/inventory", headers=h, json={"source": "t", "items": [
        "Tomato", "tomatoes", {"label": "Paneer", "score": 0.9}, {"item": "ghost", "confidence": 0.2}, {"item": "ice_cream", "confidence": 0.8}]}).json()
    assert r["ok"] and set(r["items"]) == {"tomato", "paneer", "ice cream"} and r["dropped_low_confidence"] == ["ghost"]
    inv = httpx.get(f"{base}/api/inventory", headers=h).json()
    assert {i["item"]: i["qty"] for i in inv["items"]}["tomato"] == 2
    # the new snapshot is what Alexa+ now sees, and the agent produced dishes from it
    os.environ.update(server); os.environ["LLM_PROVIDER"] = "demo"
    from simulator import agent
    agent.MCP_URL, agent.MCP_KEY = server["MCP_URL"], "secret"
    out = asyncio.run(agent.chat([], "what can I make for dinner?"))
    t = out["trace"][0]
    assert t["tool"] == "suggest_recipes" and t["result"]["source"] == "rules"  # i.e. the agent service answered, not the fallback and t["ms"] < 5000
    assert "milk" not in json.dumps(asyncio.run(agent.chat([], "what's in my fridge?"))["reply"])


def test_agent_llm_path_parses_json_and_falls_back(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from dish_agent import brain
    items = [{"item": "paneer", "qty": 1, "est_days_left": 1}, {"item": "onion", "qty": 2, "est_days_left": 20}]
    monkeypatch.setattr(brain, "PROVIDER", "anthropic")

    async def fake(_, provider=None):
        return 'Sure!\n{"dishes":[{"name":"Paneer Tikka Wrap","why":"Uses paneer today.","time_min":20,"uses_expiring":["paneer"],"have":["paneer","onion"],"missing":[],"steps":["a","b"]}]}'
    monkeypatch.setattr(brain, "_llm", fake)
    r = asyncio.run(brain.suggest(items, "vegetarian", 3))
    assert r["source"] == "llm:anthropic" and r["dishes"][0]["name"] == "Paneer Tikka Wrap"

    async def boom(_, provider=None): raise RuntimeError("no key")
    monkeypatch.setattr(brain, "_llm", boom)
    r = asyncio.run(brain.suggest(items, "", 3))
    assert r["source"] == "rules" and "no key" in r["llm_error"]


# ---------------------------------------------------------------- pull mode (classifier's edge API)
@pytest.fixture(scope="module")
def pull_stack():
    edge_p, mcp_p, ag_p = 19000, 19899, 19897
    tmp = Path(tempfile.mkdtemp())
    env = {**os.environ, "EDGE_PORT": str(edge_p), "EDGE_URL": f"http://127.0.0.1:{edge_p}", "MCP_PORT": str(mcp_p),
           "AGENT_PORT": str(ag_p), "AGENT_URL": f"http://127.0.0.1:{ag_p}", "MCP_API_KEY": "", "INGEST_API_KEY": "",
           "FRIDGE_STATE_PATH": str(tmp / "s.json"), "SHOPPING_LIST_PATH": str(tmp / "shop.json"), "EDGE_CACHE_TTL": "0",
           "MCP_URL": f"http://127.0.0.1:{mcp_p}/mcp", "AGENT_LLM_PROVIDER": "rules"}
    procs = [subprocess.Popen([sys.executable, "-m", m], cwd=ROOT, env=env)
             for m in ("classifier_stub.mock_edge_server", "dish_agent.service", "mcp_server.server")]
    for port in (edge_p + 0, mcp_p):
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/" + ("inventory" if port == edge_p else "health")).status_code == 200: break
            except Exception: time.sleep(0.2)
    yield env, procs
    for p in procs: p.terminate()


def test_pull_mode_inventory_and_correction(pull_stack):
    env, _procs = pull_stack
    os.environ.update(env); os.environ["LLM_PROVIDER"] = "demo"
    sys.path.insert(0, str(ROOT))
    from simulator import agent
    agent.MCP_URL, agent.MCP_KEY = env["MCP_URL"], ""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def go():
        async with agent.mcp_session() as (s, _):
            tr = []
            inv = json.loads(await agent.call_tool(s, "get_fridge_inventory", {}, tr))
            assert any(i["item"] == "parsley" for i in inv["items"]) and "(not sure)" in inv["speech"]
            assert inv["tracked"][0]["id"].startswith("item_") and "time_in_fridge_seconds" in inv["tracked"][0]
            assert any(i["item"] == "eggs" and i["qty"] == 2 for i in inv["items"])  # two tracked eggs aggregate
            # voice-style correction by old label; no ids needed
            res = json.loads(await agent.call_tool(s, "correct_fridge_items", {"corrections": [{"from_label": "parsley", "label": "cilantro"}]}, tr))
            assert res["success"] and res["updated_items"][0]["manual_override"] is True
            inv2 = json.loads(await agent.call_tool(s, "get_fridge_inventory", {"force_refresh": True}, tr))
            assert any(i["item"] == "cilantro" for i in inv2["items"]) and not any(i["item"] == "parsley" for i in inv2["items"])
            bad = json.loads(await agent.call_tool(s, "correct_fridge_items", {"corrections": [{"id": "item_999", "label": "x"}]}, tr))
            assert bad["success"] is False
            st = json.loads(await agent.call_tool(s, "get_system_status", {}, tr))
            assert st["online"] is True and st["mode"] == "pull"
            dishes = json.loads(await agent.call_tool(s, "suggest_recipes", {}, tr))
            assert dishes["dishes"] and dishes["source"] == "rules"
    asyncio.run(go())


def test_pull_mode_edge_offline_degrades_gracefully(pull_stack):
    env, procs = pull_stack
    from simulator import agent
    agent.MCP_URL, agent.MCP_KEY = env["MCP_URL"], ""
    procs[0].terminate(); procs[0].wait()          # camera laptop goes to sleep

    async def go():
        async with agent.mcp_session() as (s, _):
            tr = []
            inv = json.loads(await agent.call_tool(s, "get_fridge_inventory", {}, tr))
            assert inv["items"] and "can't reach the fridge camera" in inv["speech"]   # last known snapshot, flagged
            st = json.loads(await agent.call_tool(s, "get_system_status", {}, tr))
            assert st["online"] is False and "offline" in st["speech"]
            fix = json.loads(await agent.call_tool(s, "correct_fridge_items", {"corrections": [{"id": "item_1", "label": "x"}]}, tr))
            assert fix["success"] is False
    asyncio.run(go())
