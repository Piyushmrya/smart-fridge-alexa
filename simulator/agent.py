"""Imitation of the Alexa+ orchestrator.

What real Alexa+ does (per developer.amazon.com/docs/alexaplus/add-ons):
  * is the MCP *client*; talks JSON-RPC over Streamable HTTP to your server
  * sends `initialize` as "Alexa+ MCP Client" 1.0.0, then discovers tools/resources
  * refreshes tool info only on *deployment*  -> we cache tools and expose /api/redeploy
  * AI reasoning picks tools from the customer's intent, calls `tools/call`, speaks the result
  * expects round trips < 500 ms
This module reproduces that loop with a pluggable LLM (Anthropic / OpenAI-compatible) or an offline keyword router.
"""
import json, os, re, time
from contextlib import asynccontextmanager

import bedrock_util, llm_usage
from fridge_log import log, debug, short
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8899/mcp")
MCP_KEY = os.environ.get("MCP_API_KEY", "")
PROVIDER = os.environ.get("LLM_PROVIDER", "demo").lower()
LATENCY_BUDGET_MS = 500

SYSTEM = (
    "You are Alexa+, a voice assistant in a kitchen. The user is speaking to you, so reply in 1-3 short, natural spoken "
    "sentences: no markdown, no lists, no emojis. You have tools connected to the user's smart fridge. NEVER guess what "
    "is in the fridge -- always call a tool first. Prefer the `speech` field in tool results when it fits. If a recipe "
    "is missing an ingredient, offer to add it to the shopping list."
)

_tools_cache: list[dict] | None = None
_server_info: dict | None = None


@asynccontextmanager
async def mcp_session():
    headers = {"Authorization": f"Bearer {MCP_KEY}"} if MCP_KEY else None
    async with streamablehttp_client(MCP_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w, client_info=Implementation(name="Alexa+ MCP Client", version="1.0.0")) as s:
            init = await s.initialize()
            yield s, init


async def deploy() -> dict:
    """Simulates `alexa-ai deploy`: connect, handshake, register tools."""
    global _tools_cache, _server_info
    async with mcp_session() as (s, init):
        tl = await s.list_tools()
        _tools_cache = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema} for t in tl.tools]
        _server_info = {"server": init.serverInfo.name, "version": init.serverInfo.version,
                        "protocol": init.protocolVersion, "tools": [t["name"] for t in _tools_cache]}
    log("SIM", f"deploy: connected to {_server_info['server']}, registered {len(_server_info['tools'])} tools", "ok")
    return _server_info


async def get_tools() -> list[dict]:
    if _tools_cache is None:
        await deploy()
    return _tools_cache or []


async def call_tool(session: ClientSession, name: str, args: dict, trace: list) -> str:
    t0 = time.perf_counter()
    try:
        res = await session.call_tool(name, args)
        data = res.structuredContent
        if data is None and res.content:
            txt = getattr(res.content[0], "text", "")
            try: data = json.loads(txt)
            except Exception: data = {"text": txt}
        err = bool(res.isError)
    except Exception as e:
        data, err = {"error": str(e)}, True
    ms = round((time.perf_counter() - t0) * 1000)
    log("SIM", f"tool {name} -> {'ERROR' if err else 'ok'} ({ms} ms{', over 500 ms budget' if ms > LATENCY_BUDGET_MS else ''})", "fail" if err else "ok")
    debug("SIM", f"tool {name} args={short(args)} result={short(data, 400)}")
    trace.append({"tool": name, "args": args, "result": data, "ms": ms, "over_budget": ms > LATENCY_BUDGET_MS, "error": err})
    return json.dumps(data)


# ---------------------------------------------------------------- LLM back-ends
async def _anthropic(history, user, tools, session, trace):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    spec = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools]
    msgs = history + [{"role": "user", "content": user}]
    for _ in range(6):
        log("SIM", f"asking Claude ({model}), round {_ + 1}")
        r = await client.messages.create(model=model, max_tokens=600, system=SYSTEM, tools=spec, messages=msgs)
        llm_usage.record("SIM", model, r.usage.input_tokens, r.usage.output_tokens, what="anthropic round")
        wanted = [b.name for b in r.content if b.type == "tool_use"]
        if wanted:
            log("SIM", f"Claude wants tool(s): {', '.join(wanted)}")
        if r.stop_reason != "tool_use":
            log("SIM", "Claude gave the final answer")
            return "".join(b.text for b in r.content if b.type == "text").strip()
        msgs.append({"role": "assistant", "content": [b.model_dump() for b in r.content]})
        results = []
        for b in r.content:
            if b.type == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": await call_tool(session, b.name, b.input, trace)})
        msgs.append({"role": "user", "content": results})
    return "Sorry, that took too many steps."


async def _openai(history, user, tools, session, trace):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    spec = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
    msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": user}]
    for _ in range(6):
        log("SIM", f"asking OpenAI-compatible model {model}, round {_ + 1}")
        r = (await client.chat.completions.create(model=model, messages=msgs, tools=spec)).choices[0].message
        if r.tool_calls:
            log("SIM", f"model wants tool(s): {', '.join(tc.function.name for tc in r.tool_calls)}")
        if not r.tool_calls:
            return (r.content or "").strip()
        msgs.append(r.model_dump(exclude_none=True))
        for tc in r.tool_calls:
            out = await call_tool(session, tc.function.name, json.loads(tc.function.arguments or "{}"), trace)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "Sorry, that took too many steps."


def _strands_tool(spec: dict, session, trace: list):
    """Wrap one MCP tool as a Strands tool that goes through call_tool(), so tracing/timing/logging are identical for every brain."""
    from strands.types._events import ToolResultEvent
    from strands.types.tools import AgentTool

    class McpTool(AgentTool):
        @property
        def tool_name(self): return spec["name"]
        @property
        def tool_spec(self): return {"name": spec["name"], "description": spec["description"] or spec["name"],
                                     "inputSchema": {"json": spec["input_schema"]}}
        @property
        def tool_type(self): return "mcp"
        async def stream(self, tool_use, invocation_state, **kwargs):
            out = await call_tool(session, spec["name"], tool_use["input"], trace)
            failed = bool(trace and trace[-1]["tool"] == spec["name"] and trace[-1]["error"])
            yield ToolResultEvent({"toolUseId": tool_use["toolUseId"], "status": "error" if failed else "success",
                                   "content": [{"text": out}]})
    return McpTool()


async def _bedrock(history, user, tools, session, trace):
    """Alexa+ brain = a Strands agent on Amazon Bedrock whose tools are our MCP server's tools."""
    from strands import Agent
    model = bedrock_util.make_model("sim", max_tokens=600)
    mid = bedrock_util.model_id(model)
    log("SIM", f"asking Bedrock ({mid}) via Strands, {len(tools)} MCP tools available")
    msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history if m.get("content")]
    agent = Agent(model=model, tools=[_strands_tool(t, session, trace) for t in tools], system_prompt=SYSTEM,
                  messages=msgs, callback_handler=None)
    res = await agent.invoke_async(user)
    llm_usage.record("SIM", mid, *llm_usage.usage_from_result(res), what="bedrock turn")
    return str(res).strip()


async def _demo(history, user, tools, session, trace):
    """Offline keyword router so the project runs with zero API keys. Not smart -- just enough for a demo/test."""
    u = user.lower().strip()
    names = {t["name"] for t in tools}
    def run(name, **a):
        return call_tool(session, name, a, trace)
    out = None
    if m := re.search(r"add (.+?) to (?:my |the )?(?:shopping|grocery)", u):
        out = await run("add_to_shopping_list", item=m.group(1))
    elif m := re.search(r"remove (.+?) from (?:my |the )?(?:shopping|grocery)", u):
        out = await run("remove_from_shopping_list", item=m.group(1))
    elif re.search(r"shopping|grocery", u):
        out = await run("get_shopping_list")
    elif m := re.search(r"(?:steps|how (?:do i|to) (?:make|cook)|recipe for) (?:for )?(.+?)\??$", u):
        out = await run("get_recipe_steps", dish=m.group(1))
    elif re.search(r"expir|going bad|spoil|use up|fresh", u):
        out = await run("get_expiring_items")
    elif re.search(r"cook|make|dinner|lunch|breakfast|hungry|eat|recipe|hungry", u):
        out = await run("suggest_recipes")
    elif m := re.search(r"(?:do i have|have i got|is there|any) (?:any |some )?(?:more )?([a-z ]+?)(?: in (?:my|the) fridge)?\??$", u):
        out = await run("check_item", item=m.group(1))
    elif re.search(r"fridge|inventory|what do i have|what.s in", u):
        out = await run("get_fridge_inventory")
    if out is None:
        return "I can tell you what's in your fridge, what's about to expire, or what to cook. What would you like?"
    return json.loads(out).get("speech") or "Done."


async def chat(history: list[dict], user: str) -> dict:
    t0 = time.perf_counter()
    log("SIM", f'you said: "{short(user, 120)}"  (brain: {PROVIDER})')
    tools = await get_tools()
    trace: list[dict] = []
    async with mcp_session() as (session, init):
        fn = {"bedrock": _bedrock, "anthropic": _anthropic, "openai": _openai}.get(PROVIDER, _demo)
        try:
            try:
                reply = await fn(history, user, tools, session, trace)
            except Exception as e:
                if PROVIDER != "bedrock" or not os.environ.get("ANTHROPIC_API_KEY"):
                    raise
                log("SIM", f"Bedrock failed ({type(e).__name__}: {short(e, 120)}) -> retrying with the Anthropic API", "warn")
                trace.clear()
                reply = await _anthropic(history, user, tools, session, trace)
        except Exception as e:  # missing key, network, etc.
            reply = f"Sorry, my brain hit an error: {type(e).__name__}: {e}"
            log("SIM", f"brain error: {type(e).__name__}: {short(e, 160)}", "fail")
    log("SIM", f'reply in {(time.perf_counter() - t0) * 1000:.0f} ms, {len(trace)} tool call(s): "{short(reply, 100)}"', "fail" if reply.startswith("Sorry, my brain") else "ok")
    return {"reply": reply, "trace": trace, "provider": PROVIDER if PROVIDER in ("bedrock", "anthropic", "openai") else "demo",
            "total_ms": round((time.perf_counter() - t0) * 1000)}
