# How it works: one question, start to finish

Example: you type **"What can I make for dinner?"** in the simulator.

```
Browser (index.html)
  │  POST /api/chat {"message": ..., "session_id": ...}
  ▼
simulator/app.py            @app.post("/api/chat") -> chat()
  │  agent.chat(history, message)
  ▼
simulator/agent.py          chat() -> fn = _anthropic | _openai | _demo
  │  1. asks the LLM (Claude) what to do, giving it the tool list
  │  2. LLM answers "call suggest_recipes"
  │  3. call_tool() -> MCP request over HTTP
  ▼
mcp_server/server.py        @tool() suggest_recipes()
  │  store.load_state() -> edge.fetch()  ──►  friend's camera API  GET /inventory
  │  _dishes() -> POST /suggest          ──►  dish_agent/service.py -> brain.suggest() -> Claude / rules
  ▼
result {dishes, speech}  travels back up the same path
  ▼
simulator/agent.py          LLM sees the result and writes a short spoken reply
  ▼
Browser shows the reply + the trace panel (every tool call with its time)
```

## 1. Browser -> server
- `simulator/static/index.html` is the whole UI (HTML + CSS + JavaScript in one file).
  The function `send(text)` runs when you press Enter / click Send and does
  `fetch('/api/chat', {method:'POST', body: JSON.stringify({message, session_id})})`.
- Python serves that file at `GET /` (`simulator/app.py`, function `index`).
- The URL `/api/chat` is bound to a Python function by the decorator
  `@app.post("/api/chat")` in `simulator/app.py`. FastAPI keeps a table "URL -> function".
  `body: ChatIn` makes FastAPI parse and validate the JSON into the `ChatIn` class.
- `chat()` there loads the last 12 messages of the session (`SESSIONS` dict), calls
  `agent.chat(...)`, stores the exchange, and returns the dictionary as JSON.

Other endpoints the page uses: `/api/status` (green pill), `/api/redeploy` (the "alexa-ai deploy"
button), `/api/fridge` (the live fridge panel, fetched through MCP), `/api/reset` (clear chat).

## 2. `agent.chat` and the `fn` line (simulator/agent.py)

```python
tools = await get_tools()                       # tool list, cached after the first "deploy"
async with mcp_session() as (session, init):    # open a live MCP connection to :8899
    fn = {"anthropic": _anthropic, "openai": _openai}.get(PROVIDER, _demo)
    reply = await fn(history, user, tools, session, trace)
```

* In Python a function is a value. `_anthropic` (no brackets) is the function itself;
  `_anthropic()` would *run* it.
* `{"anthropic": _anthropic, "openai": _openai}` is a dictionary from a name to a function.
* `.get(PROVIDER, _demo)` looks up the name from `.env` (`LLM_PROVIDER`); if it is not found,
  it returns `_demo`.
* So `fn` now holds one of the three functions, and the next line calls it. With
  `LLM_PROVIDER=anthropic` this is exactly `await _anthropic(history, user, tools, session, trace)`.
* All three have the same five parameters, so adding another brain (for example Bedrock) means
  writing one more function with that signature and adding one entry to the dictionary.

`trace` is an empty list that `call_tool()` fills in. The UI shows it, and the logs print from it.

## 3. How `_anthropic` gets a reply (the tool loop)

```python
for round in range(6):
    r = client.messages.create(model, system=SYSTEM, tools=spec, messages=msgs)
    if r.stop_reason != "tool_use":       # Claude wrote a final text answer
        return that text
    # otherwise Claude *asked* for tools: run each, append the results, ask again
```

1. We send Claude: the system prompt ("you are Alexa+, never guess, call a tool first"), the tool
   list (name + description + input schema, taken from the MCP server), and the conversation.
2. Claude cannot see the fridge. It answers with a *request*: "call `suggest_recipes` with `{}`"
   (`stop_reason == "tool_use"`). Claude never runs tools itself.
3. `call_tool(session, name, args, trace)` sends that request over MCP, measures the time, and
   adds `{tool, args, result, ms}` to `trace`.
4. We append Claude's request and the tool result to `msgs` and ask again. Claude now has the data
   and writes a short spoken answer (`stop_reason == "end_turn"`), which becomes `reply`.

Claude may call several tools in a row (for example when something fails it checks the system
status). The loop allows up to 6 rounds.

`_openai` is the same loop in OpenAI's message format. `_demo` uses no LLM: regular expressions
pick a tool from keywords ("dinner" -> `suggest_recipes`) and the tool's `speech` text is the reply.

## 4. Inside the MCP server (mcp_server/server.py)

`session.call_tool("suggest_recipes", {})` arrives over HTTP at `/mcp`. FastMCP finds the Python
function registered with `@tool()` (our logging wrapper around `@mcp.tool()`) under that name and runs it:

1. `store.load_state()` returns the fridge: `edge.fetch()` calls `GET {EDGE_URL}/inventory`
   (2 s cache, 2 s timeout). `store._from_edge()` converts the raw rows into grouped items with an
   estimated number of days left. If the camera API fails, the last good snapshot is returned
   flagged `edge_error`; with no snapshot, an `error` is returned.
2. `_dishes()` computes a fingerprint of the fridge. If the same fridge was seen before it
   returns the cached answer; otherwise it POSTs to the dish agent; if that is down or empty it
   falls back to `dish_agent.brain.rules()`.
3. The tool builds a `speech` sentence and returns a dictionary. FastMCP sends it back as JSON.

A background loop (`_poll_edge`) refreshes the fridge and pre-computes the dishes every
`EDGE_POLL_SECONDS`, so most `suggest_recipes` calls hit the cache and stay fast.

## 5. Inside the dish agent (dish_agent/)
`service.py` exposes `/suggest` and `/recipe` and forwards to `brain.py`:
`_prompt()` turns the items into text, `_llm()` calls Claude with `SYSTEM` (JSON-only answer),
`_json()` extracts the JSON, `suggest()` returns the dishes, and `rules()` is the offline fallback
built on `mcp_server/recipes.py`.

## 6. Why a failure looks the way it does
If the camera API times out: `edge.fetch` raises -> `store.load_state` has no snapshot -> tools
return empty lists plus an `error` -> Claude reads the error and tells you the camera is offline.
That is deliberate: the assistant reports the problem instead of inventing food.

## Reading the logs
Each line is `time  TAG  marker  message`. Tags: `SIM` simulator, `MCP` tool server, `EDGE` camera
API client, `AGENT` dish agent. Markers: `+` success, `x` failure, `!` warning. Set `LOG_LEVEL`
in `.env` to `quiet`, `normal` (default) or `debug` (adds full arguments and results).
