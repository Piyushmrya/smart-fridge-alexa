# simulator/ — the imitation Alexa+ (block 1)

A web page plus an agent that behaves like Alexa+: it is an **MCP client** that discovers the server's
tools, lets an LLM choose which to call, and speaks the answer. Run: `python -m simulator.app`
then open http://127.0.0.1:8898. See `docs/HOW_IT_WORKS.md` for the full flow.

| File | What it does |
|---|---|
| `app.py` | Web server: serves the page and the `/api/*` endpoints the page calls. |
| `agent.py` | The Alexa+ brain: MCP connection, tool list, LLM loop, tracing. |
| `static/index.html` | The whole UI (HTML/CSS/JS): chat, mic, trace panel, fridge panel. |

## app.py — endpoints
- `GET /` — the page.
- `POST /api/chat` — one conversation turn -> `agent.chat()`. Keeps the last 12 messages per `session_id`.
- `POST /api/reset` — forget a session.
- `GET /api/status` — connected? provider? tool list (the green pill).
- `POST /api/redeploy` — like `alexa-ai deploy`: re-reads the tool list. Alexa+ only does this on deploy.
- `GET /api/fridge` — live fridge panel, fetched **through MCP** (same path Alexa+ uses).

## agent.py — important parts
- `mcp_session()`: opens the MCP connection (identifies itself as "Alexa+ MCP Client").
- `deploy()` / `get_tools()`: list and cache the tools.
- `call_tool(session, name, args, trace)`: runs one tool, times it, flags > 500 ms, appends to `trace`.
- `chat(history, user)`: **entry point.** Chooses the brain with
  `fn = {"anthropic": _anthropic, "openai": _openai}.get(PROVIDER, _demo)` and calls it.
- `_anthropic`, `_openai`: the tool loop (LLM asks for tools -> we run them -> LLM answers), max 6 rounds.
- `_demo`: keyword router, no key needed.
- `SYSTEM`: the Alexa+ persona prompt (short spoken sentences, never guess, always call a tool first).

## Settings (.env)
`LLM_PROVIDER` (`demo|anthropic|openai`), `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `MCP_URL`,
`MCP_API_KEY`, `SIM_PORT`, `LOG_LEVEL`.
