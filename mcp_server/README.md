# mcp_server/ — the MCP server (block 2)

The thing Alexa+ (or our simulator) connects to. It exposes the fridge as **tools** over MCP
(Streamable HTTP, `http://127.0.0.1:8899/mcp`). Run: `python -m mcp_server.server`.
See `docs/HOW_IT_WORKS.md` for the full request flow.

| File | What it does |
|---|---|
| `server.py` | Defines every tool, the HTTP routes, auth and the background poller. |
| `store.py` | Turns raw fridge data into the internal state; shelf-life estimates; shopping-list files. |
| `edge.py` | HTTP client for the classifier teammate's API (`GET`/`PATCH /inventory`). |
| `recipes.py` | The 14 built-in recipes and the ranking used by the offline fallback. |

## server.py — important parts
- `tool()`: our decorator. Registers a function as an MCP tool (`mcp.tool()`) and logs each call
  (name, arguments, duration, success/failure). The **docstring of each tool is what the LLM reads** to
  decide when to call it, so write them carefully.
- Tools: `get_fridge_inventory`, `correct_fridge_items`, `get_system_status`, `check_item`,
  `get_expiring_items`, `suggest_recipes`, `get_recipe_steps`, `search_online_recipes`,
  `add_to_shopping_list`, `get_shopping_list`, `remove_from_shopping_list`.
  Every tool returns data plus a short `speech` sentence, because Alexa speaks it.
- `_dishes(state, prefs, n)`: cache -> dish agent (`POST /suggest`) -> local rules. Never fails. Results from the rules fallback are not cached, so a slow or failed LLM call is retried next time.
- `_ask_agent(path, payload)`: HTTP call to the dish agent with a timeout (`AGENT_TIMEOUT`).
- `_poll_edge()`: background loop; keeps the fridge fresh and the dish cache warm (`EDGE_POLL_SECONDS`).
- `BearerAuth`: checks `Authorization: Bearer ...` when `MCP_API_KEY` / `INGEST_API_KEY` are set;
  returns 401 without `WWW-Authenticate` (as Amazon documents).
- `build_app()`: assembles the web app (MCP + routes + auth + poller).
- Routes besides `/mcp`: `GET /health`, `POST|GET /api/inventory` (legacy push mode).
- `_stale_note(state)`: adds "(I can't reach the camera…)" to speech when data is old.

## store.py — important parts
- `norm(name)`: cleans labels ("Curd", "tomatoes" -> `yogurt`, `tomato`) using `SYNONYMS`.
- `SHELF_LIFE`: typical days per food. `est_days_left = shelf life − days in fridge`.
- `_from_edge(raw)`: friend's JSON (one row per object) -> `items` (grouped by label) + `tracked` (with ids).
- `load_state(force_refresh)`: **what every tool calls.** Pull mode if `EDGE_URL` is set; falls back
  to the last good snapshot (flag `edge_error`) or returns an `error`. Never raises.
- `ingest()`, `_load_file_state()`: legacy push mode (classifier POSTs to us; state in `shared/fridge_state.json`).
- `load_shopping()` / `save_shopping()`: `shared/shopping_list.json`.

## edge.py — important parts
- `ENABLED`: true when `EDGE_URL` is set.
- `fetch(force_refresh)`: `GET /inventory`, 2 s cache (`EDGE_CACHE_TTL`), timeout `EDGE_TIMEOUT`.
  Logs only when the camera goes offline/online, not on every poll.
- `patch(updates)`: `PATCH /inventory` for corrections; clears the cache.
- `last_good()`, `last_latency_ms()`: used for fallbacks and status.

## recipes.py
`RECIPES` (name, required, optional, time, steps), `rank_recipes(have, expiring, limit)`, `find(name)`.

## Settings (.env)
`EDGE_URL`, `EDGE_API_KEY`, `EDGE_TIMEOUT`, `EDGE_POLL_SECONDS`, `MCP_PORT`, `MCP_API_KEY`,
`INGEST_API_KEY`, `AGENT_URL`, `AGENT_TIMEOUT`, `LOG_LEVEL`.
