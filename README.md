# Smart Fridge × Alexa+ — four-part architecture

```
 [4] CLASSIFIER + EDGE API (teammate; ViT/CNN on IoT camera images)   GET/PATCH /inventory
        ▲  MCP server PULLS (and PATCHes corrections)
        │
 [2] MCP SERVER  :8899  ──── /mcp (Streamable HTTP, JSON-RPC) ◀──── [1] ALEXA+ SIMULATOR :8898 (web UI + LLM + MCP client)
        │  POST /suggest, /recipe (cached, 6 s timeout, rule-based fallback)
        ▼
 [3] DISH AI AGENT  :8897  (LLM → ranked dishes, prefers items about to expire)
```
| Part | Owner | Code |
|---|---|---|
| 1. Alexa+ simulator | you | `simulator/` |
| 2. MCP server (+ ingest API) | you | `mcp_server/` |
| 3. Dish-prediction AI agent | you | `dish_agent/` |
| 4. Image classifier + edge API | teammate | `classifier/api/` & `classifier/image_model/` (real model; run `python -m classifier.api.main`) or `classifier_stub/mock_edge_server.py` (mock) |

## Run
```bash
pip install -r requirements.txt            # simulator + MCP server + dish agent
# only if you run the real vision classifier: pip install -r classifier/requirements.txt (adds PyTorch, ~2 GB)
cp .env.example .env        # (Windows: copy .env.example .env) then edit it; it loads automatically
python -m classifier.api.main        # [4] :9000   (real vision classifier & edge API)
python -m mcp_server.server          # [2] :8899   (run each in its own terminal, from the project root)
python -m dish_agent.service         # [3] :8897
python -m simulator.app              # [1] :8898  -> open http://127.0.0.1:8898
python -m pytest -q tests
```
Use `python -m package.module` from the project root (not a file path).
To run the real vision detection model: `python -m classifier.api.main` and set `EDGE_URL=http://127.0.0.1:9000`.
For a fake camera mock API without PyTorch: `python -m classifier_stub.mock_edge_server`.
No keys are needed: the agent defaults to `rules` and the simulator to `demo`. For real LLMs set
`AGENT_LLM_PROVIDER=anthropic` and `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`.
Without `EDGE_URL` the server runs in legacy **push mode** (see `classifier_stub/push_example.py`).
Logs: one clean line per step (`LOG_LEVEL=quiet|normal|debug`).

## Where to read more
`docs/HOW_IT_WORKS.md` (full request walkthrough) and a `README.md` inside each folder:
`mcp_server/`, `dish_agent/`, `simulator/`, `classifier_stub/`, `tests/`, `addon/`, `shared/`.

## The contract with the classifier teammate (final, from his chat) — two endpoints on HIS machine
**`GET /inventory?force_refresh=true|false`** — `force_refresh=true` = grab a fresh frame now and bypass smoothing.
```json
{"last_updated_epoch": 1791853200,
 "items": [{"id": "item_1", "label": "milk", "confidence": 0.94, "time_in_fridge_seconds": 18400, "manual_override": false}]}
```
**`PATCH /inventory`** — batch partial update; body is a JSON array, `id` required, everything else optional (`label`, `confidence`, `time_in_fridge_seconds`, `manual_override`).
```json
[{"id": "item_2", "label": "cilantro", "confidence": 1.0, "manual_override": true}, {"id": "item_1", "time_in_fridge_seconds": 0}]
```
→ `200 {"success": true, "updated_items": [...]}`; errors `400/404 {"success": false, "error": "..."}`.
He must make his tracker honour `manual_override` (don't let the model flip a corrected item back). Optional `Authorization: Bearer` is sent if you set `EDGE_API_KEY`. There is no third "status" endpoint: `get_system_status` simply tests `GET /inventory`.
Notes: every tracked *object* is one row (two eggs = two rows); the MCP server groups by label. Labels should be common food names. Freshness is estimated from `time_in_fridge_seconds` plus a shelf-life table (`mcp_server/store.py`).

## MCP tools
`get_fridge_inventory(force_refresh)`, `correct_fridge_items` (voice fixes like "that's cilantro, not parsley" → PATCH), `get_system_status`, `check_item`, `get_expiring_items`, `suggest_recipes(max_results, preferences)` → asks the dish agent, `get_recipe_steps`, `search_online_recipes` (TheMealDB), `add_to_shopping_list`, `get_shopping_list`, `remove_from_shopping_list`; resource `fridge://inventory`. Every result has structured data + a short `speech` string.

## Latency design (Alexa+ expects < 500 ms per tool call)
An LLM takes 1–3 s, so: (1) the MCP server **prewarms** (on every ingest, or by polling the edge API every 10 s in pull mode) the agent's answer into a cache; (2) `suggest_recipes` serves from cache when the fridge hasn't changed; (3) if the agent is slow/down it falls back to the built-in rules after `AGENT_TIMEOUT`. The simulator flags any call over 500 ms.

## What the simulator mirrors from real Alexa+
Alexa+ is the MCP **client** over Streamable HTTP; handshake as "Alexa+ MCP Client" 1.0.0; tools are registered at **deploy** (button re-registers); results carry `structuredContent` + JSON text; 500 ms budget; account linking = OAuth 2.1 + PKCE with a Bearer token on each call (here: static bearer only, 401 without `WWW-Authenticate`; full OAuth not implemented). Real deployment (US-only toolkit): tunnel with `cloudflared`, `alexa-ai new mcp … --mcp-server-url`, `alexa-ai deploy` — see `addon/addon.json`.

## Submission reminders
Demo video < 3 min, GitHub repo, and written **product feedback on Amazon's MCP tools** — jot friction down as you go.

## Design note: who picks the dishes?
Your friend's rule of thumb — *tools provide facts, the agent reasons* — is already satisfied: the Alexa+ LLM can call `get_fridge_inventory` and reason about dishes itself, with full conversational flexibility ("vegan", "under 15 min"). `suggest_recipes` is an extra helper that delegates to your dish agent (cached, with rule-based fallback). If you'd rather drop it, delete that tool; nothing else depends on it.
