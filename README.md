# Smart Fridge x Alexa+

Ask an Alexa+ style assistant what is in your fridge and what to cook. A camera image of the fridge is read by a
vision model; the assistant answers through MCP tools; a dish agent suggests meals that use up what is about to expire.

```
 [4] CLASSIFIER + EDGE API  :9000   camera image -> vision model -> GET/PATCH /inventory
        ^  MCP server pulls (and PATCHes corrections)
        |
 [2] MCP SERVER  :8899  ---- /mcp (Streamable HTTP) <---- [1] ALEXA+ SIMULATOR :8898 (web UI + LLM + MCP client)
        |  POST /suggest, /recipe (cached, timeout, rule-based fallback)
        v
 [3] DISH AGENT  :8897  (LLM -> ranked dishes, prefers items about to expire)
```

| Part | Code |
|---|---|
| 1. Alexa+ simulator (web UI, LLM agent, MCP client) | `simulator/` |
| 2. MCP server (11 tools, caching, background refresh) | `mcp_server/` |
| 3. Dish-prediction agent | `dish_agent/` |
| 4. Image classifier + Edge API | `classifier/` |

## Quick start (about 5 minutes, one API key)

Needs Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/). Run everything from the project root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r classifier/requirements.txt
cp .env.example .env          # Windows: copy .env.example .env   -> then put your key in ANTHROPIC_API_KEY=
```

Start four processes, each in its own terminal (venv active):

```bash
python -m classifier.api.main    # [4] camera/classifier API :9000 (first sync takes ~10-15 s)
python -m mcp_server.server      # [2] MCP server           :8899
python -m dish_agent.service     # [3] dish agent           :8897
python -m simulator.app          # [1] simulator            :8898  -> open http://127.0.0.1:8898
```

Try: "What's in my fridge?", "What's about to expire?", "What can I make for dinner?", "Something vegetarian and quick",
"How do I make <dish>?". The classifier reads `classifier/image_model/sample_fridge.png` (a stand-in for the camera).
Tests need no keys or network: `python -m pytest -q tests`.

## Configuration (`.env`)

Everything has a default except the API key. See `.env.example` for every setting with comments.

| Setting | Required? | Default | What it does |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Required** | none | Used by the classifier, the dish agent and the simulator |
| `LLM_PROVIDER` | optional | `anthropic` | Simulator brain: `anthropic`, `bedrock`, `openai`, `auto`, or `demo` (no key) |
| `AGENT_LLM_PROVIDER` | optional | `anthropic` | Dish agent: same options, or `rules` (no key, 14 built-in recipes) |
| `EDGE_URL` | optional | `http://127.0.0.1:9000` | Where the classifier API runs; point it at another machine if needed |
| `AGENT_TIMEOUT`, `AGENT_RUN_TIMEOUT`, `SIM_RUN_TIMEOUT`, `SIM_MAX_TOOL_CALLS`, `LLM_COOLDOWN_SECONDS` | optional | `20`, `8`, `12`, `6`, `300` | Time and call limits that keep answers fast |
| `LOG_LEVEL` | optional | `normal` | `quiet`, `normal` or `debug` |
| `MCP_API_KEY`, `INGEST_API_KEY`, `EDGE_API_KEY` | optional | empty | Bearer tokens; leave empty locally |
| `AWS_BEARER_TOKEN_BEDROCK` or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_*_MODEL_ID` | optional | region `us-east-1` | Only for the Amazon Bedrock path |
| `OPENAI_API_KEY`, `AGENT_OPENAI_API_KEY` | optional | empty | Extra fallback provider |

**No key at all:** the simulator uses its keyword router (`demo`) and the dish agent uses built-in rules, so the MCP server,
simulator and tests run. The classifier needs a key to detect items from an image.

### Amazon Bedrock (optional, Strands Agents SDK)

The dish agent and the simulator both have a Strands agent path on Amazon Bedrock (Claude Sonnet 4.6 for dishes, Claude
Haiku 4.5 for the simulator). To use it, add AWS credentials, set `LLM_PROVIDER` and `AGENT_LLM_PROVIDER` to `bedrock` or
`auto`, and run:

```bash
python scripts/check_bedrock.py        # one tiny call per configured model; prints latency, tokens and cost
python scripts/bedrock_diag.py --all   # lists which models your account can actually call (and which support tools)
```

`auto` tries Bedrock first (when credentials exist), then Anthropic, then OpenAI, then the built-in fallback. A provider that
errors on access, quota or auth, or that runs too long, is skipped for 5 minutes, so one broken service costs one slow call.
Bedrock access for Anthropic models can require a Marketplace subscription, a one-time use-case form and quota; the scripts
explain which one is missing. See `llm_chain.py`.

Without `EDGE_URL` the MCP server runs in legacy **push mode** (see `classifier_stub/push_example.py`); for a fake camera API:
`python -m classifier_stub.mock_edge_server`.

## Where to read more
`docs/HOW_IT_WORKS.md` (full request walkthrough) and a `README.md` inside each folder:
`mcp_server/`, `dish_agent/`, `simulator/`, `classifier/api/`, `classifier_stub/`, `tests/`, `addon/`, `shared/`.

## The contract with the classifier (two endpoints)
**`GET /inventory?force_refresh=true|false`** returns `last_updated_epoch` and `items` (`id`, `label`, `confidence`, `time_in_fridge_seconds`, `manual_override`).
**`PATCH /inventory`** takes a JSON array of partial updates (`id` required; `label`, `confidence`, `time_in_fridge_seconds`, `manual_override` optional) and returns `{"success": true, "updated_items": [...]}`.
Every tracked object is one row (two eggs = two rows); the MCP server groups by label and normalizes names (`red onion` -> `onion`). Freshness comes from `time_in_fridge_seconds` plus a shelf-life table (`mcp_server/store.py`).

## MCP tools
`get_fridge_inventory(force_refresh)`, `correct_fridge_items` (voice fixes like "that's cilantro, not parsley" → PATCH), `get_system_status`, `check_item`, `get_expiring_items`, `suggest_recipes(max_results, preferences)` → asks the dish agent, `get_recipe_steps`, `search_online_recipes` (TheMealDB), `add_to_shopping_list`, `get_shopping_list`, `remove_from_shopping_list`; resource `fridge://inventory`. Every result has structured data + a short `speech` string.

## Latency design (Alexa+ expects < 500 ms per tool call)
An LLM takes 1–3 s, so: (1) the MCP server **prewarms** (on every ingest, or by polling the edge API every 10 s in pull mode) the agent's answer into a cache; (2) `suggest_recipes` serves from cache when the fridge hasn't changed; (3) if the agent is slow/down it falls back to the built-in rules after `AGENT_TIMEOUT`. The simulator flags any call over 500 ms.

## What the simulator mirrors from real Alexa+
Alexa+ is the MCP **client** over Streamable HTTP; handshake as "Alexa+ MCP Client" 1.0.0; tools are registered at **deploy** (button re-registers); results carry `structuredContent` + JSON text; 500 ms budget; account linking = OAuth 2.1 + PKCE with a Bearer token on each call (here: static bearer only, 401 without `WWW-Authenticate`; full OAuth not implemented). Real deployment (US-only toolkit): tunnel with `cloudflared`, `alexa-ai new mcp … --mcp-server-url`, `alexa-ai deploy` — see `addon/addon.json`.
