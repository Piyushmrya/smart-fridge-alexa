# dish_agent/ — the dish-prediction AI agent (block 3)

A separate small web service (`python -m dish_agent.service`, port 8897). The MCP server sends it the
fridge contents; it answers with ranked dishes. The MCP server never depends on it: if it is down or slow,
the MCP server falls back to the same rule-based logic.

| File | What it does |
|---|---|
| `service.py` | FastAPI app: `GET /health`, `POST /suggest`, `POST /recipe`. Validates input, logs, calls `brain`. |
| `brain.py` | The actual thinking: prompt, LLM call, JSON parsing, rule-based fallback. |

## brain.py — important parts
- `PROVIDER`: from `AGENT_LLM_PROVIDER` — `anthropic`, `openai`, or `rules` (no key needed).
- `SYSTEM`: the instruction given to the LLM (Indian home cooking, use soon-to-expire items first,
  assume pantry staples, answer with JSON only).
- `_prompt(items, preferences, n)`: turns the fridge list into the user message.
- `_llm(user)`: the single place that calls the LLM. **The Bedrock/Strands integration will replace this.**
- `_json(text)`: pulls the JSON object out of the model's reply.
- `suggest(items, preferences, n)`: LLM first; on any error or empty answer, `rules()`. Returns
  `{"dishes": [...], "source": "llm:anthropic" | "rules", "llm_error"?}`.
- `rules(items, n)`: offline ranking using `mcp_server/recipes.py`.
- `recipe(dish, items)`: steps for one dish (recipe book first, then the LLM).

Each dish: `name, why, time_min, uses_expiring, have, missing, steps`.

## Settings (.env)
`AGENT_LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `AGENT_ANTHROPIC_MODEL`, `AGENT_OPENAI_*`, `AGENT_PORT`, `LOG_LEVEL`.
