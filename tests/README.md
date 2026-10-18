# tests/

`test_e2e.py` starts real servers on spare ports and checks the whole chain. Run: `python -m pytest -q tests`

| Test | Checks |
|---|---|
| `test_auth_rejects_without_token` | 401 without the bearer token, and no `WWW-Authenticate` header. |
| `test_alexa_style_flow` | Simulator client connects, lists tools, calls them. |
| `test_classifier_ingest_api` | Push-mode ingest API. |
| `test_agent_llm_path_parses_json_and_falls_back` | Dish agent parses LLM JSON and falls back to rules on error (fake LLM, no key needed). |
| `test_pull_mode_inventory_and_correction` | Reads inventory and sends a correction via the mock camera API. |
| `test_pull_mode_edge_offline_degrades_gracefully` | Camera API down -> clear message, no crash. |

No real API keys are needed or used.
