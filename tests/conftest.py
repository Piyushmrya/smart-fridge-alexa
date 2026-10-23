"""Make the test run hermetic: a developer's real .env (EDGE_URL, API keys, provider choices)
must never leak into tests. python-dotenv does not override variables that already exist,
so setting them here (before any project module is imported) wins over .env."""
import os

_HERMETIC = {
    "EDGE_URL": "", "EDGE_API_KEY": "",
    "LLM_PROVIDER": "demo", "AGENT_LLM_PROVIDER": "rules",
    "ANTHROPIC_API_KEY": "", "AGENT_OPENAI_API_KEY": "", "OPENAI_API_KEY": "",
    "AWS_BEARER_TOKEN_BEDROCK": "",
    "MCP_API_KEY": "", "INGEST_API_KEY": "",
    "AGENT_TIMEOUT": "6",
    "OPENAI_BASE_URL": "", "AGENT_OPENAI_BASE_URL": "",   # client caches in classifier tests key on the base URL
}
for _k, _v in _HERMETIC.items():
    os.environ[_k] = _v
