"""One place that builds the Amazon Bedrock model used by the Strands agents (dish agent and simulator).

Auth: a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK (preferred, expiring) or normal AWS credentials.
Never log or print the key.
"""
import os

DEFAULT_AGENT_MODEL = "global.anthropic.claude-sonnet-4-6"                    # dish agent (quality)
DEFAULT_SIM_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"         # simulator (speed)


def make_model(kind: str = "agent", max_tokens: int = 1500, temperature: float = 0.3):
    """kind: 'agent' (dish agent) or 'sim' (Alexa+ simulator). Model ids come from .env."""
    from strands.models import BedrockModel
    env, default = (("BEDROCK_AGENT_MODEL_ID", DEFAULT_AGENT_MODEL) if kind == "agent"
                    else ("BEDROCK_SIM_MODEL_ID", DEFAULT_SIM_MODEL))
    kw = {}
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if key:
        kw["api_key"] = key
    return BedrockModel(model_id=os.environ.get(env, default), region_name=os.environ.get("AWS_REGION", "us-east-1"),
                        max_tokens=max_tokens, temperature=temperature, **kw)


def model_id(model) -> str:
    try:
        return model.get_config().get("model_id", "?")
    except Exception:
        return "?"
