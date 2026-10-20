"""Tiny Amazon Bedrock smoke test: one very small call per model, prints latency, tokens and estimated cost.

    python scripts/check_bedrock.py            # uses .env (AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, BEDROCK_*_MODEL_ID)

Costs a fraction of a cent. Never prints your key.
"""
import asyncio, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import bedrock_util, llm_usage  # noqa: E402


def explain(e: Exception) -> str:
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "accessdenied" in low or "not available for this account" in low:
        return s + "\n  -> This model is not enabled for your account. Try another model id, or the Bedrock playground."
    if "validationexception" in low and "model identifier" in low:
        return s + "\n  -> Wrong model id. Copy the exact id from the Bedrock console (Inference profiles)."
    if "unrecognizedclient" in low or "security token" in low or "invalidsignature" in low or "expired" in low or "bearer" in low:
        return s + "\n  -> Key problem: expired or wrong AWS_BEARER_TOKEN_BEDROCK. Create a new key in Bedrock > API keys."
    if "nocredentials" in low or "unable to locate credentials" in low:
        return s + "\n  -> No key found. Put AWS_BEARER_TOKEN_BEDROCK=... in .env (same folder as README.md)."
    return s


async def one(kind: str) -> bool:
    from strands import Agent
    model = bedrock_util.make_model(kind, max_tokens=60)
    mid = bedrock_util.model_id(model)
    t0 = time.perf_counter()
    try:
        res = await Agent(model=model, system_prompt="Reply in at most 6 words.", callback_handler=None).invoke_async("Say hello to a fridge.")
    except Exception as e:
        print(f"[FAIL] {kind:<5} {mid}\n  {explain(e)}")
        return False
    ms = (time.perf_counter() - t0) * 1000
    tin, tout = llm_usage.usage_from_result(res)
    cost = llm_usage.estimate(mid, tin, tout)
    print(f"[ OK ] {kind:<5} {mid}\n  reply: {str(res).strip()!r}  |  {ms:.0f} ms  |  {tin} in / {tout} out tokens  |  ~${cost:.5f}" if cost is not None
          else f"[ OK ] {kind:<5} {mid}  reply: {str(res).strip()!r}  {ms:.0f} ms")
    return True


async def main() -> int:
    print(f"region: {os.environ.get('AWS_REGION', 'us-east-1')}   key set: {'yes' if os.environ.get('AWS_BEARER_TOKEN_BEDROCK') else 'NO (using normal AWS credentials, if any)'}")
    results = [await one("agent"), await one("sim")]
    print("\nAll good - set AGENT_LLM_PROVIDER=bedrock and LLM_PROVIDER=bedrock in .env." if all(results) else "\nSomething failed - see the messages above.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
