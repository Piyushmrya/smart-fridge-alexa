"""Step-by-step Bedrock diagnosis. Run:  python scripts/bedrock_diag.py [region]
Uses credentials from .env / the environment (IAM keys or a Bedrock API key). Never prints secrets."""
import os, sys, time
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass
import boto3
from botocore.exceptions import ClientError

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ALL = "--all" in sys.argv
region = args[0] if args else os.environ.get("AWS_REGION", "us-east-1")
print(f"region: {region}")
print("auth: bearer token" if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") else "auth: AWS access keys / default chain (no bearer token)")
if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "":
    print("  note: AWS_BEARER_TOKEN_BEDROCK is set but empty -> delete that line from .env")

def short(e):
    s = str(e); return s if len(s) < 330 else s[:330] + " ..."

# 1. who am I
try:
    ident = boto3.client("sts", region_name=region).get_caller_identity()
    print(f"[1] identity OK: {ident['Arn']}")
except Exception as e:
    print("[1] identity FAILED:", short(e)); sys.exit("Fix credentials first (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).")

br = boto3.client("bedrock", region_name=region)
rt = boto3.client("bedrock-runtime", region_name=region)

# 2. what can I see
profiles, models = [], []
try:
    profiles = [p["inferenceProfileId"] for p in br.list_inference_profiles()["inferenceProfileSummaries"]]
    print(f"[2] inference profiles visible: {len(profiles)}")
except Exception as e:
    print("[2] list_inference_profiles FAILED:", short(e))
text_ondemand = []
try:
    ms = br.list_foundation_models()["modelSummaries"]
    models = [m["modelId"] for m in ms]
    text_ondemand = [m["modelId"] for m in ms if "TEXT" in m.get("outputModalities", []) and "ON_DEMAND" in m.get("inferenceTypesSupported", [])
                     and "TEXT" in m.get("inputModalities", [])]
    print(f"    foundation models visible: {len(models)}")
except Exception as e:
    print("    list_foundation_models FAILED:", short(e))

def pick(sub): return [i for i in profiles + models if sub in i]
cands = []
for sub in ("nova-lite", "nova-pro", "nova-micro", "nova-2-lite", "claude-haiku-4-5", "claude-sonnet-4-6"):
    cands += pick(sub)
if ALL:
    text_profiles = [p for p in profiles if not any(x in p for x in ("embed", "image", "video", "rerank", "stable", "sonic", "canvas", "reel"))]
    cands = [c for c in dict.fromkeys(text_profiles + text_ondemand) if "anthropic" not in c and ":24k" not in c and ":300k" not in c and ":128k" not in c and ":256k" not in c]
cands = list(dict.fromkeys(cands))
print("[3] candidate ids to try:", cands or "none found")

# 4. real test call per candidate (parallel)
from concurrent.futures import ThreadPoolExecutor
TOOL = {"tools": [{"toolSpec": {"name": "get_time", "description": "Get the time", "inputSchema": {"json": {"type": "object", "properties": {}}}}}]}

def probe(mid):
    t = time.time()
    try:
        r = rt.converse(modelId=mid, messages=[{"role": "user", "content": [{"text": "Say OK"}]}], inferenceConfig={"maxTokens": 10})
        txt = r["output"]["message"]["content"][0].get("text", "")
    except ClientError as e:
        return mid, False, f"{e.response['Error']['Code']}: {short(e.response['Error']['Message'])}", False
    except Exception as e:
        return mid, False, short(e), False
    tools = True
    try:
        rt.converse(modelId=mid, messages=[{"role": "user", "content": [{"text": "What time is it? use the tool"}]}],
                    toolConfig=TOOL, inferenceConfig={"maxTokens": 50})
    except Exception:
        tools = False
    return mid, True, f"({time.time()-t:.1f}s) -> {txt!r}", tools

ok, errs = [], {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for mid, good, msg, tools in ex.map(probe, cands):
        if good:
            print(f"  OK    {mid}  tools={'yes' if tools else 'NO'}  {msg}"); ok.append((mid, tools))
        else:
            key = msg.split(":")[0] if not ALL else msg[:60]
            errs.setdefault(msg.split(":")[0], []).append(mid)
            if not ALL: print(f"  FAIL  {mid}\n        {msg}")
if ALL:
    print("\nfailures by reason:")
    for k, v in errs.items(): print(f"  {k}: {len(v)} models (e.g. {v[0]})")

print("\nWORKING MODELS:", [m for m, _ in ok] or "none")
good = [m for m, t in ok if t]
print("WORKING WITH TOOLS (needed by the agents):", good or "none")
if good:
    print("Put two of these in .env as BEDROCK_AGENT_MODEL_ID / BEDROCK_SIM_MODEL_ID, then run scripts/check_bedrock.py")
