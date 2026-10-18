"""For the classifier teammate: how to hand detected items to the MCP server. No dependencies beyond httpx.

    POST {MCP_BASE}/api/inventory          Authorization: Bearer <INGEST_API_KEY>   (only if the server sets one)
    {"source": "vit-v1", "items": [ {"item": "milk", "qty": 1, "confidence": 0.93}, ... ]}

Accepted shapes for `items` (all equivalent):
    ["milk", "eggs", "eggs"]                                  # plain labels; repeats are counted
    [{"item": "milk"}, {"label": "eggs", "count": 6}]         # `item`/`label`/`class`, `qty`/`count`, `confidence`/`score`
Rules: each push is the FULL current snapshot (items missing from the push are treated as removed);
       detections under confidence 0.5 are dropped; do your own smoothing (e.g. seen in 10 straight frames) BEFORE pushing;
       push whenever the snapshot changes or every ~5-10 s.
Run `python classifier_stub/push_example.py milk eggs tomato` to test against a running server.
"""
import os, sys
import httpx

BASE = os.environ.get("MCP_BASE", "http://127.0.0.1:8899")
KEY = os.environ.get("INGEST_API_KEY", "")


def push(items, source="classifier"):
    r = httpx.post(f"{BASE}/api/inventory", json={"source": source, "items": items},
                   headers={"Authorization": f"Bearer {KEY}"} if KEY else None, timeout=5)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print(push(sys.argv[1:] or ["milk", "eggs", "tomato", "onion", "paneer", "spinach"]))
