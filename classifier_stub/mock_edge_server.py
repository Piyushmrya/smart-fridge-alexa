"""Reference/mock implementation of the classifier teammate's EDGE API (the agreed contract).
Use it to develop the MCP side before the real classifier exists -- and as the spec the real one must match.

    python -m classifier_stub.mock_edge_server        -> http://127.0.0.1:9000
    GET   /inventory?force_refresh=true|false
    PATCH /inventory          body: [{"id": "...", "label"?, "confidence"?, "time_in_fridge_seconds"?, "manual_override"?}]
    PUT   /_mock/items        (mock only) body: ["milk","eggs",...]  -> replace the fridge contents for demos
Errors: 400/404 -> {"success": false, "error": "..."}
"""
import os, random, time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock edge (classifier) API")
START = ["milk", "eggs", "eggs", "spinach", "tomato", "tomato", "onion", "paneer", "cheese", "bread", "parsley"]
ITEMS: dict[str, dict] = {}
_n = 0
FIELDS = {"label": str, "confidence": (int, float), "time_in_fridge_seconds": int, "manual_override": bool}


def _add(label: str, age: int | None = None, conf: float | None = None):
    global _n
    _n += 1
    iid = f"item_{_n}"
    ITEMS[iid] = {"id": iid, "label": label, "confidence": conf or round(random.uniform(0.75, 0.97), 2),
                  "since": time.time() - (age if age is not None else random.randint(600, 5 * 86400)), "manual_override": False}


for l in START:
    _add(l, conf=0.62 if l == "parsley" else None)


def _view(it: dict) -> dict:
    return {"id": it["id"], "label": it["label"], "confidence": it["confidence"],
            "time_in_fridge_seconds": int(time.time() - it["since"]), "manual_override": it["manual_override"]}


@app.get("/inventory")
def get_inventory(force_refresh: bool = False):
    if force_refresh:
        time.sleep(0.4)  # pretend to grab a frame + run inference
    return {"last_updated_epoch": int(time.time()), "items": [_view(i) for i in ITEMS.values()]}


@app.patch("/inventory")
async def patch_inventory(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "Body must be JSON."}, 400)
    if not isinstance(body, list) or not body:
        return JSONResponse({"success": False, "error": "Body must be a non-empty JSON array."}, 400)
    for u in body:  # validate everything first so a batch is all-or-nothing
        if not isinstance(u, dict) or "id" not in u:
            return JSONResponse({"success": False, "error": "Every item needs an 'id'."}, 400)
        if u["id"] not in ITEMS:
            return JSONResponse({"success": False, "error": f"Item id '{u['id']}' not found in tracking inventory."}, 404)
        for k, v in u.items():
            if k != "id" and (k not in FIELDS or not isinstance(v, FIELDS[k]) or (k == "confidence" and not 0 <= v <= 1)):
                return JSONResponse({"success": False, "error": f"Invalid field or value: '{k}'."}, 400)
    out = []
    for u in body:
        it = ITEMS[u["id"]]
        for k, v in u.items():
            if k == "time_in_fridge_seconds":
                it["since"] = time.time() - v
            elif k != "id":
                it[k] = v
        out.append(_view(it))
    return {"success": True, "updated_items": out}


@app.put("/_mock/items")
async def mock_set(request: Request):
    ITEMS.clear()
    for l in await request.json():
        _add(str(l))
    return {"ok": True, "count": len(ITEMS)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("EDGE_PORT", "9000")), log_level="warning")
