"""Fridge MCP server (Streamable HTTP) -- the thing Alexa+ connects to.

Run:  python -m mcp_server.server        -> http://127.0.0.1:8899/mcp
Tools return structured data PLUS a short `speech` string, because Alexa+ speaks the answer.
Alexa+ wants round trips < 500 ms, so every tool is local-first; the one network tool has a hard timeout.
"""
import asyncio, functools, hashlib, inspect, json, os, time
from datetime import datetime
from typing import Any

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from fridge_log import log, debug, short
from . import edge, recipes, store

HOST = os.environ.get("MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_PORT", "8899"))
API_KEY = os.environ.get("MCP_API_KEY", "")            # Alexa+ -> MCP (bearer from account linking)
INGEST_KEY = os.environ.get("INGEST_API_KEY", "")      # classifier -> MCP /api/inventory
AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8897")
AGENT_TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "6"))
_cache: dict[str, dict] = {}                           # (inventory, prefs) -> agent answer; prewarmed on every ingest

mcp = FastMCP(
    "smart-fridge",
    instructions=("Live view of the user's refrigerator, kept up to date by a camera running on-device AI. "
                  "Use it when the user asks what food they have, what is about to go bad, what to cook, "
                  "or wants to manage a shopping list."),
    host=HOST, port=PORT,
    # Behind a tunnel (cloudflared/ngrok) the Host header is not localhost; auth is handled by the bearer middleware.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def tool():
    """Register an MCP tool AND log each call: name+args when it starts, timing/result when it ends."""
    def wrap(fn):
        name = fn.__name__

        def _done(t0, res):
            ms = (time.perf_counter() - t0) * 1000
            if isinstance(res, dict) and res.get("error"):
                log("MCP", f"{name} failed: {short(res['error'])} ({ms:.0f} ms)", "fail")
            else:
                log("MCP", f"{name} done ({ms:.0f} ms)", "ok")
            debug("MCP", f"{name} result: {short(res, 400)}")

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def inner(*a, **k):
                log("MCP", f"{name} {short(k or a)}")
                t0 = time.perf_counter()
                try:
                    res = await fn(*a, **k)
                except Exception as e:
                    log("MCP", f"{name} crashed: {type(e).__name__}: {e}", "fail"); raise
                _done(t0, res); return res
        else:
            @functools.wraps(fn)
            def inner(*a, **k):
                log("MCP", f"{name} {short(k or a)}")
                t0 = time.perf_counter()
                try:
                    res = fn(*a, **k)
                except Exception as e:
                    log("MCP", f"{name} crashed: {type(e).__name__}: {e}", "fail"); raise
                _done(t0, res); return res
        return mcp.tool()(inner)
    return wrap


def _names(state) -> set[str]:
    return {i["item"] for i in state["items"]}


def _stale_note(state) -> str:
    if state.get("edge_error"):
        return " (note: I can't reach the fridge camera right now, so this may be out of date)"
    a = state.get("age_seconds")
    return " (note: the fridge camera hasn't updated recently)" if a is not None and a > 300 else ""


@tool()
def get_fridge_inventory(force_refresh: bool = False) -> dict[str, Any]:
    """List everything currently detected in the fridge. `items` is grouped by name (qty, confidence, estimated days
    left); `tracked` has one row per physical object with the `id` needed by correct_fridge_items. Low confidence
    (< 0.7) means the camera is unsure -- say so and offer to correct it. Set force_refresh=true only if the user says
    they just changed something; it makes the camera re-scan (slower)."""
    s = store.load_state(force_refresh)
    if s.get("error"):
        return {"items": [], "speech": "I can't see inside the fridge right now.", "error": s["error"]}
    parts = []
    for i in s["items"]:
        unsure = i.get("confidence") is not None and i["confidence"] < 0.7 and not i.get("manual_override")
        parts.append((f"{i['qty']} {i['item']}" if i["qty"] > 1 else i["item"]) + (" (not sure)" if unsure else ""))
    speech = ("Your fridge has " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "."
              if parts else "Your fridge looks empty.") + _stale_note(s)
    out = {"last_updated": s["last_updated"], "count": len(s["items"]), "items": s["items"], "speech": speech}
    if s.get("tracked"):
        out["tracked"] = s["tracked"]
    return out


@tool()
def correct_fridge_items(corrections: list[dict[str, Any]]) -> dict[str, Any]:
    """Fix what the camera got wrong, in one batch. Each correction needs `id` (from get_fridge_inventory.tracked) OR
    `from_label` (the wrong name the user said, e.g. 'parsley', which fixes every matching object), plus any of: `label`
    (new name), `confidence`, `time_in_fridge_seconds` (e.g. 0 = 'I just bought it'), `manual_override` (default true when
    label changes, so the camera stops flipping it back). Example: [{"from_label":"parsley","label":"cilantro"}]."""
    if not edge.ENABLED:
        return {"success": False, "speech": "Corrections need the fridge camera service, which isn't connected."}
    tracked = store.load_state(True).get("tracked", [])
    updates, unresolved = [], []
    for c in corrections:
        c = dict(c)
        ids = [c["id"]] if c.get("id") else [t["id"] for t in tracked if t["item"] == store.norm(str(c.get("from_label", "")))]
        if not ids:
            unresolved.append(c.get("id") or c.get("from_label"))
            continue
        fields = {k: c[k] for k in ("label", "confidence", "time_in_fridge_seconds", "manual_override") if k in c}
        if "label" in fields:
            fields.setdefault("confidence", 1.0)
            fields.setdefault("manual_override", True)
        updates += [{"id": i, **fields} for i in ids]
    if not updates:
        return {"success": False, "unresolved": unresolved, "speech": "I couldn't find that item in the fridge."}
    try:
        code, body = edge.patch(updates)
    except Exception as e:
        return {"success": False, "error": str(e), "speech": "I couldn't reach the fridge camera to save that correction."}
    if code != 200 or not body.get("success"):
        err = body.get("error") or (str(body.get("detail"))[:300] if body.get("detail") else f"HTTP {code}")  # 422 = FastAPI validation
        return {"success": False, "error": err, "speech": "The fridge camera rejected that correction."}
    changed = [f"{u['label']}" for u in updates if "label" in u]
    return {"success": True, "updated_items": body.get("updated_items", []), "unresolved": unresolved,
            "speech": ("Got it, I've updated " + ", ".join(sorted(set(changed))) + ".") if changed else "Done, I've updated the fridge."}


@tool()
def get_system_status() -> dict[str, Any]:
    """Check whether the fridge camera/classifier is online and how fresh its data is. Call this when an inventory
    request fails or the user asks if the fridge is working."""
    if not edge.ENABLED:
        s = store.load_state()
        return {"mode": "push", "online": s.get("age_seconds") is not None and s["age_seconds"] < 600,
                "last_updated": s.get("last_updated"), "age_seconds": s.get("age_seconds"),
                "speech": "The fridge camera is reporting in." if s.get("age_seconds") is not None and s["age_seconds"] < 600 else "I haven't heard from the fridge camera recently."}
    try:
        edge.fetch()
        s = store.load_state()
        ms = edge.last_latency_ms()
        stale = s.get("age_seconds") is not None and s["age_seconds"] > 300
        return {"mode": "pull", "online": True, "last_updated": s.get("last_updated"), "age_seconds": s.get("age_seconds"),
                "edge_latency_ms": round(ms) if ms else None, "stale": stale,
                "speech": "The fridge camera is online" + (", but its last update was a while ago." if stale else " and up to date.")}
    except Exception as e:
        return {"mode": "pull", "online": False, "error": f"{type(e).__name__}",
                "speech": "The fridge camera seems to be offline right now."}


@tool()
def check_item(item: str) -> dict[str, Any]:
    """Check whether a specific food item (e.g. 'milk', 'paneer', 'curd') is in the fridge and how fresh it is."""
    s = store.load_state()
    n = store.norm(item)
    hit = next((i for i in s["items"] if i["item"] == n), None)
    if not hit:
        return {"item": n, "present": False, "speech": f"I don't see any {n} in your fridge."}
    left = hit["est_days_left"]
    fresh = "" if left is None else (f" It should stay fresh for about {left} more days." if left > 0 else " It may be past its best.")
    return {"item": n, "present": True, "qty": hit["qty"], "est_days_left": left,
            "speech": f"Yes, you have {hit['qty']} {n}.{fresh}"}


@tool()
def get_expiring_items(within_days: int = 3) -> dict[str, Any]:
    """Find items that will likely spoil within the next N days (default 3). Expiry is estimated from how
    long each item has been in the fridge. Use for 'what's going bad?' or 'what should I use up?'."""
    s = store.load_state()
    soon = sorted((i for i in s["items"] if i["est_days_left"] is not None and i["est_days_left"] <= within_days),
                  key=lambda i: i["est_days_left"])
    if not soon:
        return {"items": [], "speech": f"Nothing in your fridge is expected to spoil in the next {within_days} days."}
    desc = [f"{i['item']} ({'expired' if i['est_days_left'] < 0 else str(i['est_days_left']) + (' day left' if i['est_days_left'] == 1 else ' days left')})" for i in soon]
    return {"within_days": within_days, "items": soon, "speech": "Use these up soon: " + ", ".join(desc) + "."}


def _key(state, prefs: str, n: int) -> str:
    sig = json.dumps([(i["item"], i["qty"], i["est_days_left"]) for i in sorted(state["items"], key=lambda i: i["item"])]) + f"|{prefs}|{n}"
    return hashlib.sha1(sig.encode()).hexdigest()


async def _ask_agent(path: str, payload: dict, timeout: float = AGENT_TIMEOUT) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(AGENT_URL + path, json=payload)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


async def _dishes(state, prefs: str, n: int) -> dict:
    """Cache -> dish AI agent -> local rule-based fallback. Never fails: the voice assistant must always answer."""
    k = _key(state, prefs, n)
    if k in _cache:
        debug("MCP", "dishes: served from cache")
        return {**_cache[k], "cached": True}
    t0 = time.perf_counter()
    res = await _ask_agent("/suggest", {"items": state["items"], "preferences": prefs, "n": n})
    if not res or not res.get("dishes"):
        from dish_agent import brain  # offline rules; agent service is down or empty
        log("MCP", "dish agent unavailable or empty -> using built-in rules", "warn")
        res = {"dishes": brain.rules(state["items"], n), "source": "rules-fallback"}
    else:
        log("MCP", f"dish agent answered: {len(res['dishes'])} dishes, source={res.get('source')} ({(time.perf_counter() - t0) * 1000:.0f} ms)", "ok")
    _cache[k] = res
    if len(_cache) > 50:
        _cache.pop(next(iter(_cache)))
    return {**res, "cached": False}


@tool()
async def suggest_recipes(max_results: int = 3, preferences: str = "") -> dict[str, Any]:
    """Ask the dish-planning AI agent what the user can cook right now from the fridge, prioritising ingredients
    about to expire. `preferences` is free text such as 'vegetarian', 'quick', 'spicy', 'no eggs'."""
    s = store.load_state()
    if not s["items"]:
        return {"dishes": [], "speech": "Your fridge looks empty, so I can't suggest anything yet."}
    n = max(1, min(max_results, 5))
    res = await _dishes(s, preferences.strip()[:100], n)
    dishes = res["dishes"]
    if not dishes:
        return {"dishes": [], "source": res.get("source"), "speech": "I couldn't find a good match with what's in the fridge."}
    top = dishes[0]
    names = [d["name"] for d in dishes]
    speech = f"You could make {', '.join(names[:-1]) + ' or ' if len(names) > 1 else ''}{names[-1]}. I'd start with {top['name']}"
    speech += f", about {top['time_min']} minutes." if top.get("time_min") else "."
    if top.get("why"):
        speech += " " + top["why"]
    if top.get("missing"):
        speech += f" You're missing {', '.join(top['missing'])}."
    return {"dishes": dishes, "source": res.get("source"), "cached": res.get("cached"), "speech": speech + _stale_note(s)}


@tool()
async def get_recipe_steps(dish: str) -> dict[str, Any]:
    """Get step-by-step cooking instructions for a dish (e.g. 'Egg Bhurji') and which ingredients are missing."""
    s = store.load_state()
    d = next((d for v in _cache.values() for d in v.get("dishes", []) if d["name"].lower() == dish.lower().strip()), None)
    if not d:
        d = await _ask_agent("/recipe", {"dish": dish, "items": s["items"]})
        if d is None:
            from dish_agent import brain
            d = await brain.recipe(dish, s["items"])
    if not d.get("steps"):
        return {"found": False, "speech": f"I don't have a recipe for {dish}."}
    speech = f"{d['name']}" + (f" takes about {d['time_min']} minutes. " if d.get("time_min") else ". ") + " ".join(f"Step {n}: {t}" for n, t in enumerate(d["steps"], 1))
    if d.get("missing"):
        speech += f" You're missing {', '.join(d['missing'])}."
    return {"found": True, "name": d["name"], "time_min": d.get("time_min"), "missing": d.get("missing", []), "steps": d["steps"], "speech": speech}


@tool()
async def search_online_recipes(ingredient: str) -> dict[str, Any]:
    """Search TheMealDB (public recipe API) for dishes featuring one ingredient. Use for variety when the built-in
    suggestions aren't enough. Falls back gracefully if offline."""
    try:
        async with httpx.AsyncClient(timeout=2.5) as c:
            r = await c.get("https://www.themealdb.com/api/json/v1/1/filter.php", params={"i": ingredient})
            meals = (r.json().get("meals") or [])[:5]
    except Exception:
        return {"meals": [], "speech": "I couldn't reach the online recipe service just now."}
    if not meals:
        return {"meals": [], "speech": f"I didn't find online recipes with {ingredient}."}
    names = [m["strMeal"] for m in meals]
    return {"meals": [{"name": m["strMeal"], "thumb": m["strMealThumb"]} for m in meals],
            "speech": f"Online, people cook with {ingredient}: " + ", ".join(names) + "."}


@tool()
def add_to_shopping_list(item: str, qty: int = 1) -> dict[str, Any]:
    """Add an item to the user's shopping list (e.g. when something is missing for a recipe)."""
    lst = store.load_shopping()
    n = store.norm(item)
    for e in lst:
        if e["item"] == n:
            e["qty"] += qty
            break
    else:
        lst.append({"item": n, "qty": qty, "added": datetime.now(store.IST).isoformat(timespec="seconds")})
    store.save_shopping(lst)
    return {"shopping_list": lst, "speech": f"Added {n} to your shopping list."}


@tool()
def get_shopping_list() -> dict[str, Any]:
    """Read back the user's current shopping list."""
    lst = store.load_shopping()
    speech = ("Your shopping list has " + ", ".join(f"{e['qty']} {e['item']}" for e in lst) + ".") if lst else "Your shopping list is empty."
    return {"shopping_list": lst, "speech": speech}


@tool()
def remove_from_shopping_list(item: str) -> dict[str, Any]:
    """Remove an item from the shopping list."""
    n = store.norm(item)
    lst = [e for e in store.load_shopping() if e["item"] != n]
    store.save_shopping(lst)
    return {"shopping_list": lst, "speech": f"Removed {n} from your shopping list."}


@mcp.resource("fridge://inventory")
def inventory_resource() -> str:
    """Raw JSON snapshot of the fridge (resource form of get_fridge_inventory)."""
    import json
    return json.dumps(store.load_state())


@mcp.custom_route("/api/inventory", methods=["POST"])
async def ingest(request: Request):
    """Classifier -> MCP. Body: {"items": ["milk", ...]} or {"items": [{"item","qty","confidence"}, ...]} (also accepts a bare list)."""
    try:
        body = await request.json()
        entries = body if isinstance(body, list) else body.get("items", body.get("detections"))
        if not isinstance(entries, list):
            raise ValueError("expected a list under 'items'")
        out = store.ingest(entries, source=(body.get("source") if isinstance(body, dict) else None) or "classifier")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    asyncio.create_task(_dishes(store.load_state(), "", 3))  # prewarm so the next Alexa+ call is instant
    return JSONResponse({"ok": True, **out})


@mcp.custom_route("/api/inventory", methods=["GET"])
async def read_inventory(_: Request):
    return JSONResponse(store.load_state())


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request):
    s = store.load_state()
    return JSONResponse({"ok": True, "mode": "pull" if edge.ENABLED else "push", "items": len(s["items"]), "state_age_seconds": s.get("age_seconds")})


class BearerAuth(BaseHTTPMiddleware):
    """/mcp (Alexa+): Bearer MCP_API_KEY -- after account linking Alexa+ sends it on every request.
    Unauthenticated -> 401 *without* WWW-Authenticate (matches Amazon's documented behaviour).
    /api/* (classifier): Bearer INGEST_API_KEY."""
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        key = INGEST_KEY if path.startswith("/api/") else API_KEY
        if path != "/health" and key and request.headers.get("authorization") != f"Bearer {key}":
            return PlainTextResponse("Unauthorized", status_code=401)
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Response-Ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return resp


async def _poll_edge():
    """Pull mode: keep the dish-agent cache warm so Alexa+'s suggest_recipes call is instant (LLMs are slower than 500 ms)."""
    interval = float(os.environ.get("EDGE_POLL_SECONDS", "10"))
    while True:
        try:
            st = await asyncio.to_thread(store.load_state)
            if st["items"] and not st.get("edge_error"):
                await _dishes(st, "", 3)
        except Exception:
            pass
        await asyncio.sleep(interval)


def build_app():
    from contextlib import asynccontextmanager
    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuth)
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(a):
        task = asyncio.create_task(_poll_edge()) if edge.ENABLED else None
        try:
            async with inner(a) as state:
                yield state
        finally:
            if task:
                task.cancel()
    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    log("MCP", f"smart-fridge MCP on http://{HOST}:{PORT}/mcp  (auth: {'bearer' if API_KEY else 'OFF'}, camera API: {edge.EDGE_URL or 'none (push mode)'})")
    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")
