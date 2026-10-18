"""Client for the classifier teammate's edge API (pull mode).

Contract (agreed in the classifier chat):
  GET   {EDGE_URL}/inventory?force_refresh=true|false
        -> {"last_updated_epoch": int, "items": [{"id","label","confidence","time_in_fridge_seconds","manual_override"}]}
  PATCH {EDGE_URL}/inventory   body: [{"id": "...", "label"?, "confidence"?, "time_in_fridge_seconds"?, "manual_override"?}]
        -> {"success": true, "updated_items": [...]}  |  400/404 {"success": false, "error": "..."}
"""
import os, time
import httpx
from fridge_log import log, debug

EDGE_URL = os.environ.get("EDGE_URL", "").rstrip("/")
EDGE_KEY = os.environ.get("EDGE_API_KEY", "")
TIMEOUT = float(os.environ.get("EDGE_TIMEOUT", "2.0"))
TTL = float(os.environ.get("EDGE_CACHE_TTL", "2.0"))     # several tools in one Alexa turn share one fetch
ENABLED = bool(EDGE_URL)

_last: dict | None = None
_last_at = 0.0
_last_ms: float | None = None
_online: bool | None = None   # None = unknown yet; used to log only state changes


def _headers():
    return {"Authorization": f"Bearer {EDGE_KEY}"} if EDGE_KEY else None


def fetch(force_refresh: bool = False) -> dict:
    """Return the edge payload. Raises on failure. force_refresh bypasses our TTL cache AND asks the edge to re-run inference."""
    global _last, _last_at, _last_ms, _online
    if not force_refresh and _last is not None and time.monotonic() - _last_at < TTL:
        return _last
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"{EDGE_URL}/inventory", params={"force_refresh": str(force_refresh).lower()}, headers=_headers(),
                      timeout=max(TIMEOUT, 8.0) if force_refresh else TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        if _online is not False:  # print only when the state changes, so a dead camera doesn't spam the log
            log("EDGE", f"GET {EDGE_URL}/inventory failed: {type(e).__name__} -- camera API unreachable", "fail")
        _online = False
        raise
    _last_ms = (time.perf_counter() - t0) * 1000
    _last, _last_at = r.json(), time.monotonic()
    n = len(_last.get("items", []))
    if _online is not True:
        log("EDGE", f"camera API online: {n} items ({_last_ms:.0f} ms)", "ok")
    _online = True
    debug("EDGE", f"GET /inventory -> {n} items ({_last_ms:.0f} ms)")
    return _last


def last_good() -> dict | None:
    return _last


def last_latency_ms() -> float | None:
    return _last_ms


def patch(updates: list[dict]) -> tuple[int, dict]:
    global _last
    r = httpx.patch(f"{EDGE_URL}/inventory", json=updates, headers=_headers(), timeout=TIMEOUT + 2)
    _last = None  # next read must see the correction
    log("EDGE", f"PATCH /inventory {len(updates)} update(s) -> HTTP {r.status_code}", "ok" if r.status_code == 200 else "fail")
    try:
        body = r.json()
    except Exception:
        body = {"success": False, "error": r.text[:200]}
    return r.status_code, body
