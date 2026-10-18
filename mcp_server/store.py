"""Reads the fridge state written by the edge layer and manages the shopping list."""
import json, os, re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from . import edge

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = Path(os.environ.get("FRIDGE_STATE_PATH", ROOT / "shared/fridge_state.json"))
SHOP_PATH = Path(os.environ.get("SHOPPING_LIST_PATH", ROOT / "shared/shopping_list.json"))
IST = timezone(timedelta(hours=5, minutes=30))

# Camera cannot see expiry dates, so estimate from first_seen + typical shelf life (days).
SHELF_LIFE = {"milk": 7, "eggs": 21, "chicken": 2, "spinach": 5, "tomato": 7, "apple": 21, "cheese": 21,
              "bread": 5, "yogurt": 10, "butter": 60, "carrot": 21, "onion": 30, "potato": 30, "paneer": 5,
              "banana": 5, "lettuce": 5, "orange": 14, "capsicum": 7, "cucumber": 7, "juice": 7}
DEFAULT_SHELF = 7
SYNONYMS = {"egg": "eggs", "curd": "yogurt", "dahi": "yogurt", "tomatoes": "tomato", "onions": "onion",
            "potatoes": "potato", "bell pepper": "capsicum", "pepper": "capsicum", "carrots": "carrot",
            "apples": "apple", "bananas": "banana", "cucumbers": "cucumber", "chicken breast": "chicken"}


def norm(name: str) -> str:
    n = re.sub(r"[^a-z ]", "", re.sub(r"[_\-]+", " ", name.lower())).strip()
    n = re.sub(r"\s+", " ", n)
    return SYNONYMS.get(n, n)


def _parse(ts: str | None):
    try:
        d = datetime.fromisoformat(ts)
        return d if d.tzinfo else d.replace(tzinfo=IST)
    except Exception:
        return None


def _load_file_state() -> dict:
    """Return {"last_updated", "age_seconds", "items": [...]} with days_left computed. Never raises."""
    try:
        raw = json.loads(STATE_PATH.read_text())
    except Exception as e:
        return {"last_updated": None, "age_seconds": None, "items": [], "error": f"fridge state unavailable: {e}"}
    now = datetime.now(IST)
    items = []
    for it in raw.get("inventory", []):
        name = norm(str(it.get("item", "")))
        if not name:
            continue
        seen = _parse(it.get("first_seen"))
        days_in = (now - seen).days if seen else None
        left = (SHELF_LIFE.get(name, DEFAULT_SHELF) - days_in) if days_in is not None else None
        items.append({"item": name, "qty": int(it.get("qty", 1)), "confidence": it.get("confidence"),
                      "days_in_fridge": days_in, "est_days_left": left})
    upd = _parse(raw.get("last_updated"))
    return {"last_updated": raw.get("last_updated"),
            "age_seconds": int((now - upd).total_seconds()) if upd else None, "items": items}


def load_shopping() -> list[dict]:
    try:
        return json.loads(SHOP_PATH.read_text())
    except Exception:
        return []


def save_shopping(lst: list[dict]) -> None:
    SHOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHOP_PATH.write_text(json.dumps(lst, indent=2))


def ingest(entries: list, source: str = "classifier", min_conf: float = 0.5) -> dict:
    """Replace the fridge snapshot with what the classifier currently sees.

    entries: ["milk", ...] or [{"item": "milk", "qty": 2, "confidence": 0.9}, ...]
    Repeated labels are summed (a detector that returns one box per object works). first_seen is preserved for items
    that were already present, so freshness estimates survive across pushes. Smoothing/debouncing is the classifier's job.
    """
    try:
        old = {i["item"]: i for i in json.loads(STATE_PATH.read_text()).get("inventory", [])}
    except Exception:
        old = {}
    now = datetime.now(IST).isoformat(timespec="seconds")
    agg: dict[str, dict] = {}
    dropped = []
    for e in entries:
        d = {"item": e} if isinstance(e, str) else dict(e)
        name = norm(str(d.get("item") or d.get("label") or d.get("class") or ""))
        conf = d.get("confidence", d.get("score"))
        if not name or (conf is not None and float(conf) < min_conf):
            dropped.append(name or "?")
            continue
        qty = max(1, int(d.get("qty", d.get("count", 1))))
        a = agg.setdefault(name, {"item": name, "qty": 0, "confidence": 0.0})
        a["qty"] += qty
        a["confidence"] = round(max(a["confidence"], float(conf) if conf is not None else 1.0), 3)
    inv = []
    for name, a in agg.items():
        a["first_seen"] = old.get(name, {}).get("first_seen", now)
        inv.append(a)
    tmp = STATE_PATH.with_suffix(".tmp")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps({"last_updated": now, "source": source, "inventory": inv}, indent=2))
    os.replace(tmp, STATE_PATH)  # atomic: readers never see a half-written file
    return {"accepted": len(inv), "dropped_low_confidence": dropped, "items": [i["item"] for i in inv]}


def _from_edge(raw: dict) -> dict:
    """Edge payload -> internal state. `tracked` = one row per physical object (has ids); `items` = aggregated by label."""
    now = datetime.now(IST)
    tracked, agg = [], {}
    for it in raw.get("items", []):
        name = norm(str(it.get("label", "")))
        if not name:
            continue
        secs = max(0, int(it.get("time_in_fridge_seconds") or 0))
        days = secs // 86400
        left = SHELF_LIFE.get(name, DEFAULT_SHELF) - days
        conf = it.get("confidence")
        tracked.append({"id": it.get("id"), "item": name, "confidence": conf, "time_in_fridge_seconds": secs,
                        "manual_override": bool(it.get("manual_override")), "est_days_left": left})
        a = agg.setdefault(name, {"item": name, "qty": 0, "confidence": 1.0, "days_in_fridge": 0, "est_days_left": left,
                                  "ids": [], "manual_override": False})
        a["qty"] += 1
        a["confidence"] = min(a["confidence"], conf if conf is not None else 1.0)
        a["days_in_fridge"] = max(a["days_in_fridge"], days)
        a["est_days_left"] = min(a["est_days_left"], left)
        a["ids"].append(it.get("id"))
        a["manual_override"] = a["manual_override"] or bool(it.get("manual_override"))
    ep = raw.get("last_updated_epoch")
    upd = datetime.fromtimestamp(ep, IST) if ep else None
    return {"last_updated": upd.isoformat(timespec="seconds") if upd else None,
            "age_seconds": int((now - upd).total_seconds()) if upd else None,
            "items": list(agg.values()), "tracked": tracked, "source": "edge"}


def load_state(force_refresh: bool = False) -> dict:
    """Pull mode (EDGE_URL set): ask the classifier's API; on failure serve the last good snapshot flagged stale.
    Push/file mode: read shared/fridge_state.json written by POST /api/inventory. Never raises."""
    if not edge.ENABLED:
        return _load_file_state()
    try:
        return _from_edge(edge.fetch(force_refresh))
    except Exception as e:
        good = edge.last_good()
        if good is not None:
            st = _from_edge(good)
            st["edge_error"] = f"edge unreachable ({type(e).__name__}); showing last known snapshot"
            return st
        return {"last_updated": None, "age_seconds": None, "items": [], "tracked": [], "source": "edge",
                "error": f"fridge camera unreachable: {type(e).__name__}"}
