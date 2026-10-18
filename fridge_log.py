"""Tiny shared logger so the three servers print one clean, easy-to-follow line per event.

    15:44:14  MCP    > suggest_recipes {}
    15:44:14  EDGE   GET /inventory -> 3 items (42 ms)

Tags: SIM (Alexa+ simulator), MCP (tool server), EDGE (friend's camera API), AGENT (dish agent).
LOG_LEVEL in .env:  quiet = only problems | normal (default) = one line per step | debug = also full arguments/results.
Secrets are never logged: only tool names and short summaries.
"""
import json, logging, os, sys, time

LEVEL = os.environ.get("LOG_LEVEL", "normal").lower()
if LEVEL != "debug":  # hide the libraries' own chatter ("HTTP Request: ...", "Processing request ...") so only our lines show
    os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
    for _n in ("httpx", "httpcore", "mcp", "anthropic", "openai", "uvicorn.access"):
        logging.getLogger(_n).setLevel(logging.WARNING)
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if _USE_COLOR and os.name == "nt":
    os.system("")  # switches on ANSI colors in Windows terminals

_TAG_COLOR = {"SIM": "36", "MCP": "35", "EDGE": "33", "AGENT": "34"}   # cyan, magenta, yellow, blue
_KIND_COLOR = {"ok": "32", "fail": "31", "warn": "33"}                  # green, red, yellow
_ICON = {"info": " ", "ok": "+", "fail": "x", "warn": "!"}


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def short(obj, n: int = 90) -> str:
    """Compact one-line preview of any value."""
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def log(tag: str, msg: str, kind: str = "info") -> None:
    """kind: info | ok | fail | warn. In quiet mode only fail/warn are shown."""
    if LEVEL == "quiet" and kind in ("info", "ok"):
        return
    t = time.strftime("%H:%M:%S")
    line = f"{_c('90', t)}  {_c(_TAG_COLOR.get(tag, '0'), f'{tag:<5}')} {_c(_KIND_COLOR.get(kind, '0'), _ICON[kind])} {msg}"
    print(line, flush=True)


def debug(tag: str, msg: str) -> None:
    if LEVEL == "debug":
        print(f"{_c('90', time.strftime('%H:%M:%S'))}  {_c(_TAG_COLOR.get(tag, '0'), f'{tag:<5}')}   {_c('90', msg)}", flush=True)
