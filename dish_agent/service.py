"""Dish agent as its own service.   python -m dish_agent.service   -> http://127.0.0.1:8897
POST /suggest {items:[{item,qty,est_days_left}], preferences?, n?}   POST /recipe {dish, items}"""
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
from fridge_log import log, debug, short
from . import brain

app = FastAPI(title="Dish Agent")


class SuggestIn(BaseModel):
    items: list[dict]
    preferences: str = Field("", max_length=200)
    n: int = 3


class RecipeIn(BaseModel):
    dish: str = Field(max_length=100)
    items: list[dict] = []


@app.get("/health")
def health():
    return {"ok": True, "provider": brain.PROVIDER}


@app.post("/suggest")
async def suggest(b: SuggestIn):
    log("AGENT", f"/suggest {len(b.items)} items, prefs={short(b.preferences or 'none', 40)}, provider={brain.PROVIDER}")
    t0 = time.perf_counter()
    out = await brain.suggest(b.items, b.preferences, b.n)
    ms = (time.perf_counter() - t0) * 1000
    if out.get("llm_error"):
        log("AGENT", f"LLM failed ({short(out['llm_error'], 120)}) -> used rules ({ms:.0f} ms)", "fail")
    else:
        log("AGENT", f"{len(out['dishes'])} dishes from {out.get('source')}: {short([d['name'] for d in out['dishes']], 80)} ({ms:.0f} ms)", "ok")
    debug("AGENT", f"full result: {short(out, 600)}")
    return out


@app.post("/recipe")
async def recipe(b: RecipeIn):
    log("AGENT", f"/recipe {b.dish!r}")
    out = await brain.recipe(b.dish, b.items)
    log("AGENT", f"recipe source={out.get('source')}, {len(out.get('steps', []))} steps", "ok" if out.get("steps") else "warn")
    return out


if __name__ == "__main__":
    log("AGENT", f"dish agent on http://127.0.0.1:{os.environ.get('AGENT_PORT', '8897')} (provider: {brain.PROVIDER}, model: {os.environ.get('AGENT_ANTHROPIC_MODEL', '-')})")
    uvicorn.run(app, host=os.environ.get("AGENT_HOST", "127.0.0.1"), port=int(os.environ.get("AGENT_PORT", "8897")), log_level="warning")
