"""The dish-prediction AI agent's brain.

Input : fridge items (+ estimated days left) and optional preferences ("vegetarian", "quick", ...)
Output: ranked dishes {name, why, time_min, uses_expiring, have, missing, steps}
LLM providers: anthropic | openai (any OpenAI-compatible endpoint) | rules (offline fallback, no key).
The LLM may invent dishes beyond the built-in recipe book -- that is the point of making it an agent.
"""
import json, os, re

from mcp_server import recipes

PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "rules").lower()

SYSTEM = (
    "You are a practical home-cooking assistant for an Indian household kitchen. You get the items currently in the "
    "fridge with estimated days left. Assume common pantry staples are available (oil, salt, spices, flour, rice, sugar). "
    "Suggest dishes that can be cooked NOW, strongly prefer using items that expire soonest, respect the user's preferences, "
    "and keep it realistic (<=45 min). Respond with ONLY a JSON object, no prose:\n"
    '{"dishes":[{"name":str,"why":str (<=20 words),"time_min":int,"uses_expiring":[str],"have":[str],"missing":[str],'
    '"steps":[str,...<=6 short steps]}]}\n'
    "'missing' should list only fridge-type ingredients the user lacks (ideally empty). Use item names exactly as given."
)


def _prompt(items: list[dict], preferences: str, n: int) -> str:
    lines = [f"- {i['item']} x{i.get('qty', 1)}" + (f", ~{i['est_days_left']} days left" if i.get("est_days_left") is not None else "") for i in items]
    return f"Fridge contents:\n" + "\n".join(lines) + f"\n\nPreferences: {preferences or 'none'}\nGive {n} dishes, best first."


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


async def _llm(user: str) -> str:
    if PROVIDER == "anthropic":
        from anthropic import AsyncAnthropic
        r = await AsyncAnthropic().messages.create(model=os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                                                   max_tokens=1200, system=SYSTEM, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import AsyncOpenAI
    c = AsyncOpenAI(base_url=os.environ.get("AGENT_OPENAI_BASE_URL"), api_key=os.environ.get("AGENT_OPENAI_API_KEY"))
    r = await c.chat.completions.create(model=os.environ.get("AGENT_OPENAI_MODEL", "gpt-4o-mini"),
                                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def rules(items: list[dict], n: int) -> list[dict]:
    have = {i["item"] for i in items}
    expiring = {i["item"] for i in items if i.get("est_days_left") is not None and i["est_days_left"] <= 3}
    cook, near = recipes.rank_recipes(have, expiring, n)
    out = []
    for e in cook + near[: max(0, n - len(cook))]:
        r = recipes.find(e["name"])
        out.append({"name": e["name"], "why": "Uses " + ", ".join(e["uses_expiring"]) + " before it spoils." if e["uses_expiring"] else "Made from what you have.",
                    "time_min": e["time_min"], "uses_expiring": e["uses_expiring"], "have": [i for i in r["required"] + r["optional"] if i in have],
                    "missing": e["missing"], "steps": r["steps"]})
    return out


async def suggest(items: list[dict], preferences: str = "", n: int = 3) -> dict:
    n = max(1, min(n, 5))
    if PROVIDER in ("anthropic", "openai"):
        try:
            data = _json(await _llm(_prompt(items, preferences, n)))
            dishes = [d for d in data.get("dishes", []) if d.get("name")][:n]
            if dishes:
                return {"dishes": dishes, "source": f"llm:{PROVIDER}"}
        except Exception as e:  # fall through to rules; the MCP server must always get an answer
            return {"dishes": rules(items, n), "source": "rules", "llm_error": f"{type(e).__name__}: {e}"}
    return {"dishes": rules(items, n), "source": "rules"}


async def recipe(dish: str, items: list[dict]) -> dict:
    r = recipes.find(dish)
    have = {i["item"] for i in items}
    if r:
        return {"name": r["name"], "time_min": r["time_min"], "steps": r["steps"], "missing": [i for i in r["required"] if i not in have], "source": "rules"}
    if PROVIDER in ("anthropic", "openai"):
        try:
            data = _json(await _llm(_prompt(items, f"Give the detailed recipe for: {dish}. Return it as the single dish in dishes.", 1)))
            d = (data.get("dishes") or [None])[0]
            if d:
                return {**d, "source": f"llm:{PROVIDER}"}
        except Exception:
            pass
    return {"name": dish, "steps": [], "missing": [], "source": "none"}
