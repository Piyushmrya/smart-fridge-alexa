"""The dish-prediction AI agent's brain.

Input : fridge items (+ estimated days left) and optional preferences ("vegetarian", "quick", ...)
Output: ranked dishes {name, why, time_min, uses_expiring, have, missing, steps}
LLM providers: bedrock (Strands agent on Amazon Bedrock) | anthropic | openai (any OpenAI-compatible endpoint) | rules (offline, no key).
With bedrock the agent can call tools (recipe book, TheMealDB, shopping list); if Bedrock fails it falls back to the Anthropic API
(if a key is set) and finally to the built-in rules, so the MCP server always gets an answer.
The LLM may invent dishes beyond the built-in recipe book -- that is the point of making it an agent.
"""
import asyncio, json, os, re, time

import bedrock_util, llm_usage
from fridge_log import log, short
from mcp_server import recipes

PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "rules").lower()   # bedrock | anthropic | openai | rules

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


async def _llm(user: str, provider: str | None = None) -> str:
    """One plain LLM call (no tools) on the given provider (default: the configured one)."""
    provider = provider or PROVIDER
    if provider == "bedrock":
        from strands import Agent
        model = bedrock_util.make_model("agent", max_tokens=1500)
        res = await Agent(model=model, system_prompt=SYSTEM, callback_handler=None).invoke_async(user)
        llm_usage.record("AGENT", bedrock_util.model_id(model), *llm_usage.usage_from_result(res), what="bedrock call")
        return str(res)
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        mid = os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-5-5")
        r = await AsyncAnthropic().messages.create(model=mid, max_tokens=1200, system=SYSTEM,
                                                   messages=[{"role": "user", "content": user}])
        llm_usage.record("AGENT", mid, r.usage.input_tokens, r.usage.output_tokens, what="anthropic call")
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import AsyncOpenAI
    c = AsyncOpenAI(base_url=os.environ.get("AGENT_OPENAI_BASE_URL"), api_key=os.environ.get("AGENT_OPENAI_API_KEY"))
    r = await c.chat.completions.create(model=os.environ.get("AGENT_OPENAI_MODEL", "gpt-4o-mini"),
                                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def _chain() -> list[str]:
    """Providers to try in order (see llm_chain.py): the chosen one, then others that have credentials."""
    import llm_chain
    return llm_chain.chain(PROVIDER, agent=True)


AGENT_NOTE = (
    "\n\nYou are an agent with tools. First call rank_builtin_recipes to see what the recipe book can make from the fridge "
    "(and what is about to expire). You may call get_recipe_steps, search_online_recipes (for ideas using an expiring "
    "ingredient) and get_shopping_list at most a couple of times in total -- be quick. Then reply with ONLY the JSON object."
)


def _build_tools(items: list[dict]):
    """Tools the Strands agent may call. They close over this request's fridge contents."""
    from strands import tool
    have = {i["item"] for i in items}
    expiring = {i["item"] for i in items if i.get("est_days_left") is not None and i["est_days_left"] <= 3}

    @tool
    def rank_builtin_recipes(limit: int = 5) -> dict:
        """Rank the built-in recipe book against what is in the fridge right now.

        Args:
            limit: how many recipes to return per group (1-8)

        Returns:
            cook_now: recipes whose required ingredients are all in the fridge (best first);
            almost: recipes missing exactly one ingredient. Each lists time_min, uses_expiring and missing.
        """
        cook, near = recipes.rank_recipes(have, expiring, max(1, min(int(limit), 8)))
        return {"cook_now": cook, "almost": near, "expiring_now": sorted(expiring)}

    @tool
    def get_recipe_steps(name: str) -> dict:
        """Get the ingredients and cooking steps of a built-in recipe.

        Args:
            name: recipe name, e.g. "Egg Bhurji"
        """
        r = recipes.find(name)
        return {"found": False} if not r else {"found": True, "name": r["name"], "required": r["required"],
                                               "optional": r["optional"], "time_min": r["time_min"], "steps": r["steps"]}

    @tool
    def search_online_recipes(ingredient: str) -> dict:
        """Search the public TheMealDB recipe API for dish ideas that use one ingredient.

        Args:
            ingredient: a single ingredient, e.g. "paneer" or "spinach"
        """
        import httpx
        try:
            r = httpx.get("https://www.themealdb.com/api/json/v1/1/filter.php", params={"i": ingredient}, timeout=2.5)
            return {"meals": [m["strMeal"] for m in (r.json().get("meals") or [])[:6]]}
        except Exception:
            return {"meals": [], "note": "online lookup unavailable"}

    @tool
    def get_shopping_list() -> dict:
        """Read the user's shopping list (items they already plan to buy)."""
        from mcp_server import store
        return {"shopping_list": [e["item"] for e in store.load_shopping()]}

    return [rank_builtin_recipes, get_recipe_steps, search_online_recipes, get_shopping_list]


async def _bedrock_suggest(items: list[dict], preferences: str, n: int) -> list[dict]:
    """The Strands agent: plans with tools on Amazon Bedrock, returns dishes (raises on any failure)."""
    from strands import Agent
    model = bedrock_util.make_model("agent", max_tokens=1500)
    agent = Agent(model=model, tools=_build_tools(items), system_prompt=SYSTEM + AGENT_NOTE, callback_handler=None)
    t0 = time.perf_counter()
    res = await asyncio.wait_for(agent.invoke_async(_prompt(items, preferences, n)), timeout=float(os.environ.get("AGENT_RUN_TIMEOUT", "18")))
    used = {k: getattr(v, "call_count", "?") for k, v in getattr(res.metrics, "tool_metrics", {}).items()}
    log("AGENT", f"strands agent finished in {(time.perf_counter() - t0) * 1000:.0f} ms, tools used: {short(used) if used else 'none'}")
    llm_usage.record("AGENT", bedrock_util.model_id(model), *llm_usage.usage_from_result(res), what="strands run")
    dishes = [d for d in _json(str(res)).get("dishes", []) if d.get("name")][:n]
    if not dishes:
        raise ValueError("agent returned no parsable dishes")
    return dishes


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
    errors = []
    for provider in _chain():
        try:
            if provider == "bedrock":
                dishes = await _bedrock_suggest(items, preferences, n)
            else:
                data = _json(await _llm(_prompt(items, preferences, n), provider))
                dishes = [d for d in data.get("dishes", []) if d.get("name")][:n]
            if dishes:
                out = {"dishes": dishes, "source": f"llm:{provider}"}
                if errors:
                    out["llm_error"] = "; ".join(errors) + f" (recovered with {provider})"
                return out
            errors.append(f"{provider}: empty answer")
        except Exception as e:  # try the next provider; the MCP server must always get an answer
            errors.append(f"{provider}: {type(e).__name__}: {str(e)[:160]}")
            log("AGENT", f"{provider} failed -> {errors[-1]}", "warn")
            import llm_chain; llm_chain.mark_down(provider, e)
    out = {"dishes": rules(items, n), "source": "rules"}
    if errors:
        out["llm_error"] = "; ".join(errors)
    return out


async def recipe(dish: str, items: list[dict]) -> dict:
    r = recipes.find(dish)
    have = {i["item"] for i in items}
    if r:
        return {"name": r["name"], "time_min": r["time_min"], "steps": r["steps"], "missing": [i for i in r["required"] if i not in have], "source": "rules"}
    for provider in _chain():
        try:
            data = _json(await _llm(_prompt(items, f"Give the detailed recipe for: {dish}. Return it as the single dish in dishes.", 1), provider))
            d = (data.get("dishes") or [None])[0]
            if d:
                return {**d, "source": f"llm:{provider}"}
        except Exception as e:
            import llm_chain; llm_chain.mark_down(provider, e)
            continue
    return {"name": dish, "steps": [], "missing": [], "source": "none"}
