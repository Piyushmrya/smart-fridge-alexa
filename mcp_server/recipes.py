"""Built-in offline recipe book + ranking. Online lookup (TheMealDB) lives in server.py."""
RECIPES = [
    {"name": "Masala Omelette", "required": ["eggs", "onion", "tomato"], "optional": ["capsicum", "cheese"], "time_min": 10,
     "steps": ["Chop onion, tomato and capsicum.", "Whisk eggs with salt, pepper and chilli.", "Saute veg 2 min, pour eggs, cook until set, fold."]},
    {"name": "Egg Bhurji", "required": ["eggs", "onion"], "optional": ["tomato", "capsicum", "butter"], "time_min": 12,
     "steps": ["Saute onion (and tomato) in butter/oil with turmeric and chilli.", "Crack in eggs and scramble on medium heat.", "Finish with coriander; serve with bread."]},
    {"name": "Paneer Bhurji", "required": ["paneer", "onion", "tomato"], "optional": ["capsicum"], "time_min": 15,
     "steps": ["Saute onion, tomato and capsicum with spices.", "Crumble paneer in and cook 4 min.", "Garnish and serve with roti or bread."]},
    {"name": "Palak Paneer", "required": ["spinach", "paneer", "onion", "tomato"], "optional": ["yogurt"], "time_min": 30,
     "steps": ["Blanch spinach and blend to a puree.", "Saute onion and tomato masala.", "Add puree and paneer cubes, simmer 8 min, finish with a little cream or yogurt."]},
    {"name": "Spinach Cheese Sandwich", "required": ["bread", "spinach", "cheese"], "optional": ["tomato", "butter"], "time_min": 10,
     "steps": ["Butter the bread.", "Layer spinach, cheese and tomato.", "Toast until golden and the cheese melts."]},
    {"name": "Grilled Cheese Sandwich", "required": ["bread", "cheese"], "optional": ["butter", "tomato"], "time_min": 8,
     "steps": ["Butter the outside of the bread.", "Add cheese between slices.", "Grill both sides until crisp."]},
    {"name": "French Toast", "required": ["bread", "eggs", "milk"], "optional": ["banana"], "time_min": 12,
     "steps": ["Whisk eggs, milk and a pinch of sugar.", "Dip bread slices.", "Pan-fry both sides until golden; top with banana."]},
    {"name": "Banana Smoothie", "required": ["banana", "milk"], "optional": ["yogurt"], "time_min": 5,
     "steps": ["Blend banana, cold milk and yogurt.", "Sweeten if needed and serve chilled."]},
    {"name": "Aloo Sabzi", "required": ["potato", "onion"], "optional": ["tomato"], "time_min": 25,
     "steps": ["Boil and cube the potatoes.", "Saute onion (and tomato) with cumin and turmeric.", "Toss in potatoes and cook 5 min."]},
    {"name": "Cucumber Raita", "required": ["yogurt", "cucumber"], "optional": ["onion", "tomato"], "time_min": 5,
     "steps": ["Whisk yogurt with salt.", "Fold in grated cucumber and roasted cumin."]},
    {"name": "Chicken Stir Fry", "required": ["chicken", "onion", "capsicum"], "optional": ["carrot"], "time_min": 20,
     "steps": ["Slice chicken and veg.", "Sear chicken on high heat 5 min.", "Add veg and soy/spices, toss 4 min."]},
    {"name": "Carrot Apple Salad", "required": ["carrot", "apple"], "optional": ["yogurt"], "time_min": 7,
     "steps": ["Grate carrot and apple.", "Toss with lemon, honey and a spoon of yogurt."]},
    {"name": "Egg Cheese Sandwich", "required": ["bread", "eggs", "cheese"], "optional": ["tomato", "butter"], "time_min": 12,
     "steps": ["Make a thin omelette.", "Layer with cheese in toasted bread.", "Grill until melted."]},
    {"name": "Tomato Soup", "required": ["tomato", "onion"], "optional": ["carrot", "butter"], "time_min": 25,
     "steps": ["Saute onion and carrot in butter.", "Add chopped tomato and water, simmer 15 min.", "Blend, season and serve."]},
]


def rank_recipes(have: set[str], expiring: set[str], limit: int = 3):
    """Return (cookable, near_miss). Cookable = all required in fridge. Bonus for using expiring items."""
    cookable, near = [], []
    for r in RECIPES:
        miss = [i for i in r["required"] if i not in have]
        used_opt = [i for i in r["optional"] if i in have]
        uses_expiring = [i for i in r["required"] + r["optional"] if i in expiring and i in have]
        entry = {"name": r["name"], "time_min": r["time_min"], "uses_expiring": uses_expiring,
                 "optional_available": used_opt, "missing": miss}
        score = len(uses_expiring) * 3 + len(used_opt)
        if not miss:
            cookable.append((score, entry))
        elif len(miss) == 1 and len(r["required"]) > 1:
            near.append((score, entry))
    key = lambda t: (-t[0], t[1]["time_min"])
    return [e for _, e in sorted(cookable, key=key)[:limit]], [e for _, e in sorted(near, key=key)[:limit]]


def find(name: str):
    n = name.lower().strip()
    for r in RECIPES:
        if r["name"].lower() == n or n in r["name"].lower():
            return r
    return None
