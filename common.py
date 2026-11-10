"""Shared label maps for the context-relevance smoke test.

Dishes are Food-101 class names. Dishes with strongly shared origins
(dumplings, falafel, hummus, churros, ...) are left out on purpose, so each
dish has one clear "home" country.
"""

DISH_COUNTRY = {
    "sushi": "japan", "ramen": "japan", "takoyaki": "japan",
    "pho": "vietnam",
    "pad_thai": "thailand",
    "bibimbap": "korea",
    "peking_duck": "china", "hot_and_sour_soup": "china",
    "tacos": "mexico", "huevos_rancheros": "mexico",
    "paella": "spain",
    "pizza": "italy", "spaghetti_carbonara": "italy",
    "french_onion_soup": "france", "escargots": "france",
    "samosa": "india", "chicken_curry": "india",
}

COUNTRIES = sorted(set(DISH_COUNTRY.values()))

COUNTRY_NAME = {
    "japan": "Japan", "vietnam": "Vietnam", "thailand": "Thailand",
    "korea": "South Korea", "china": "China", "mexico": "Mexico",
    "spain": "Spain", "italy": "Italy", "france": "France", "india": "India",
}

LETTERS = "ABCD"


def pretty(dish: str) -> str:
    return dish.replace("_", " ")
