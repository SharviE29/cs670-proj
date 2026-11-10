"""Food classes and the class -> region mapping.

OPEN ITEM: this list is a PROPOSAL for review. Lines marked `# CHECK:` are mappings with
contested or shared origins; decide whether to keep, remap or replace them.

Rules used for the proposal:
  * 50 Food-101 class names (exact spelling), 40 seen + 10 held-out.
  * Every held-out class's region also has >= 1 seen class, so region-associated distractors
    and background matches exist on both sides of the seen/held-out split.
  * Every region has >= 1 class; regions are the background labels for q_S.
"""

from __future__ import annotations

# Region keys are the labels used everywhere (backgrounds, q_S answers).
REGION_NAME = {
    "japan": "Japan", "korea": "South Korea", "china": "China", "vietnam": "Vietnam",
    "thailand": "Thailand", "india": "India", "mexico": "Mexico", "peru": "Peru",
    "spain": "Spain", "italy": "Italy", "france": "France", "greece": "Greece",
    "lebanon": "Lebanon", "usa": "United States", "uk": "United Kingdom", "canada": "Canada",
}

CLASS_REGION = {
    # --- Japan
    "sushi": "japan",
    "ramen": "japan",
    "sashimi": "japan",
    "gyoza": "japan",               # CHECK: Japanese adaptation of Chinese jiaozi; Food-101 photos look Japanese
    "miso_soup": "japan",
    "takoyaki": "japan",
    # --- Korea
    "bibimbap": "korea",
    # --- China
    "peking_duck": "china",
    "hot_and_sour_soup": "china",
    "dumplings": "china",           # CHECK: generic; Food-101 class mixes jiaozi, momo, xiaolongbao
    # --- Vietnam
    "pho": "vietnam",
    "spring_rolls": "vietnam",      # CHECK: Food-101 mixes Vietnamese fresh rolls and Chinese fried rolls
    # --- Thailand
    "pad_thai": "thailand",
    # --- India
    "chicken_curry": "india",       # CHECK: also Thai/Japanese curries in Food-101 photos
    "samosa": "india",
    # --- Mexico
    "tacos": "mexico",
    "huevos_rancheros": "mexico",
    "guacamole": "mexico",
    "chicken_quesadilla": "mexico",
    "nachos": "mexico",             # CHECK: arguably Tex-Mex (USA)
    # --- Peru
    "ceviche": "peru",              # CHECK: also strongly Mexican/Ecuadorian; Peru is the usual origin
    # --- Spain
    "paella": "spain",
    "churros": "spain",             # CHECK: also strongly Mexican
    # --- Italy
    "pizza": "italy",
    "spaghetti_carbonara": "italy",
    "lasagna": "italy",
    "risotto": "italy",
    "tiramisu": "italy",
    "cannoli": "italy",
    "bruschetta": "italy",
    "caprese_salad": "italy",
    # --- France
    "french_onion_soup": "france",
    "escargots": "france",
    "creme_brulee": "france",
    "macarons": "france",
    "croque_madame": "france",
    "foie_gras": "france",
    # --- Greece
    "greek_salad": "greece",
    "baklava": "greece",            # CHECK: equally Turkish/Levantine
    # --- Lebanon
    "hummus": "lebanon",            # CHECK: pan-Levantine; region label could be "Middle East"
    "falafel": "lebanon",           # CHECK: pan-Levantine / Egyptian
    # --- USA
    "hamburger": "usa",
    "hot_dog": "usa",
    "clam_chowder": "usa",
    "lobster_roll_sandwich": "usa",
    "pulled_pork_sandwich": "usa",
    "shrimp_and_grits": "usa",
    "chicken_wings": "usa",         # CHECK: "buffalo wings" are US, but Food-101 photos include Asian wings
    # --- UK
    "fish_and_chips": "uk",
    # --- Canada
    "poutine": "canada",
}

# 10 held-out classes: only in the unseen-class test split.
HELDOUT_CLASSES = [
    "takoyaki",       # japan  (seen: sushi, ramen, ...)
    "dumplings",      # china  (seen: peking_duck, hot_and_sour_soup)
    "spring_rolls",   # vietnam (seen: pho)
    "samosa",         # india  (seen: chicken_curry)
    "guacamole",      # mexico (seen: tacos, ...)
    "cannoli",        # italy
    "macarons",       # france
    "baklava",        # greece (seen: greek_salad)
    "falafel",        # lebanon (seen: hummus)
    "clam_chowder",   # usa
]

CLASSES = sorted(CLASS_REGION)
SEEN_CLASSES = sorted(c for c in CLASSES if c not in HELDOUT_CLASSES)
REGIONS = sorted(REGION_NAME)


def pretty(cls: str) -> str:
    """Display name used in question options."""
    return cls.replace("_", " ")


def region_name(region: str) -> str:
    return REGION_NAME[region]


def classes_in_region(region: str, pool: list[str] | None = None) -> list[str]:
    pool = CLASSES if pool is None else pool
    return sorted(c for c in pool if CLASS_REGION[c] == region)


def validate() -> None:
    assert len(CLASSES) == 50, len(CLASSES)
    assert len(HELDOUT_CLASSES) == 10 and len(set(HELDOUT_CLASSES)) == 10
    assert set(HELDOUT_CLASSES) <= set(CLASSES)
    assert set(CLASS_REGION.values()) == set(REGIONS), "every region needs >= 1 class"
    for c in HELDOUT_CLASSES:
        r = CLASS_REGION[c]
        assert classes_in_region(r, SEEN_CLASSES), f"held-out {c}: region {r} has no seen class"


validate()

if __name__ == "__main__":
    from collections import Counter

    cnt = Counter(CLASS_REGION.values())
    print(f"{len(CLASSES)} classes ({len(SEEN_CLASSES)} seen, {len(HELDOUT_CLASSES)} held-out), "
          f"{len(REGIONS)} regions")
    for r in REGIONS:
        seen = classes_in_region(r, SEEN_CLASSES)
        held = classes_in_region(r, HELDOUT_CLASSES)
        print(f"  {r:9s} {cnt[r]:2d}  seen={seen}  heldout={held}")
