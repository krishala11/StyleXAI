"""STEP 3 — Metadata filtering into category pools."""
import config
from src.logging_utils import append_trace


def _gender_match(product: dict, gender: str) -> bool:
    pg = str(product.get("gender", "")).lower()
    if pg in ("unisex", ""):
        return True
    return pg == gender.lower() or pg == gender.replace("male", "men").replace("female", "women")


def _occasion_match(product: dict, occasion: str) -> bool:
    po = str(product.get("occasion", "")).lower()
    if not occasion:
        return True
    # Neutral occasions in dataset: allow cross-match for office/formal
    if po == occasion.lower():
        return True
    formal_set = {"office", "wedding"}
    if occasion.lower() in formal_set and po in formal_set:
        return True
    if occasion.lower() == "party" and po in ("party", "casual"):
        return True
    return False


def _style_match(product: dict, style: str, intent: dict) -> bool:
    cat = str(product.get("category", "")).lower()
    wear = str(product.get("wear_type", "")).lower()
    style_l = (style or "").lower()

    if style_l == "formal":
        return cat in config.FORMAL_CATEGORIES or product.get("occasion") == "office"
    if style_l == "ethnic":
        return wear == "ethnic"
    if style_l == "smart-casual":
        return wear == "western" and product.get("occasion") in ("casual", "office")
    if style_l == "western":
        return wear == "western"
    return True


def filter_candidates(kb, intent: dict, trace: list) -> dict:
    gender = intent.get("gender", "men")
    occasion = intent.get("occasion", "casual")
    style = intent.get("style", "casual")

    pools = {"topwear": [], "bottomwear": [], "footwear": [], "accessory": [], "layer": []}
    before = len(kb.products)

    for pid, product in kb.products.items():
        if not _gender_match(product, gender):
            continue
        if not _occasion_match(product, occasion):
            continue
        if not _style_match(product, style, intent):
            continue

        slot = kb.get_slot(pid)
        if slot in pools:
            pools[slot].append(pid)

    after = sum(len(v) for v in pools.values())
    log = {
        "before_total": before,
        "after_total": after,
        "pools": {k: len(v) for k, v in pools.items()},
        "filters": {"gender": gender, "occasion": occasion, "style": style},
    }
    append_trace("metadata_filtering", log, trace)

    if after == 0:
        # Relax style filter
        for pid, product in kb.products.items():
            if not _gender_match(product, gender):
                continue
            if not _occasion_match(product, occasion):
                continue
            slot = kb.get_slot(pid)
            if slot in pools and pid not in pools[slot]:
                pools[slot].append(pid)
        append_trace("filter_relaxation", {"reason": "empty_pools", "new_counts": {k: len(v) for k, v in pools.items()}}, trace)

    return pools
