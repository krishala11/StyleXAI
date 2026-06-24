"""STEP 6 — Pair compatibility scoring (graph + metadata + color)."""
import config
from src.logging_utils import append_trace

# Color harmony rules (explainable fashion heuristics)
NEUTRALS = {"white", "black", "grey", "navy", "beige", "cream", "tan", "brown", "olive", "silver"}
CLASH_PAIRS = {
    frozenset({"red", "green"}),
    frozenset({"orange", "pink"}),
    frozenset({"yellow", "purple"}),
}


def _graph_score(kb, a: str, b: str) -> tuple[float, str]:
    if b in kb.graph.get(a, {}):
        return kb.graph[a][b], "direct"
    # Check inferred via proxy
    nearest = kb.faiss_nearest(a, k=1)
    if nearest:
        proxy, _ = nearest[0]
        if b in kb.graph.get(proxy, {}):
            return config.GRAPH_INFERRED_WEIGHT, "inferred"
    return 0.0, "none"


def _metadata_score(kb, a: str, b: str, intent: dict) -> float:
    pa, pb = kb.products[a], kb.products[b]
    constraints = []
    matched = 0

    gender = intent.get("gender", "men")
    if _gender_ok(pa, gender):
        matched += 1
    constraints.append("gender")

    occasion = intent.get("occasion")
    if occasion and (pa.get("occasion") == occasion or pb.get("occasion") == occasion):
        matched += 1
    constraints.append("occasion")

    if pa.get("gender") == pb.get("gender") or "unisex" in (pa.get("gender"), pb.get("gender")):
        matched += 1
    constraints.append("gender_compat")

    # Category compatibility: top+bottom, bottom+footwear OK
    slot_a, slot_b = kb.get_slot(a), kb.get_slot(b)
    valid_pairs = {
        ("topwear", "bottomwear"), ("bottomwear", "topwear"),
        ("topwear", "footwear"), ("footwear", "topwear"),
        ("bottomwear", "footwear"), ("footwear", "bottomwear"),
        ("topwear", "layer"), ("layer", "topwear"),
        ("topwear", "accessory"), ("accessory", "topwear"),
    }
    if (slot_a, slot_b) in valid_pairs or slot_a == slot_b:
        matched += 1
    constraints.append("category")

    total = len(constraints)
    return matched / total if total else 0.5


def _gender_ok(product: dict, gender: str) -> bool:
    pg = str(product.get("gender", "")).lower()
    return pg in ("unisex", gender.lower(), gender.replace("male", "men"))


def _color_score(kb, a: str, b: str) -> float:
    ca = kb.colors.get(a, {})
    cb = kb.colors.get(b, {})
    c1 = ca.get("primary", "unknown")
    c2 = cb.get("primary", "unknown")

    if c1 == "unknown" or c2 == "unknown":
        return 0.6

    if c1 == c2:
        return 0.85

    if c1 in NEUTRALS and c2 in NEUTRALS:
        return 0.92

    if (c1 in NEUTRALS) != (c2 in NEUTRALS):
        return 0.78

    if frozenset({c1, c2}) in CLASH_PAIRS:
        return 0.25

    return 0.55


def pair_compatibility(kb, a: str, b: str, intent: dict) -> dict:
    gs, edge_type = _graph_score(kb, a, b)
    ms = _metadata_score(kb, a, b, intent)
    cs = _color_score(kb, a, b)

    score = (
        config.PAIR_GRAPH_WEIGHT * gs
        + config.PAIR_METADATA_WEIGHT * ms
        + config.PAIR_COLOR_WEIGHT * cs
    )
    return {
        "pair_score": round(score, 4),
        "graph_score": round(gs, 4),
        "metadata_score": round(ms, 4),
        "color_score": round(cs, 4),
        "edge_type": edge_type,
        "rejected": score < config.PAIR_REJECT_THRESHOLD,
    }


def score_pairs_batch(kb, pairs: list[tuple[str, str]], intent: dict, trace: list) -> list[dict]:
    results = []
    rejected = 0
    for a, b in pairs:
        r = pair_compatibility(kb, a, b, intent)
        r["a"] = a
        r["b"] = b
        if r["rejected"]:
            rejected += 1
        else:
            results.append(r)

    results.sort(key=lambda x: -x["pair_score"])
    append_trace("pair_compatibility", {
        "total_pairs": len(pairs),
        "rejected_count": rejected,
        "top_pairs": results[:5],
    }, trace)
    return results
