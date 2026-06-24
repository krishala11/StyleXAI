"""STEP 8-10 — Preference scoring, outfit-level ranking, top-3 selection."""
import config
from src.logging_utils import append_trace, save_checkpoint
from src.online.slot_fill import one_piece_label


def preference_score(kb, outfit_path: dict, profile: dict, intent: dict) -> dict:
    style = (profile.get("style") or intent.get("style") or "").lower()
    age = profile.get("age", 25)
    scores = []
    weights_applied = []

    item_ids = [outfit_path.get(k) for k in ("topwear", "bottomwear", "footwear", "accessory") if outfit_path.get(k)]

    for pid in item_ids:
        p = kb.products[pid]
        s = 0.5

        if style == "formal" and p.get("occasion") == "office":
            s += 0.25
            weights_applied.append("formal→office boost")
        if style == "formal" and p.get("category") in config.FORMAL_CATEGORIES:
            s += 0.15

        if age and age < 30 and p.get("occasion") in ("casual", "party"):
            s += 0.05
            weights_applied.append("20s→trendy allowance")

        # Neutral palette preference
        colors = kb.colors.get(pid, {})
        if colors.get("primary") in {"white", "grey", "navy", "black", "beige"}:
            s += 0.08

        scores.append(min(s, 1.0))

    final = sum(scores) / len(scores) if scores else 0.5
    return {"preference_score": round(final, 4), "weights_applied": list(set(weights_applied))}


def outfit_level_scores(kb, path: dict, ranked: dict, profile: dict, intent: dict) -> dict:
    pairs = path.get("pairs", [])
    compatibility_score = sum(p["pair_score"] for p in pairs) / len(pairs) if pairs else 0.0

    pref = preference_score(kb, path, profile, intent)

    # Average retrieval of outfit items
    retrieval_map = {}
    for slot_items in ranked.values():
        for r in slot_items:
            retrieval_map[r["id"]] = r["retrieval_score"]

    item_ids = [path.get(k) for k in ("topwear", "bottomwear", "footwear", "accessory") if path.get(k)]
    retrieval_score = sum(retrieval_map.get(i, 0) for i in item_ids) / len(item_ids) if item_ids else 0

    occasion = intent.get("occasion", "")
    occasion_hits = sum(1 for i in item_ids if kb.products[i].get("occasion") == occasion)
    occasion_match = occasion_hits / len(item_ids) if item_ids else 0

    outfit_score = (
        config.OUTFIT_COMPAT_WEIGHT * compatibility_score
        + config.OUTFIT_PREFERENCE_WEIGHT * pref["preference_score"]
        + config.OUTFIT_RETRIEVAL_WEIGHT * retrieval_score
        + config.OUTFIT_OCCASION_WEIGHT * occasion_match
    )

    return {
        "compatibility_score": round(compatibility_score, 4),
        "preference_score": pref["preference_score"],
        "preference_weights": pref["weights_applied"],
        "retrieval_score": round(retrieval_score, 4),
        "occasion_match": round(occasion_match, 4),
        "outfit_score": round(outfit_score, 4),
        "pair_breakdown": pairs,
    }


def rank_outfits(kb, paths: list, ranked: dict, profile: dict, intent: dict, trace: list) -> list[dict]:
    outfits = []
    for path in paths:
        scores = outfit_level_scores(kb, path, ranked, profile, intent)
        items = {}
        for slot in ("topwear", "bottomwear", "footwear", "accessory"):
            pid = path.get(slot)
            if pid:
                items[slot] = {
                    "id": pid,
                    "name": kb.products[pid]["name"],
                    "image": kb.products[pid]["image"],
                    "brand": kb.products[pid].get("brand"),
                    "category": kb.products[pid].get("category_label"),
                }
        outfits.append({
            "items": items,
            "scores": scores,
            "path": path,
            "display_meta": path.get("display_meta", {
                "one_piece": kb.is_one_piece(path.get("topwear", "")) if path.get("topwear") else False,
                "bottomwear_note": one_piece_label(kb, path["topwear"]) if path.get("topwear") else None,
            }),
        })

    outfits.sort(key=lambda x: -x["scores"]["outfit_score"])
    top = outfits[: config.TOP_K_OUTFITS]

    ranking_table = [
        {
            "rank": i + 1,
            "outfit_score": o["scores"]["outfit_score"],
            "top": o["items"].get("topwear", {}).get("name"),
            "score_diff_from_first": round(top[0]["scores"]["outfit_score"] - o["scores"]["outfit_score"], 4) if top else 0,
        }
        for i, o in enumerate(top)
    ]
    append_trace("outfit_ranking", {"ranking_table": ranking_table}, trace)
    save_checkpoint("checkpoint_5_scoring", {"outfits": [{k: v for k, v in o.items() if k != "path"} for o in top]})
    return top
