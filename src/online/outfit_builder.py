"""STEP 7 — Outfit construction via greedy compatibility expansion."""
import pandas as pd

import config
from src.logging_utils import append_trace
from src.online.compatibility import pair_compatibility
from src.online.expansion import expand_neighbors
from src.online.slot_fill import find_best_accessory, one_piece_label

SLOT_KEYS = ("topwear", "bottomwear", "footwear", "accessory", "layer")


def _curated_paths(kb, intent: dict, pools: dict | None = None, ranked: dict | None = None) -> list[dict]:
    """Seed from outfits.csv when greedy search yields too few paths."""
    outfits_df = pd.read_csv(config.OUTFITS_CSV)
    gender = intent.get("gender", "men")
    occasion = intent.get("occasion", "office")
    paths = []

    id_cols = {
        "topwear": "hero_id",
        "bottomwear": "second_id",
        "layer": "layer_id",
        "footwear": "footwear_id",
        "accessory": "accessory_1_id",
    }
    for _, row in outfits_df.iterrows():
        if str(row.get("gender", "")).lower() != gender:
            continue
        if occasion and str(row.get("occasion", "")).lower() != occasion.lower():
            continue
        path = {"pairs": []}
        slot_ids = {}
        for slot, col in id_cols.items():
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                slot_ids[slot] = str(val).strip()
        if "topwear" not in slot_ids:
            continue
        path["topwear"] = slot_ids.get("topwear")
        path["bottomwear"] = slot_ids.get("bottomwear")
        path["footwear"] = slot_ids.get("footwear")
        path["accessory"] = slot_ids.get("accessory")
        if path["topwear"] and path["footwear"]:
            if path["topwear"] and path.get("bottomwear"):
                path["pairs"].append(pair_compatibility(kb, path["topwear"], path["bottomwear"], intent))
            anchor = path.get("bottomwear") or path["topwear"]
            if anchor and path["footwear"]:
                path["pairs"].append(pair_compatibility(kb, anchor, path["footwear"], intent))
            if not path.get("accessory"):
                acc = find_best_accessory(kb, path["topwear"], pools or {}, ranked or {}, intent)
                if acc:
                    path["accessory"] = acc
            if path.get("topwear") and path.get("accessory"):
                path["pairs"].append(pair_compatibility(kb, path["topwear"], path["accessory"], intent))
            path["display_meta"] = {
                "one_piece": kb.is_one_piece(path["topwear"]),
                "bottomwear_note": one_piece_label(kb, path["topwear"]) if kb.is_one_piece(path["topwear"]) else None,
            }
            paths.append(path)
    return paths


def _best_from_pool(kb, anchor: str, pool_ids: list[str], intent: dict, retrieval_ranked: list) -> str | None:
    if not pool_ids:
        return None

    retrieval_map = {r["id"]: r["retrieval_score"] for r in retrieval_ranked}
    graph_neighbors = {n["id"]: n for n in expand_neighbors(kb, anchor, [])}

    scored = []
    for pid in pool_ids:
        if pid == anchor:
            continue
        pc = pair_compatibility(kb, anchor, pid, intent)
        if pc["rejected"]:
            continue
        bonus = 0.1 if pid in graph_neighbors else 0
        combined = pc["pair_score"] + bonus + 0.05 * retrieval_map.get(pid, 0)
        scored.append((pid, combined, pc))

    if not scored:
        # Fallback: top retrieval in pool
        for r in retrieval_ranked:
            if r["id"] in pool_ids and r["id"] != anchor:
                return r["id"]
        return pool_ids[0] if pool_ids else None

    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


def construct_outfits(kb, pools: dict, ranked: dict, intent: dict, trace: list, max_outfits: int = 10) -> list[dict]:
    """
    Greedy: start from top shirts → expand pants → expand shoes.
    Also handle one-piece outfits (dresses).
    """
    paths = []
    seen = set()
    top_candidates = [r["id"] for r in ranked.get("topwear", [])[:15]]
    bottom_pool = pools.get("bottomwear", [])
    shoe_pool = pools.get("footwear", [])
    accessory_pool = pools.get("accessory", [])

    for top_id in top_candidates:
        path = {"topwear": top_id, "pairs": []}
        one_piece = kb.is_one_piece(top_id)

        if one_piece:
            shoe = _best_from_pool(kb, top_id, shoe_pool, intent, ranked.get("footwear", []))
            path["bottomwear"] = None
            path["footwear"] = shoe
            if shoe:
                path["pairs"].append(pair_compatibility(kb, top_id, shoe, intent))
        else:
            pant = _best_from_pool(kb, top_id, bottom_pool, intent, ranked.get("bottomwear", []))
            path["bottomwear"] = pant
            if pant:
                path["pairs"].append(pair_compatibility(kb, top_id, pant, intent))
                shoe = _best_from_pool(kb, pant, shoe_pool, intent, ranked.get("footwear", []))
                path["footwear"] = shoe
                if shoe:
                    path["pairs"].append(pair_compatibility(kb, pant, shoe, intent))

        acc = _best_from_pool(kb, top_id, accessory_pool, intent, ranked.get("accessory", []))
        if not acc:
            acc = find_best_accessory(kb, top_id, pools, ranked, intent)
        path["accessory"] = acc
        if acc:
            path["pairs"].append(pair_compatibility(kb, top_id, acc, intent))

        path["display_meta"] = {
            "one_piece": one_piece,
            "bottomwear_note": one_piece_label(kb, top_id) if one_piece else None,
        }

        if path.get("footwear") or (one_piece and path.get("footwear")):
            key = tuple(path.get(s) for s in ("topwear", "bottomwear", "footwear"))
            if key not in seen:
                seen.add(key)
                paths.append(path)

        if len(paths) >= max_outfits:
            break

    if len(paths) < config.TOP_K_OUTFITS:
        for cp in _curated_paths(kb, intent, pools, ranked):
            key = tuple(cp.get(s) for s in ("topwear", "bottomwear", "footwear"))
            if key not in seen:
                seen.add(key)
                paths.append(cp)
            if len(paths) >= max_outfits:
                break

    append_trace("outfit_construction", {
        "constructed_paths": len(paths),
        "top_candidates_tried": len(top_candidates),
        "sample_path": paths[0] if paths else None,
    }, trace)
    return paths
