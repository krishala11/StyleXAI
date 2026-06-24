"""Shared slot-filling helpers for chat + guided modes (uses existing scoring/graph only)."""
import config
from src.online.compatibility import pair_compatibility
from src.online.expansion import expand_neighbors


def _gender_match(kb, pid: str, gender: str) -> bool:
    pg = str(kb.products[pid].get("gender", "")).lower()
    return pg in (gender.lower(), "unisex", "")


def _pool_for_slot(kb, pools: dict, slot: str, intent: dict) -> list[str]:
    """Use filtered pool; if empty, fall back to all catalog items in that slot (gender-scoped)."""
    gender = intent.get("gender", "men")
    pool = list(pools.get(slot, []))
    if not pool:
        pool = [pid for pid in kb.products if kb.get_slot(pid) == slot]
    return [pid for pid in pool if _gender_match(kb, pid, gender)]


def find_best_accessory(kb, top_id: str, pools: dict, ranked: dict, intent: dict) -> str | None:
    """
    Find a compatible accessory for chat-mode outfits.
    Same rules as guided step 4: graph neighbors first, then pair_compatibility.
    """
    if not top_id:
        return None

    accessory_pool = _pool_for_slot(kb, pools, "accessory", intent)
    neighbor_ids = [n["id"] for n in expand_neighbors(kb, top_id, []) if kb.get_slot(n["id"]) == "accessory"]

    ordered = []
    seen = set()
    for pid in neighbor_ids + accessory_pool:
        if pid and pid not in seen and pid != top_id:
            seen.add(pid)
            ordered.append(pid)

    for pid in ordered:
        pc = pair_compatibility(kb, top_id, pid, intent)
        if not pc["rejected"]:
            return pid

    # Soft fallback: graph-linked accessory from curated outfits
    for pid in neighbor_ids:
        return pid

    return None


def one_piece_label(kb, top_id: str) -> str | None:
    """Human-readable note when bottomwear is part of the hero garment."""
    if not top_id or not kb.is_one_piece(top_id):
        return None
    cat = kb.products[top_id].get("category", "")
    labels = {
        "suits": "Trousers are included in this 2-piece suit",
        "sherwanis": "Bottom is included in this sherwani set",
        "party-dresses": "Dress is a complete one-piece — no separate bottom needed",
        "casual-dresses": "Dress is a complete one-piece — no separate bottom needed",
        "maxi-dresses": "Dress is a complete one-piece — no separate bottom needed",
        "wedding-sarees": "Saree is a complete drape — no separate bottom needed",
        "kurta-sets": "Kurta set includes matching bottom",
        "sharara-sets": "Sharara set is a complete co-ordinated look",
        "salwar-suits": "Salwar suit includes matching bottom",
        "co-ord-sets": "Co-ord set is a matching top + bottom set",
    }
    return labels.get(cat, "Complete garment — no separate bottomwear needed")
