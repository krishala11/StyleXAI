"""STEP 5 — Graph expansion with FAISS proxy fallback."""
import config
from src.logging_utils import append_trace


def expand_neighbors(kb, item_id: str, trace: list) -> list[dict]:
    """
    Return compatible candidates from graph.
    Direct edges weight 1.0; inferred via FAISS proxy weight 0.7.
    """
    direct = []
    inferred = []
    fallback_count = 0

    neighbors = kb.graph.get(item_id, {})
    if neighbors:
        for nid, w in neighbors.items():
            direct.append({"id": nid, "graph_score": w, "edge_type": "direct"})
    else:
        fallback_count += 1
        nearest = kb.faiss_nearest(item_id, k=1)
        if nearest:
            proxy_id, _ = nearest[0]
            proxy_neighbors = kb.graph.get(proxy_id, {})
            for nid in proxy_neighbors:
                inferred.append({
                    "id": nid,
                    "graph_score": config.GRAPH_INFERRED_WEIGHT,
                    "edge_type": "inferred",
                    "proxy": proxy_id,
                })

    results = direct + inferred
    append_trace("graph_expansion", {
        "item_id": item_id,
        "direct_count": len(direct),
        "inferred_count": len(inferred),
        "fallback_events": fallback_count,
    }, trace)
    return results
