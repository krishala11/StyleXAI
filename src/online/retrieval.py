"""STEP 4 — Hybrid retrieval scoring per category."""
import numpy as np

import config
from src.logging_utils import append_trace


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_intent_text(intent: dict, query: str) -> str:
    parts = [
        intent.get("theme", ""),
        intent.get("style", ""),
        intent.get("occasion", ""),
        intent.get("formality", ""),
        query,
    ]
    return " ".join(p for p in parts if p).lower()


def score_retrieval(kb, pools: dict, intent: dict, query: str, trace: list) -> dict:
    intent_text = build_intent_text(intent, query)
    query_emb = kb.encoder.encode_text(intent_text)

    ranked = {}
    for slot, candidates in pools.items():
        scores = []
        for pid in candidates:
            img_sim = _cosine(query_emb, kb.image_embeddings[pid])
            txt_sim = _cosine(query_emb, kb.text_embeddings[pid])
            retrieval_score = (
                config.RETRIEVAL_IMAGE_WEIGHT * img_sim
                + config.RETRIEVAL_TEXT_WEIGHT * txt_sim
            )
            scores.append({
                "id": pid,
                "name": kb.products[pid]["name"],
                "image": kb.products[pid]["image"],
                "retrieval_score": round(retrieval_score, 4),
                "image_similarity": round(img_sim, 4),
                "text_similarity": round(txt_sim, 4),
            })
        scores.sort(key=lambda x: -x["retrieval_score"])
        ranked[slot] = scores

    log = {
        "intent_text": intent_text,
        "top_per_category": {
            slot: items[:10] for slot, items in ranked.items() if items
        },
        "score_distributions": {
            slot: {
                "min": round(min(i["retrieval_score"] for i in items), 4) if items else 0,
                "max": round(max(i["retrieval_score"] for i in items), 4) if items else 0,
                "mean": round(sum(i["retrieval_score"] for i in items) / len(items), 4) if items else 0,
            }
            for slot, items in ranked.items()
        },
    }
    append_trace("retrieval_scoring", log, trace)
    return ranked
