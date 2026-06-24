"""STEP 5 — FAISS vector index (IndexFlatIP for cosine on L2-normalized vectors)."""
import json

import faiss
import numpy as np

import config
from src.logging_utils import save_checkpoint


def build_faiss_index(embeddings: dict, product_ids: list[str]) -> dict:
    image_matrix = np.stack([embeddings["image"][pid] for pid in product_ids]).astype(np.float32)
    text_matrix = np.stack([embeddings["text"][pid] for pid in product_ids]).astype(np.float32)
    dim = image_matrix.shape[1]

    image_index = faiss.IndexFlatIP(dim)
    text_index = faiss.IndexFlatIP(dim)
    image_index.add(image_matrix)
    text_index.add(text_matrix)

    faiss.write_index(image_index, str(config.ARTIFACTS_DIR / "image_index.faiss"))
    faiss.write_index(text_index, str(config.ARTIFACTS_DIR / "text_index.faiss"))

    with open(config.ARTIFACTS_DIR / "product_id_map.json", "w") as f:
        json.dump(product_ids, f)

    log = {
        "index_type": "IndexFlatIP",
        "reason": "N=68 products — exact cosine search via inner product on unit vectors",
        "dimension": dim,
        "vector_count": len(product_ids),
    }
    save_checkpoint("checkpoint_3_index", log)
    print(f"[FAISS] Built IndexFlatIP for {len(product_ids)} vectors (dim={dim})")
    return {"image_index": image_index, "text_index": text_index, "product_ids": product_ids, "log": log}


def search_index(index, query: np.ndarray, k: int, product_ids: list[str]) -> list[tuple[str, float]]:
    scores, indices = index.search(query.reshape(1, -1).astype(np.float32), k)
    return [
        (product_ids[i], float(scores[0][j]))
        for j, i in enumerate(indices[0])
        if i >= 0
    ]
