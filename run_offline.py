"""Offline phase orchestrator — runs all index-building steps once at startup."""
import json
import os

# Prevent sklearn/loky + torch thread conflict on macOS (causes Bus error)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")

import config
from src.offline.dataset_analysis import load_datasets
from src.offline.embeddings import extract_embeddings
from src.offline.graph import build_outfit_graph
from src.offline.index_builder import build_faiss_index


def run_offline_pipeline(force: bool = False) -> dict:
    marker = config.ARTIFACTS_DIR / "build_complete.json"
    if marker.exists() and not force:
        print("[OFFLINE] Artifacts exist; loading metadata (use force=True to rebuild)")
        with open(marker) as f:
            return json.load(f)

    print("=" * 60)
    print("OFFLINE PHASE — Building knowledge base")
    print("=" * 60)

    products_df, outfits_df, analysis = load_datasets()

    # Embeddings BEFORE sklearn (colors) — torch+sklearn import order matters on macOS
    emb_result = extract_embeddings(products_df)

    from src.offline.colors import extract_all_colors  # lazy: sklearn after torch encoding

    colors = extract_all_colors(products_df)
    graph = build_outfit_graph(outfits_df)
    product_ids = list(products_df["id"])
    index = build_faiss_index(emb_result, product_ids)

    products_df.to_json(config.ARTIFACTS_DIR / "products.json", orient="records", indent=2)

    summary = {
        "products": len(products_df),
        "outfits": len(outfits_df),
        "embedding_model": emb_result["log"]["model_used"],
        "graph_nodes": graph["log"]["nodes"],
        "index_type": index["log"]["index_type"],
    }
    with open(marker, "w") as f:
        json.dump(summary, f, indent=2)

    print("[OFFLINE] Complete.")
    return summary


if __name__ == "__main__":
    run_offline_pipeline(force=True)
