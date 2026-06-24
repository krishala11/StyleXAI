"""STEP 4 — Outfit compatibility graph from curated outfits."""
import json
from collections import defaultdict

import pandas as pd

import config
from src.logging_utils import save_checkpoint

SLOT_COLUMNS = [
    "hero_id", "second_id", "layer_id", "footwear_id",
    "accessory_1_id", "accessory_2_id",
]


def build_outfit_graph(outfits_df: pd.DataFrame) -> dict:
    """
    For each outfit, connect all non-null item IDs pairwise (undirected).
    Edge weight default = 1.0 (stylist-curated).
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    edge_count = 0
    category_patterns: dict[str, int] = defaultdict(int)

    for _, outfit in outfits_df.iterrows():
        items = []
        for col in SLOT_COLUMNS:
            val = outfit.get(col)
            if pd.notna(val) and str(val).strip():
                items.append(str(val).strip())

        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                graph[a][b] = config.GRAPH_DIRECT_WEIGHT
                graph[b][a] = config.GRAPH_DIRECT_WEIGHT
                edge_count += 1
                category_patterns[f"{a}->{b}"] = category_patterns.get(f"{a}->{b}", 0) + 1

    nodes = len(graph)
    max_edges = nodes * (nodes - 1) / 2 if nodes > 1 else 1
    density = edge_count / max_edges if max_edges else 0

    # Adjacency list serializable form
    adjacency = {k: list(v.keys()) for k, v in graph.items()}

    log = {
        "nodes": nodes,
        "directed_edge_pairs": edge_count,
        "edge_density": round(density, 4),
        "adjacency_list_size": sum(len(v) for v in graph.values()),
        "category_level_note": "Edges span hero-second-footwear-accessory slots per curated outfit",
        "sample_neighbors": {k: adjacency[k][:5] for k in list(adjacency.keys())[:3]},
    }
    save_checkpoint("checkpoint_3_graph", log)

    artifact = {
        "graph": {k: dict(v) for k, v in graph.items()},
        "adjacency": adjacency,
        "log": log,
    }
    with open(config.ARTIFACTS_DIR / "outfit_graph.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[GRAPH] {nodes} nodes, {edge_count} edge pairs, density={density:.4f}")
    return artifact
