"""Load offline artifacts for online recommendation."""
import json
import numpy as np
import pandas as pd

import config
from src.online.encoder import EmbeddingEncoder

class KnowledgeBase:
    def __init__(self):
        self.products_df = pd.read_json(config.ARTIFACTS_DIR / "products.json")
        self.products = {row["id"]: row for _, row in self.products_df.iterrows()}

        with open(config.ARTIFACTS_DIR / "colors.json") as f:
            self.colors = json.load(f)

        with open(config.ARTIFACTS_DIR / "outfit_graph.json") as f:
            graph_data = json.load(f)
            self.graph = graph_data["graph"]
            self.adjacency = graph_data["adjacency"]

        with open(config.ARTIFACTS_DIR / "product_id_map.json") as f:
            self.product_ids = json.load(f)

        emb = np.load(config.ARTIFACTS_DIR / "embeddings.npz")
        self.image_embeddings = {pid: emb["image"][i] for i, pid in enumerate(emb["product_ids"])}
        self.text_embeddings = {pid: emb["text"][i] for i, pid in enumerate(emb["product_ids"])}

        self.encoder = EmbeddingEncoder(self.products_df)

    def get_slot(self, product_id: str) -> str:
        cat = self.products[product_id]["category"]
        if cat in config.TOPWEAR_CATEGORIES:
            return "topwear"
        if cat in config.BOTTOMWEAR_CATEGORIES:
            return "bottomwear"
        if cat in config.FOOTWEAR_CATEGORIES:
            return "footwear"
        if cat in config.ACCESSORY_CATEGORIES:
            return "accessory"
        if cat in config.LAYER_CATEGORIES:
            return "layer"
        return "other"

    def is_one_piece(self, product_id: str) -> bool:
        cat = self.products[product_id]["category"]
        return cat in {
            "party-dresses", "casual-dresses", "maxi-dresses", "dresses",
            "wedding-sarees", "co-ord-sets", "kurta-sets", "sharara-sets",
            "salwar-suits", "sherwanis", "suits",
        }

    def faiss_nearest(self, product_id: str, k: int = 3) -> list:
        if product_id not in self.product_ids:
            return []
        query = self.image_embeddings[product_id]
        
        # We completely removed FAISS to fix the MacOS Apple Silicon bug!
        # Since we only have a small dataset, numpy is instantly fast anyway.
        scores = []
        for pid, emb in self.image_embeddings.items():
            if pid != product_id:
                score = float(np.dot(query, emb))
                scores.append((pid, score))
                
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
