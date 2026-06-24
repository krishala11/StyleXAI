"""Runtime FashionCLIP encoder for online query encoding."""
import pandas as pd

import config
from src.offline.embeddings import FashionCLIPEncoder


class EmbeddingEncoder:
    def __init__(self, products_df: pd.DataFrame | None = None):
        self._enc = FashionCLIPEncoder()

    def encode_text(self, text: str):
        return self._enc.encode_text(text)

    def encode_image(self, image_path):
        return self._enc.encode_image(image_path)
