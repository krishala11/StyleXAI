"""STEP 2 — Multimodal embedding extraction using FashionCLIP (torch + transformers)."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config
from src.logging_utils import save_checkpoint

EMBEDDING_DIM = config.EMBEDDING_DIM


def build_product_text(row: pd.Series) -> str:
    tags = str(row.get("tags", "")).replace(";", " ")
    parts = [
        str(row.get("name", "")),
        tags,
        str(row.get("description", "")),
        str(row.get("brand", "")),
        str(row.get("category_label", "")),
        str(row.get("occasion", "")),
    ]
    return " ".join(p for p in parts if p and p != "nan")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


class FashionCLIPEncoder:
    """FashionCLIP image + text encoder with CLIP fallback."""

    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name = config.EMBEDDING_MODEL
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=False)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            fallback = "openai/clip-vit-base-patch32"
            print(f"[EMBEDDINGS] FashionCLIP load failed ({e}); using {fallback}")
            self.model_name = fallback
            self.processor = CLIPProcessor.from_pretrained(fallback, use_fast=False)
            self.model = CLIPModel.from_pretrained(fallback).to(self.device)
        self.model.eval()
        self.normalization = "L2 unit vectors (cosine via dot product)"
        if self.device == "cuda":
            self.model = self.model.half()

    @property
    def loaded(self) -> str:
        return self.model_name

    @torch.inference_mode()
    def encode_image(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        out = self.model.get_image_features(**inputs)
        if not isinstance(out, torch.Tensor):
            out = getattr(out, "image_embeds", getattr(out, "pooler_output", out[0]))
        emb = out.float().cpu().numpy().squeeze()
        return _normalize(emb.astype(np.float32))

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text.lower().strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        if not isinstance(out, torch.Tensor):
            out = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
        emb = out.float().cpu().numpy().squeeze()
        return _normalize(emb.astype(np.float32))


def extract_embeddings(products_df: pd.DataFrame) -> dict:
    encoder = FashionCLIPEncoder()
    image_embeddings, text_embeddings, failures = {}, {}, []

    for i, (_, row) in enumerate(products_df.iterrows()):
        pid = str(row["id"])
        img_path = config.DATA_DIR / row["image"]
        text = build_product_text(row)

        try:
            if img_path.exists():
                image_embeddings[pid] = encoder.encode_image(img_path)
            else:
                failures.append({"id": pid, "type": "image_missing"})
                image_embeddings[pid] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        except Exception as e:
            print(f"Image encode error for {pid}: {e}")
            failures.append({"id": pid, "type": "image_encode", "error": str(e)})
            image_embeddings[pid] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        try:
            text_embeddings[pid] = encoder.encode_text(text)
        except Exception as e:
            print(f"Text encode error for {pid}: {e}")
            failures.append({"id": pid, "type": "text_encode", "error": str(e)})
            text_embeddings[pid] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        if i % 10 == 0:
            gc.collect()

    log = {
        "model_used": encoder.loaded,
        "normalization": encoder.normalization,
        "dimensionality": EMBEDDING_DIM,
        "text_preprocessing": "lowercase; tags semicolon→space; no stopword removal",
        "products_encoded": len(products_df),
        "failures": failures,
        "sample_text": build_product_text(products_df.iloc[0]),
    }
    save_checkpoint("checkpoint_2_embeddings", log)

    np.savez(
        config.ARTIFACTS_DIR / "embeddings.npz",
        product_ids=np.array(list(image_embeddings.keys())),
        image=np.stack([image_embeddings[k] for k in image_embeddings]),
        text=np.stack([text_embeddings[k] for k in text_embeddings]),
    )
    with open(config.ARTIFACTS_DIR / "embedding_meta.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"[EMBEDDINGS] Model={encoder.loaded}, encoded {len(products_df)} products, failures={len(failures)}")
    return {"image": image_embeddings, "text": text_embeddings, "log": log, "encoder": encoder}
