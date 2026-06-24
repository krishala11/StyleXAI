"""STEP 1 — Dataset loading and analysis with logged decisions."""
from pathlib import Path

import pandas as pd

import config
from src.logging_utils import save_checkpoint


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    products_df = pd.read_csv(config.PRODUCTS_CSV)
    outfits_df = pd.read_csv(config.OUTFITS_CSV)

    # --- Schema summary ---
    schema = {
        "products_columns": list(products_df.columns),
        "outfits_columns": list(outfits_df.columns),
        "products_count": len(products_df),
        "outfits_count": len(outfits_df),
    }

    # --- Missing values ---
    missing = {
        "products": products_df.isnull().sum().to_dict(),
        "outfits": outfits_df.isnull().sum().to_dict(),
    }

    # --- Duplicate IDs ---
    dup_ids = products_df[products_df.duplicated("id", keep=False)]["id"].tolist()

    # --- Invalid images ---
    invalid_images = []
    for _, row in products_df.iterrows():
        img_path = config.DATA_DIR / row["image"]
        if not img_path.exists():
            invalid_images.append({"id": row["id"], "path": str(row["image"])})

    # --- Category distribution ---
    category_dist = products_df["category"].value_counts().to_dict()
    gender_dist = products_df["gender"].value_counts().to_dict()
    occasion_dist = products_df["occasion"].value_counts().to_dict()

    # --- Color hints from descriptions (for analysis only) ---
    color_keywords = [
        "white", "black", "navy", "grey", "gray", "brown", "beige", "red",
        "blue", "green", "olive", "maroon", "cream", "gold", "purple", "pink",
    ]
    color_mentions = {c: 0 for c in color_keywords}
    for desc in products_df["description"].fillna("").astype(str):
        low = desc.lower()
        for c in color_keywords:
            if c in low:
                color_mentions[c] += 1

    analysis = {
        "schema": schema,
        "missing_values": missing,
        "duplicate_ids": dup_ids,
        "invalid_images": invalid_images,
        "category_distribution": category_dist,
        "gender_distribution": gender_dist,
        "occasion_distribution": occasion_dist,
        "description_color_mentions": color_mentions,
        "preprocessing_decisions": [
            "No row drops: all 68 products retained (invalid image count logged only).",
            "Missing ratings filled at scoring time, not imputed in catalog.",
            "Tags use semicolon separator in source; converted to spaces for text embedding.",
            "category field used for slot assignment (topwear/bottomwear/footwear/accessory).",
        ],
    }

    save_checkpoint("checkpoint_1_dataset_analysis", analysis)
    print(f"[DATASET] Loaded {len(products_df)} products, {len(outfits_df)} outfits")
    print(f"[DATASET] Invalid images: {len(invalid_images)}, Duplicate IDs: {len(dup_ids)}")
    return products_df, outfits_df, analysis
