"""STEP 3 — Dominant color extraction via OpenCV + sklearn KMeans."""
import json

import cv2
import numpy as np
import pandas as pd

import config
from src.logging_utils import save_checkpoint

COLOR_NAMES = {
    (255, 255, 255): "white",
    (0, 0, 0): "black",
    (128, 128, 128): "grey",
    (0, 0, 128): "navy",
    (0, 0, 255): "blue",
    (139, 69, 19): "brown",
    (245, 245, 220): "beige",
    (255, 0, 0): "red",
    (0, 128, 0): "green",
    (128, 0, 0): "maroon",
    (255, 255, 0): "yellow",
    (128, 0, 128): "purple",
    (255, 192, 203): "pink",
    (255, 215, 0): "gold",
    (85, 107, 47): "olive",
    (255, 228, 196): "cream",
}


def _rgb_to_name(rgb: tuple[int, int, int]) -> str:
    best, best_dist = "unknown", float("inf")
    for ref, name in COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if dist < best_dist:
            best_dist, best = dist, name
    return best


def extract_colors(image_path, n_clusters: int = 3) -> dict:
    """
    K=3 clusters: primary garment + accent + background.
    Center-crop reduces background noise before KMeans.
    """
    from sklearn.cluster import KMeans

    img = cv2.imread(str(image_path))
    if img is None:
        return {"primary": "unknown", "secondary": "unknown", "error": "read_failed"}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.1)
    cropped = img[margin : h - margin, margin : w - margin]
    pixels = cropped.reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        return {"primary": "unknown", "secondary": "unknown", "error": "too_few_pixels"}

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels, minlength=n_clusters)
    order = np.argsort(-counts)

    colors = []
    for idx in order:
        centroid = kmeans.cluster_centers_[idx].astype(int)
        rgb = tuple(int(x) for x in np.clip(centroid, 0, 255))
        colors.append({"name": _rgb_to_name(rgb), "rgb": list(rgb), "count": int(counts[idx])})

    filtered = [c for c in colors if not (c["name"] == "white" and c["count"] > 0.5 * len(pixels))]
    if not filtered:
        filtered = colors

    primary = filtered[0]["name"]
    secondary = filtered[1]["name"] if len(filtered) > 1 else primary
    return {"primary": primary, "secondary": secondary, "clusters": colors}


def extract_all_colors(products_df: pd.DataFrame) -> dict:
    n_clusters = 3
    color_map, failures = {}, []

    for _, row in products_df.iterrows():
        pid = str(row["id"])
        path = config.DATA_DIR / row["image"]
        if not path.exists():
            color_map[pid] = {"primary": "unknown", "secondary": "unknown"}
            failures.append(pid)
            continue
        try:
            color_map[pid] = extract_colors(path, n_clusters=n_clusters)
        except Exception as e:
            color_map[pid] = {"primary": "unknown", "secondary": "unknown", "error": str(e)}
            failures.append(pid)

    log = {
        "method": "OpenCV + sklearn KMeans",
        "n_clusters": n_clusters,
        "justification": "K=3 separates garment, accent, and background; center-crop reduces edge noise",
        "failure_count": len(failures),
        "failure_ids": failures,
    }
    save_checkpoint("checkpoint_3_colors", log)

    with open(config.ARTIFACTS_DIR / "colors.json", "w") as f:
        json.dump(color_map, f, indent=2)

    print(f"[COLORS] Extracted for {len(color_map)} products, failures={len(failures)}")
    return color_map
