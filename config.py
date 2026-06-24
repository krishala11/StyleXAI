"""Central configuration for the Fashion Recommendation System."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "ML-TASK"
PRODUCTS_CSV = DATA_DIR / "products.csv"
OUTFITS_CSV = DATA_DIR / "outfits.csv"
IMAGES_DIR = DATA_DIR / "images"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
ARTIFACTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Embedding model: FashionCLIP via HuggingFace (falls back to CLIP in code if load fails)
EMBEDDING_MODEL = "patrickjohncyh/fashion-clip"
EMBEDDING_DIM = 512

# Scoring weights (from architecture spec)
RETRIEVAL_IMAGE_WEIGHT = 0.7
RETRIEVAL_TEXT_WEIGHT = 0.3
PAIR_GRAPH_WEIGHT = 0.4
PAIR_METADATA_WEIGHT = 0.3
PAIR_COLOR_WEIGHT = 0.3
PAIR_REJECT_THRESHOLD = 0.5
GRAPH_INFERRED_WEIGHT = 0.7
GRAPH_DIRECT_WEIGHT = 1.0

OUTFIT_COMPAT_WEIGHT = 0.50
OUTFIT_PREFERENCE_WEIGHT = 0.20
OUTFIT_RETRIEVAL_WEIGHT = 0.20
OUTFIT_OCCASION_WEIGHT = 0.10

TOP_K_OUTFITS = 5

# Category → outfit slot mapping (justified by dataset category_label values)
TOPWEAR_CATEGORIES = {
    "formal-shirts", "casual-shirts", "party-shirts", "linen-shirts",
    "tshirts", "polo-tshirts", "sweaters", "sweatshirts", "tops",
    "party-dresses", "casual-dresses", "maxi-dresses", "dresses",
    "kurta-sets", "sharara-sets", "salwar-suits", "sherwanis", "suits",
    "co-ord-sets", "activewear", "wedding-sarees",
}
BOTTOMWEAR_CATEGORIES = {
    "trousers", "jeans", "chinos", "shorts", "track-pants", "skirts", "leggings",
}
FOOTWEAR_CATEGORIES = {
    "formal-shoes", "loafers", "running-shoes", "sneakers", "sandals",
    "boots", "heels", "flats", "ethnic-footwear",
}
ACCESSORY_CATEGORIES = {
    "necklaces", "earrings", "clutches", "handbags", "watches", "caps", "sunglasses",
}
LAYER_CATEGORIES = {
    "blazers", "nehru-jackets", "denim-jackets", "long-coats",
}

# Formal style indicators in metadata
FORMAL_CATEGORIES = {
    "formal-shirts", "trousers", "formal-shoes", "blazers", "suits",
}
