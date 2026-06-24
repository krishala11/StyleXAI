# AI Fashion Outfit Recommendation System

End-to-end dual-mode fashion recommendation engine for the Dare XAI internship assessment.

## Stack

| Component | Library |
|-----------|---------|
| Embeddings | **FashionCLIP** (torch + transformers) |
| Colors | **OpenCV** + **sklearn** KMeans |
| Vector search | **FAISS** IndexFlatIP |
| LLM (intent + explanations) | **Groq** free tier (`llama-3.3-70b-versatile`) |
| UI | **Streamlit** |

## Quick Start

```bash
cd "Fashion Recommender"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Free Groq API key: https://console.groq.com/keys
cp .env.example .env
# Edit .env → set GROQ_API_KEY=gsk_...

# Build offline index (FashionCLIP embeddings, graph, FAISS)
python run_offline.py

# CLI demo
python run_demo.py "I need an outfit for a business meeting."

# Streamlit UI (Mode 1 Chat + Mode 2 Guided Builder)
streamlit run app.py
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram.

## Scoring (traceable)

**Retrieval:** `0.7 × image_sim + 0.3 × text_sim`

**Pair:** `0.4 × graph + 0.3 × metadata + 0.3 × color` (reject if < 0.5)

**Outfit:** `0.50 × compatibility + 0.20 × preference + 0.20 × retrieval + 0.10 × occasion`

## Checkpoints

All pipeline stages log to `checkpoints/`:

- `checkpoint_1_dataset_analysis.json`
- `checkpoint_2_embeddings.json`
- `checkpoint_3_graph.json` / `checkpoint_3_colors.json` / `checkpoint_3_index.json`
- `checkpoint_4_retrieval.json`
- `checkpoint_5_scoring.json`
- `checkpoint_6_final_output.json`

## Data

Dataset in `ML-TASK/` (68 products, 25 curated outfits, images/).

## Design Decisions

1. **FashionCLIP** — multimodal fashion-specific embeddings (falls back to `openai/clip-vit-base-patch32` if model load fails).
2. **FAISS IndexFlatIP** — N=68; exact cosine search is sufficient.
3. **K=3 color clusters** — OpenCV read + sklearn KMeans; center-crop reduces background noise.
4. **Groq API** — free-tier LLM for intent parsing and explanations; rule-based/template fallback if key missing.
5. **Graph** — pairwise edges from curated outfits; weight 1.0 direct, 0.7 inferred via FAISS proxy.
