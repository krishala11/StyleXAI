# StyleXAI Recommendation System - Final Architecture

This document details the final, end-to-end architecture of the StyleXAI Fashion Recommendation System. It serves as an accurate, 1:1 reflection of the current live codebase, documenting the offline vector indexing pipeline, the online multimodal retrieval engine, and the complex graph-based compatibility systems.

---

## 🎯 Goal

**Input:**
```json
{
  "profile": {
    "gender": "male",
    "age": 24,
    "style": "formal"
  },
  "query": "I need an outfit for a business meeting."
}
```

**Output:**
1. White Formal Shirt
2. Grey Trouser
3. Black Oxford Shoes

**Reason:**
Selected because it matches office formality, follows a classic neutral color palette, and is highly compatible based on stylist-curated outfit relationships.

---

## 🏗 SYSTEM OVERVIEW

```mermaid
flowchart TD
    subgraph OFFLINE[OFFLINE PHASE - Run once at startup]
        D1[products.csv + outfits.csv] --> D2[Feature Extraction]
        D2 --> D3[Knowledge Base Creation]
        D3 --> D4[Graph Construction]
        D4 --> D5[Vector Index Creation]
    end

    subgraph ONLINE[ONLINE PHASE - Realtime per user query]
        O1[User Query] --> O2[Intent Extraction]
        O2 --> O3[Metadata Filtering]
        O3 --> O4[Candidate Retrieval]
        O4 --> O5[Graph Expansion]
        O5 --> O6[Pair Compatibility Scoring]
        O6 --> O7[Outfit Construction]
        O7 --> O8[Preference Scoring & Ranking]
        O8 --> O9[LLM Explanation Generation]
        O9 --> O10[Final Top 3 Outfits]
    end
```

---

## 🔧 OFFLINE PHASE (run_offline.py)

### STEP 1 — Load Dataset
- **Input:** `products.csv` (contains categories, styles, occasions) & `outfits.csv` (contains stylist relationships).
- **Process:** Loads products and builds core dictionaries into memory.
- **Output:** `products_dict`, `outfits_dict`.

### STEP 2 — Generate Image Embeddings
- **Process:** Uses `FashionCLIP` Image Encoder.
- **Why:** Captures physical appearance, style, fit, and texture which metadata might miss.
- **Output:** 512-dimensional vector per product image (e.g. `[0.12, -0.44, 0.91...]`).

### STEP 3 — Generate Text Embeddings
- **Process:** Uses `FashionCLIP` Text Encoder on a concatenated string of `name + tags + description`.
- **Why:** Captures semantic meaning, occasion, and style definitions explicitly.
- **Output:** 512-dimensional vector per product text.

### STEP 4 — Extract Dominant Colors
- **Process:** Runs `OpenCV` KMeans clustering directly on the product images.
- **Why:** Pure algorithmic understanding of color harmony for the Pair Compatibility engine.
- **Output:** `{ "primary": "white", "secondary": "grey" }`

### STEP 5 — Build Outfit Graph
- **Process:** Iterates over `outfits.csv` and draws undirected edges between curated items (e.g. `White Shirt ↔ Grey Trouser`).
- **Why:** Captures human-stylist-approved outfit structures that AI alone might overlook.

### STEP 6 — Create Vector Index
- **Process:** Aggregates the 512d image and text vectors into a searchable array space.

> [!TIP]
> **Engineering Decision: NumPy vs. FAISS**
> While standard architecture dictates using FAISS for nearest-neighbor similarity search, this was intentionally swapped to `numpy.dot` arrays in production due to Apple Silicon (`arm64`) OS bugs with FAISS. Because the dataset isn't millions of rows, NumPy's brute-force loop handles the dot product instantly while remaining stable!

---

## ⚡ ONLINE PHASE (api.py / pipeline.py)

### STEP 1 & 2 — User Input & Intent Extraction
- **Input:** User query.
- **Process:** Gemini LLM transforms natural language into structured JSON format (e.g., `{ "occasion": "office", "style": "formal", "gender": "male" }`).

### STEP 3 — Metadata Filtering
- **Process:** Immediately filters the product database down to valid candidates that strictly match the intent constraints.
- **Why:** Computationally cheaper and prevents highly-retrieved but completely irrelevant items (like a swimsuit for an office meeting).

> [!NOTE]
> **Production Fallback ("Relaxation"):** If the strict filter removes 100% of candidates (e.g., obscure query), the system catches the empty array and triggers a relaxation fallback—dropping `occasion` and `style` constraints—preferring to show a broad generic result rather than crashing to a dead-end UI.

### STEP 4 — Retrieval Scoring
- **Process:** The LLM query is embedded using the text encoder.
- **Math:** 
  `image_sim = cos(query_emb, image_emb)`
  `text_sim = cos(query_emb, text_emb)`
  `retrieval_score = (0.7 * image_sim) + (0.3 * text_sim)`
- **Output:** A mathematically ranked list of individual candidates.

### STEP 5 — Graph Expansion (Plausible Pairings)
- **Goal:** Finding the pants that go with the retrieved shirt.
- **Process:** If the retrieved `Blue Shirt` isn't in the stylist graph, the system finds the *most similar known shirt* (e.g., `White Shirt`) via nearest-neighbor embedding math, and **inherits** its known connections (e.g., `Grey Trousers`).

### STEP 6 — Pair Compatibility Scoring
- **Component 1 (Graph Score):** `1.0` if direct edge, `0.7` if inherited edge, `0.0` if unknown.
- **Component 2 (Metadata Score):** Measures if the categories and occasions actually match functionally.
- **Component 3 (Color Score):** Mathematically evaluates the KMeans color harmony (e.g. Navy + White = High).
- **Formula:** `pair_score = (0.4 * graph_score) + (0.3 * metadata_score) + (0.3 * color_score)`
- **Action:** Any pairing scoring `< 0.5` is forcefully rejected.

### STEP 7 & 8 — Outfit Construction & Preference
- **Construction:** Builds Outfits (Top + Bottom + Shoe + Accessory) using only pairs that passed Step 6.
- **Preference:** Boosts final scores based on static user preferences (e.g., age bracket, minimalist preferences).

### STEP 9 & 10 — Outfit Ranking
- **Formula:** 
  `outfit_score = (0.50 * compatibility) + (0.20 * preference) + (0.20 * retrieval) + (0.10 * occasion_match)`
- **Output:** Sorted array. Top 3 outfits are passed to the frontend.

### STEP 11 — LLM Explanation
- **Process:** The exact components of the #1 outfit, alongside its mathematical scores and user intent, are passed back to the LLM. 
- **Output:** A natural language paragraph explaining *why* the outfit was built, referencing color harmony and occasion fit.

---

## 🖱 MODE 2 (GUIDED BUILDER)
The Guided Builder uses the exact same `pipeline.py` engine, but alters the interaction model to allow human-in-the-loop decisions.

**Flow:**
1. **Topwear:** Uses Retrieval Score.
2. **Bottomwear:** Uses Graph Expansion + Pair Compatibility against Topwear.
3. **Footwear:** Uses Pair Compatibility against Bottomwear.
4. **Accessory:** *(Live Nuance)* Specifically attempts to match with the Topwear and Footwear for cohesive aesthetic finishing.
5. **Finalize:** Construct complete outfit.

> [!TIP]
> **Production Nuance: "Soft Filtering" (Sorting Boost)**
> In the live code, strict matches receive a massive `+1.0` sorting boost (`is_strict_match`). This perfectly solves the issue of casual graph-expanded items creeping into formal lists. The strictly matching items are forced to the top, while graph-inherited items are pushed to the bottom and flagged visually as "Creative Matches".
