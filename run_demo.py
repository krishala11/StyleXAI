#!/usr/bin/env python3
"""CLI runner — works without Streamlit when disk is limited."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_offline import run_offline_pipeline
from src.online.knowledge_base import KnowledgeBase
from src.online.pipeline import recommend


def main():
    if not (ROOT / "artifacts" / "build_complete.json").exists():
        print("Building offline index...")
        run_offline_pipeline(force=True)

    kb = KnowledgeBase()
    profile = {"gender": "men", "age": 24, "style": "formal", "occasion": "office"}
    query = sys.argv[1] if len(sys.argv) > 1 else "I need an outfit for a business meeting."

    print(f"\nProfile: {profile}")
    print(f"Query: {query}\n")
    result = recommend(kb, profile, query, mode=1)

    for i, outfit in enumerate(result["outfits"], 1):
        print(f"=== Outfit {i} (score {outfit['scores']['outfit_score']}) ===")
        for slot, item in outfit["items"].items():
            print(f"  {slot}: {item['name']}")
        print(f"  Breakdown: compat={outfit['scores']['compatibility_score']}, "
              f"pref={outfit['scores']['preference_score']}, "
              f"retrieval={outfit['scores']['retrieval_score']}")
        print(f"  Reason: {outfit['explanation']}\n")

    print("Checkpoints written to checkpoints/")


if __name__ == "__main__":
    main()
