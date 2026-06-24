import json
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify Fashion Recommender Dataset")
    parser.add_argument("--occasion", type=str, help="Filter by a specific occasion (e.g., 'casual', 'office')")
    parser.add_argument("--category", type=str, help="Filter by a specific category (e.g., 'party-dresses')")
    args = parser.parse_args()

    try:
        with open("artifacts/products.json", "r") as f:
            products = json.load(f)
    except FileNotFoundError:
        print("Error: artifacts/products.json not found.")
        return

    print(f"\nTotal Products in Database: {len(products)}")

    filtered_products = products
    if args.occasion:
        filtered_products = [p for p in filtered_products if str(p.get("occasion")).lower() == args.occasion.lower()]
        print(f"\nFiltering by occasion '{args.occasion}': Found {len(filtered_products)} items.")
    
    if args.category:
        filtered_products = [p for p in filtered_products if str(p.get("category")).lower() == args.category.lower()]
        print(f"\nFiltering by category '{args.category}': Found {len(filtered_products)} items.")

    if args.occasion or args.category:
        print("\nMatching Items:")
        for p in filtered_products:
            print(f" - [{p['id']}] {p.get('name', 'Unknown')[:50]}... | Category: {p.get('category')} | Occasion: {p.get('occasion')}")
    else:
        print("\n--- Breakdown by Occasion ---")
        occasions = Counter([str(p.get("occasion", "none")).lower() for p in products])
        for occ, count in occasions.most_common():
            print(f" - {occ}: {count} items")

        print("\n--- Breakdown by Category ---")
        categories = Counter([str(p.get("category", "none")).lower() for p in products])
        for cat, count in categories.most_common():
            print(f" - {cat}: {count} items")

    print("\nTip: If the AI only shows '1 dress' for a specific occasion, check the breakdown above. If there is only 1 dress in the database, the AI cannot recommend anything else!")

if __name__ == "__main__":
    main()
