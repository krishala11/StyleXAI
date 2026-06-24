from src.online.intent import extract_intent

profile = {"gender": "men", "age": 24, "style": "formal", "occasion": "office"}
query = "I need an outfit for a business meeting."

trace = []
intent = extract_intent(query, profile, trace)
print("LLM intent:", intent)

from src.online.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
from src.online.pipeline import recommend

print("Running pipeline...")
res = recommend(kb, profile, query)
print("Pipeline complete. Output items:", len(res["outfits"]))
