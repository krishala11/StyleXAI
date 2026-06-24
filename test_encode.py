import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
print("Loading KB...")
try:
    from src.online.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    print("KB successfully loaded.")
    print("Encoding text...")
    emb = kb.encoder.encode_text("I need an outfit for a business meeting.")
    print(f"Encoded shape: {emb.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()

