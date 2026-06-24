import os

# Set environment variables immediately to prevent MacOS threading bus errors 
# with PyTorch, FAISS, and Hugging Face tokenizers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

import config
from src.online.knowledge_base import KnowledgeBase
from src.online.pipeline import recommend

# Global Knowledge Base instance
kb = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb
    print("Initializing AI Models and FAISS Index (This may take a moment)...")
    try:
        kb = KnowledgeBase()
        print("Knowledge Base successfully loaded into memory!")
    except Exception as e:
        print(f"Failed to load Knowledge Base: {e}")
    yield
    print("Shutting down AI Backend...")

app = FastAPI(title="Fashion AI Recommendation Engine", lifespan=lifespan)

# Mount the static data directory so Next.js can load images
app.mount("/data", StaticFiles(directory=str(config.DATA_DIR)), name="data")

# Allow Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserProfile(BaseModel):
    gender: str = "men"
    age: int = 24
    style: str = "formal"
    occasion: str = "office"

class ChatRequest(BaseModel):
    query: str
    profile: UserProfile

class GuidedRequest(BaseModel):
    query: str
    profile: UserProfile
    selections: Dict[str, Any]

@app.post("/api/recommend")
async def api_recommend(req: ChatRequest):
    print(f"\n--- API HIT: /api/recommend ---")
    if not kb:
        print("ERROR: Knowledge base not loaded")
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    print(f"Request Query: {req.query}")
    # Run the recommendation pipeline (Mode 1)
    result = recommend(kb, req.profile.model_dump(), req.query, mode=1)
    print(f"--- API SUCCESS: Returning results ---")
    return result

@app.post("/api/guided")
async def api_guided(req: GuidedRequest):
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    # Run the recommendation pipeline (Mode 2)
    result = recommend(kb, req.profile.model_dump(), req.query, mode=2, selections=req.selections)
    return result

@app.get("/api/health")
def health_check():
    return {"status": "ok", "kb_loaded": kb is not None}
