"""STEP 2 (online) — Intent extraction via Groq (free tier) with rule-based fallback."""
import json

from src.logging_utils import append_trace
from src.online.llm import call_llm, parse_json_from_llm

KEYWORD_MAP = {
    "occasion": {
        "office": ["office", "business", "meeting", "work", "interview", "boardroom", "corporate"],
        "party": ["party", "night out", "evening", "cocktail", "club"],
        "casual": ["casual", "weekend", "everyday", "relaxed"],
        "wedding": ["wedding", "ceremony", "reception", "bridal"],
        "vacation": ["vacation", "beach", "holiday", "travel", "summer"],
        "festive": ["festive", "diwali", "celebration", "festival"],
        "sports": ["gym", "workout", "sports", "athleisure", "running"],
        "winter": ["winter", "cold", "layered"],
    },
    "style": {
        "formal": ["formal", "professional", "business", "sharp", "polished"],
        "smart-casual": ["smart casual", "smart-casual", "dinner date"],
        "casual": ["casual", "laid back", "relaxed"],
        "ethnic": ["ethnic", "traditional", "kurta", "sherwani", "saree"],
        "western": ["western"],
    },
    "formality": {
        "high": ["formal", "business", "wedding", "interview", "ceremony"],
        "medium": ["smart casual", "dinner", "office casual"],
        "low": ["casual", "beach", "weekend", "gym"],
    },
}


def _rule_based_intent(query: str, profile: dict) -> dict:
    q = query.lower()
    occasion = "casual"
    style = "casual"
    formality = "medium"
    constraints = []

    for occ, kws in KEYWORD_MAP["occasion"].items():
        if any(k in q for k in kws):
            occasion = occ
            constraints.append(f"occasion:{occ}")
            break

    for st, kws in KEYWORD_MAP["style"].items():
        if any(k in q for k in kws):
            style = st
            constraints.append(f"style:{st}")
            break

    for fm, kws in KEYWORD_MAP["formality"].items():
        if any(k in q for k in kws):
            formality = fm
            constraints.append(f"formality:{fm}")
            break

    gender = profile.get("gender", "men")
    if "male" in q or "men" in q:
        gender = "men"
    elif "female" in q or "women" in q:
        gender = "women"

    return {
        "occasion": occasion,
        "style": style,
        "formality": formality,
        "gender": gender,
        "theme": "",
        "constraints": constraints or [f"occasion:{occasion}", f"style:{style}"],
        "confidence": 0.6,
        "source": "rule_based_fallback",
    }


def extract_intent(query: str, profile: dict, trace: list) -> dict:
    raw_prompt = (
        "Parse this fashion query into JSON with keys: "
        "occasion, style, formality, gender, constraints (list of strings), confidence (0-1).\n"
        "CRITICAL: 'occasion' MUST be exactly one of: [office, party, casual, wedding, vacation, festive, sports, winter].\n"
        "CRITICAL: 'style' MUST be exactly one of: [formal, smart-casual, casual, ethnic, western].\n"
        "CRITICAL: 'formality' MUST be exactly one of: [high, medium, low].\n"
        "CRITICAL: 'gender' MUST be exactly one of: [men, women, unisex].\n"
        f"User profile: gender={profile.get('gender')}, age={profile.get('age')}\n"
        f"Query: {query}\n"
        "Return ONLY valid JSON, no markdown."
    )

    parsed = None
    llm_error = None

    text, llm_error = call_llm(
        raw_prompt,
        system="You extract structured fashion intent from natural language. Output JSON only.",
    )
    if text:
        parsed = parse_json_from_llm(text)
        if parsed:
            parsed["source"] = "groq"

    if not parsed:
        parsed = _rule_based_intent(query, profile)
        if llm_error:
            append_trace("llm_failure", {"error": llm_error, "fallback": "rule_based"}, trace)

    parsed.setdefault("gender", profile.get("gender", "men"))
    parsed.setdefault("occasion", parsed.get("occasion", "casual"))
    parsed.setdefault("style", parsed.get("style", "casual"))
    parsed.setdefault("formality", parsed.get("formality", "medium"))
    parsed.setdefault("theme", "")
    parsed.setdefault("constraints", [])
        
    parsed.setdefault("confidence", 0.7)

    append_trace("intent_extraction", {
        "raw_prompt": raw_prompt,
        "parsed_output": parsed,
        "llm_error": llm_error,
    }, trace)
    return parsed
