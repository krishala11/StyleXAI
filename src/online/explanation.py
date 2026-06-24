"""STEP 11 — LLM explanation via Groq with template fallback."""
from src.logging_utils import append_trace
from src.online.llm import call_llm


def _template_explanation(outfit: dict, intent: dict, profile: dict) -> str:
    items = outfit["items"]
    scores = outfit["scores"]
    names = [items[s]["name"] for s in ("topwear", "bottomwear", "footwear", "accessory") if s in items]
    occasion = intent.get("occasion", "the occasion")
    style = profile.get("style") or intent.get("style", "your style")

    return (
        f"This outfit was selected because {' + '.join(names[:3])} align with {occasion} formality "
        f"and {style} preference. Pair compatibility averaged {scores['compatibility_score']:.2f}, "
        f"with strong color harmony. The neutral palette maintains visual balance suitable for the occasion."
    )


def generate_explanation(outfit: dict, intent: dict, profile: dict, trace: list) -> str:
    items = outfit["items"]
    item_list = "\n".join(
        f"- {slot}: {info['name']} ({info['id']})"
        for slot, info in items.items()
    )
    scores = outfit["scores"]

    prompt = f"""Explain why this outfit was recommended. ONLY mention items listed below.
User: gender={profile.get('gender')}, age={profile.get('age')}, style={profile.get('style')}
Intent: occasion={intent.get('occasion')}, style={intent.get('style')}, formality={intent.get('formality')}
Outfit items:
{item_list}
Scores: compatibility={scores['compatibility_score']}, preference={scores['preference_score']}, retrieval={scores['retrieval_score']}
Mention: occasion fit, color harmony, compatibility, user preference. 2-3 sentences. Do not invent items."""

    explanation, error = call_llm(
        prompt,
        system="You are a fashion stylist explaining outfit recommendations clearly and honestly.",
    )

    if not explanation:
        explanation = _template_explanation(outfit, intent, profile)
        append_trace("explanation_fallback", {"error": error}, trace)

    return explanation
