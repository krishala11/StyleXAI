"""Free-tier LLM via Groq API (https://console.groq.com)."""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY not set"
    try:
        from groq import Groq
        return Groq(api_key=api_key), None
    except ImportError as e:
        return None, f"groq package not installed: {e}"


def call_llm(prompt: str, system: str = "You are a fashion AI assistant. Respond concisely.") -> tuple[str | None, str | None]:
    """Returns (response_text, error)."""
    client, err = _get_client()
    if client is None:
        return None, err

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)


def parse_json_from_llm(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None
