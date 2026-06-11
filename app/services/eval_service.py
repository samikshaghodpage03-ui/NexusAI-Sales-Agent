"""
Eval service — structured self-scoring on every response using Gemini.
"""
from __future__ import annotations
import json
import re
import google.generativeai as genai
from app.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)

EVAL_SYSTEM_PROMPT = """You are a strict response quality evaluator for a B2B SaaS sales assistant.

You will be given:
1. The user's question
2. The assistant's response
3. The catalog/tool data that was available

Score the response on three dimensions from 0.0 to 1.0:
- groundedness: Is every claim traceable to the catalog data provided? (1.0 = fully grounded, 0.0 = pure hallucination)
- relevance: Does the response directly answer what was asked? (1.0 = perfectly on-topic, 0.0 = completely off-topic)
- confidence: How certain are you in this answer's correctness? (1.0 = definitely correct, 0.0 = very uncertain)

Respond ONLY with valid JSON, no markdown fences, in exactly this format:
{
  "groundedness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "flagged": <true if any score < 0.65, else false>,
  "reasoning": "<one sentence explaining the scores>"
}"""


def evaluate_response(
    user_message: str,
    assistant_response: str,
    tool_data_used: str,
) -> dict:
    """
    Call Gemini to self-evaluate the response quality.
    Returns a dict with groundedness, relevance, confidence, flagged, reasoning.
    """
    eval_prompt = f"""User asked: {user_message}

Tool/catalog data available:
{tool_data_used[:2000]}

Assistant responded:
{assistant_response}

Score this response."""

    try:
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=EVAL_SYSTEM_PROMPT,
        )
        resp = model.generate_content(eval_prompt)
        raw = resp.text.strip()

        # Strip markdown fences if Gemini adds them anyway
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        return {
            "groundedness": float(result.get("groundedness", 0.7)),
            "relevance": float(result.get("relevance", 0.7)),
            "confidence": float(result.get("confidence", 0.7)),
            "flagged": bool(result.get("flagged", False)),
            "reasoning": str(result.get("reasoning", "Self-evaluation complete.")),
        }

    except Exception as e:
        return {
            "groundedness": 0.5,
            "relevance": 0.5,
            "confidence": 0.5,
            "flagged": True,
            "reasoning": f"Eval service error: {str(e)[:100]}. Flagged for review.",
        }