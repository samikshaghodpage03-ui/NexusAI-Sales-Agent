"""
Tool definitions — real callable functions, not prompt injections.
Each tool is also registered as an Anthropic tool schema for the agent loop.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models import FlaggedConversation
from app.memory.backend import get_memory_backend

# ── Load catalog once at import time ─────────────────────────────
_CATALOG_PATH = Path(__file__).parent.parent.parent / "data" / "catalog.json"
_CATALOG: dict = json.loads(_CATALOG_PATH.read_text())


# ── Tool: search_catalog ──────────────────────────────────────────
def search_catalog(query: str) -> str:
    """
    Keyword/semantic search over the product catalog.
    Returns relevant sections as JSON text for the agent to use.
    """
    query_lower = query.lower()
    tokens = set(re.findall(r"\w+", query_lower))

    results: list[dict] = []

    # Score plans
    for plan in _CATALOG.get("plans", []):
        plan_text = (
            plan["name"].lower()
            + " "
            + plan["price"].lower()
            + " "
            + " ".join(f.lower() for f in plan["features"])
        )
        score = sum(1 for t in tokens if t in plan_text)
        if score > 0:
            results.append({"type": "plan", "score": score, "data": plan})

    # Score add-ons
    for addon in _CATALOG.get("add_ons", []):
        addon_text = addon["name"].lower() + " " + addon.get("description", "").lower()
        score = sum(1 for t in tokens if t in addon_text)
        if score > 0:
            results.append({"type": "addon", "score": score, "data": addon})

    # Score FAQs
    for faq in _CATALOG.get("faqs", []):
        faq_text = faq["question"].lower() + " " + faq["answer"].lower()
        score = sum(1 for t in tokens if t in faq_text)
        if score > 0:
            results.append({"type": "faq", "score": score, "data": faq})

    # Sort by relevance score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:5]

    if not top:
        # Return full catalog summary if no keyword match
        return json.dumps({
            "note": "No specific match found. Here is the full catalog summary.",
            "plans": [
                {"name": p["name"], "price": p["price"], "users": p["users"]}
                for p in _CATALOG["plans"]
            ],
        })

    return json.dumps([r["data"] for r in top], indent=2)


# ── Tool: get_user_memory ─────────────────────────────────────────
def get_user_memory(user_id: str, db: Session) -> str:
    """
    Retrieves relevant past context about a user from the DB.
    Returns a prose summary of prior interests and questions.
    """
    backend = get_memory_backend(db)
    facts = backend.get_user_facts(user_id)
    recent = backend.get_recent_messages(user_id, limit=10)

    recent_text = ""
    if recent:
        recent_text = "\n\nRecent conversation turns:\n"
        for msg in recent[-6:]:  # last 6 turns
            prefix = "User" if msg["role"] == "user" else "Assistant"
            recent_text += f"{prefix}: {msg['content'][:300]}\n"

    return facts + recent_text


# ── Tool: flag_for_human ──────────────────────────────────────────
def flag_for_human(user_id: str, session_id: str, reason: str, db: Session) -> str:
    """
    Escalates a conversation to human review when confidence is low.
    Logs to flagged_conversations table.
    """
    flag = FlaggedConversation(
        user_id=user_id,
        session_id=session_id,
        reason=reason,
    )
    db.add(flag)
    db.commit()
    return f"Flagged for human review. Reason: {reason}"


# ── Catalog accessor (used by /catalog endpoint) ──────────────────
def get_full_catalog() -> dict:
    return _CATALOG