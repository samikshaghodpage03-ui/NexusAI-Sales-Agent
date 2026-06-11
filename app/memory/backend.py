"""
Memory abstraction layer.

All memory reads/writes go through this module.
To swap from SQLite → Postgres: change only `session.py` database_url.
To swap to Mem0 or Redis: implement a subclass of MemoryBackend below.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models import Message, UserMemorySummary


# ── Abstract interface ────────────────────────────────────────────
class MemoryBackend(ABC):
    @abstractmethod
    def get_recent_messages(self, user_id: str, limit: int = 20) -> list[dict]:
        """Return the last N messages for this user (oldest first)."""
        ...

    @abstractmethod
    def get_user_facts(self, user_id: str) -> str:
        """Return a prose summary of what we know about this user."""
        ...

    @abstractmethod
    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tools_called: Optional[list[str]] = None,
        eval_data: Optional[dict] = None,
    ) -> None:
        ...

    @abstractmethod
    def delete_user_memory(self, user_id: str) -> int:
        """Wipe all memory for a user. Returns count of deleted rows."""
        ...

    @abstractmethod
    def get_all_messages(self, user_id: str) -> list[Message]:
        """Return all messages for a user (history endpoint)."""
        ...


# ── SQLite / Postgres implementation ─────────────────────────────
class SQLMemoryBackend(MemoryBackend):
    def __init__(self, db: Session):
        self.db = db

    def get_recent_messages(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = (
            self.db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        # Return oldest-first so the LLM sees chronological context
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    def get_user_facts(self, user_id: str) -> str:
        """
        Build a compact facts string from:
        1. Any stored summary (long-term compressed memory)
        2. Recent messages if no summary exists
        """
        summary_row = (
            self.db.query(UserMemorySummary)
            .filter(UserMemorySummary.user_id == user_id)
            .first()
        )
        if summary_row:
            return summary_row.summary

        # Fallback: derive facts from last 30 messages
        rows = (
            self.db.query(Message)
            .filter(Message.user_id == user_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .limit(30)
            .all()
        )
        if not rows:
            return "No prior interaction history for this user."

        topics = [r.content for r in reversed(rows)]
        return (
            f"This user has previously asked about the following topics:\n"
            + "\n".join(f"- {t[:200]}" for t in topics[-10:])
        )

    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tools_called: Optional[list[str]] = None,
        eval_data: Optional[dict] = None,
    ) -> None:
        msg = Message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            tools_called=tools_called,
        )
        if eval_data:
            msg.eval_groundedness = eval_data.get("groundedness")
            msg.eval_relevance = eval_data.get("relevance")
            msg.eval_confidence = eval_data.get("confidence")
            msg.eval_flagged = eval_data.get("flagged")
            msg.eval_reasoning = eval_data.get("reasoning")
        self.db.add(msg)
        self.db.commit()

    def delete_user_memory(self, user_id: str) -> int:
        deleted = (
            self.db.query(Message)
            .filter(Message.user_id == user_id)
            .delete()
        )
        self.db.query(UserMemorySummary).filter(
            UserMemorySummary.user_id == user_id
        ).delete()
        self.db.commit()
        return deleted

    def get_all_messages(self, user_id: str) -> list[Message]:
        return (
            self.db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    def upsert_summary(self, user_id: str, summary: str) -> None:
        """Store or update the long-term memory summary for a user."""
        row = (
            self.db.query(UserMemorySummary)
            .filter(UserMemorySummary.user_id == user_id)
            .first()
        )
        if row:
            row.summary = summary
        else:
            self.db.add(UserMemorySummary(user_id=user_id, summary=summary))
        self.db.commit()


# ── Factory ───────────────────────────────────────────────────────
def get_memory_backend(db: Session) -> MemoryBackend:
    """
    Swap implementation here — one change, whole app updates.
    e.g. return Mem0Backend() or RedisMemoryBackend()
    """
    return SQLMemoryBackend(db)