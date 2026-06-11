from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON
)
from sqlalchemy.sql import func
from .base import Base


class Message(Base):
    """Stores every user ↔ assistant message turn."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)          # "user" | "assistant"
    content = Column(Text, nullable=False)
    tools_called = Column(JSON, nullable=True)     # list[str]

    # Eval fields — only populated for assistant turns
    eval_groundedness = Column(Float, nullable=True)
    eval_relevance = Column(Float, nullable=True)
    eval_confidence = Column(Float, nullable=True)
    eval_flagged = Column(Boolean, nullable=True)
    eval_reasoning = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserMemorySummary(Base):
    """Optional: compressed long-term summary of what we know about a user.
    Enables the memory-summarization bonus feature."""
    __tablename__ = "user_memory_summaries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class FlaggedConversation(Base):
    """Log entries when an agent response is flagged for human review."""
    __tablename__ = "flagged_conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    reason = Column(Text, nullable=False)
    reviewed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())