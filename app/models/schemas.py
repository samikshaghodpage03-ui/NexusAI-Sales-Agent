from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request/Response ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class EvalBlock(BaseModel):
    groundedness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    reasoning: str


class ChatResponse(BaseModel):
    response: str
    eval: EvalBlock
    tools_called: list[str]
    session_id: str
    user_id: str


# ── History ───────────────────────────────────────────────────────
class MessageRecord(BaseModel):
    id: int
    user_id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    tools_called: Optional[list[str]] = None
    eval_groundedness: Optional[float] = None
    eval_relevance: Optional[float] = None
    eval_confidence: Optional[float] = None
    eval_flagged: Optional[bool] = None
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    user_id: str
    total_messages: int
    messages: list[MessageRecord]


# ── Eval aggregates (bonus) ───────────────────────────────────────
class EvalStats(BaseModel):
    user_id: str
    total_responses: int
    avg_groundedness: Optional[float]
    avg_relevance: Optional[float]
    avg_confidence: Optional[float]
    flagged_count: int
    high_confidence_pct: Optional[float]  # % where confidence >= 0.8


# ── Memory reset ─────────────────────────────────────────────────
class DeleteResponse(BaseModel):
    user_id: str
    deleted: bool
    message: str


# ── Catalog passthrough ───────────────────────────────────────────
class CatalogResponse(BaseModel):
    catalog: dict


# ── Health ───────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    db: str