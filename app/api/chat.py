from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import Message, FlaggedConversation
from app.memory.backend import get_memory_backend
from app.services.chat_service import process_chat
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageRecord,
    DeleteResponse,
    EvalStats,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{user_id}", response_model=ChatResponse)
def chat(user_id: str, body: ChatRequest, db: Session = Depends(get_db)):
    try:
        if not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")

        return process_chat(
            user_id=user_id,
            user_message=body.message,
            db=db,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/history", response_model=HistoryResponse)
def get_history(user_id: str, db: Session = Depends(get_db)):
    """Full conversation history for a user across all sessions."""
    memory = get_memory_backend(db)
    messages = memory.get_all_messages(user_id)
    return HistoryResponse(
        user_id=user_id,
        total_messages=len(messages),
        messages=[MessageRecord.model_validate(m) for m in messages],
    )


@router.delete("/{user_id}/memory", response_model=DeleteResponse)
def delete_memory(user_id: str, db: Session = Depends(get_db)):
    """Wipe all memory for a user (GDPR-style reset)."""
    memory = get_memory_backend(db)
    deleted_count = memory.delete_user_memory(user_id)
    return DeleteResponse(
        user_id=user_id,
        deleted=True,
        message=f"Deleted {deleted_count} messages for user '{user_id}'.",
    )


@router.get("/{user_id}/evals", response_model=EvalStats)
def get_eval_stats(user_id: str, db: Session = Depends(get_db)):
    """Aggregated eval scores across all sessions (bonus endpoint)."""
    rows = (
        db.query(Message)
        .filter(
            Message.user_id == user_id,
            Message.role == "assistant",
            Message.eval_confidence.isnot(None),
        )
        .all()
    )

    if not rows:
        return EvalStats(
            user_id=user_id,
            total_responses=0,
            avg_groundedness=None,
            avg_relevance=None,
            avg_confidence=None,
            flagged_count=0,
            high_confidence_pct=None,
        )

    total = len(rows)
    avg_g = sum(r.eval_groundedness or 0 for r in rows) / total
    avg_r = sum(r.eval_relevance or 0 for r in rows) / total
    avg_c = sum(r.eval_confidence or 0 for r in rows) / total
    flagged = sum(1 for r in rows if r.eval_flagged)
    high_conf = sum(1 for r in rows if (r.eval_confidence or 0) >= 0.8)

    return EvalStats(
        user_id=user_id,
        total_responses=total,
        avg_groundedness=round(avg_g, 3),
        avg_relevance=round(avg_r, 3),
        avg_confidence=round(avg_c, 3),
        flagged_count=flagged,
        high_confidence_pct=round(high_conf / total * 100, 1),
    )