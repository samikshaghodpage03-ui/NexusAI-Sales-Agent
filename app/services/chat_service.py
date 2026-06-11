"""
Chat service — top-level orchestrator.

Connects:
  - Agent loop (generation + tool calls)
  - Memory backend (persistence)
  - Eval service (self-scoring)
  - flag_for_human (escalation)
"""
from __future__ import annotations
import uuid
from sqlalchemy.orm import Session
from app.agents.agent_loop import run_agent
from app.memory.backend import get_memory_backend
from app.services.eval_service import evaluate_response
from app.tools.catalog_tools import flag_for_human
from app.config import get_settings
from app.models.schemas import ChatResponse, EvalBlock

settings = get_settings()


def process_chat(user_id: str, user_message: str, db: Session) -> ChatResponse:
    """
    Full pipeline: message → agent → eval → persist → response.
    """
    session_id = str(uuid.uuid4())
    memory = get_memory_backend(db)

    # 1. Persist user message
    memory.save_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=user_message,
    )

    # 2. Run agent (tool calls + generation)
    response_text, tools_called, tool_data = run_agent(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        db=db,
    )

    # 3. Self-evaluate
    eval_data = evaluate_response(
        user_message=user_message,
        assistant_response=response_text,
        tool_data_used=tool_data,
    )

    # 4. Auto-flag if confidence too low
    if eval_data["confidence"] < settings.flag_threshold:
        if "flag_for_human" not in tools_called:
            flag_for_human(
                user_id=user_id,
                session_id=session_id,
                reason=f"Auto-flagged: confidence={eval_data['confidence']:.2f}",
                db=db,
            )
            tools_called.append("flag_for_human")
        eval_data["flagged"] = True

    # 5. Persist assistant response with eval data
    memory.save_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content=response_text,
        tools_called=tools_called,
        eval_data=eval_data,
    )

    return ChatResponse(
        response=response_text,
        eval=EvalBlock(**eval_data),
        tools_called=tools_called,
        session_id=session_id,
        user_id=user_id,
    )