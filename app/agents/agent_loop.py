"""
Agent loop — orchestrates tool calls, memory injection, and response generation.

Uses Google Gemini with native function calling.

Flow:
  1. Pull user memory from DB via get_user_memory tool
  2. Send message + history + tools to Gemini
  3. Handle function_call parts (search_catalog, get_user_memory, flag_for_human)
  4. Continue loop until Gemini returns a final text response
  5. Return response + tools_called list for eval
"""
from __future__ import annotations
import json
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from sqlalchemy.orm import Session
from app.config import get_settings
from app.tools.catalog_tools import (
    search_catalog,
    get_user_memory,
    flag_for_human,
)

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)

SYSTEM_PROMPT = """You are Alex, an expert sales assistant for NexusAI — an AI-native CRM platform.

Your job:
- Answer questions about NexusAI's plans, pricing, and features accurately
- Always use the search_catalog function to look up product information instead of guessing
- Always use get_user_memory at the start of every conversation to recall past context
- Be concise, friendly, and consultative — you're helping a prospect evaluate the product
- Never make up features or prices. If you're not sure, say so honestly
- If a user's question is complex or you're uncertain, use flag_for_human to escalate

Remember: accuracy > confidence. A short honest answer beats a long hallucinated one."""

# ── Gemini tool definitions ───────────────────────────────────────
GEMINI_TOOLS = Tool(function_declarations=[
    FunctionDeclaration(
        name="search_catalog",
        description=(
            "Search the NexusAI product catalog for pricing, plan features, "
            "add-ons, and FAQs. Use this whenever the user asks about plans, "
            "pricing, features, SSO, SLA, credits, or anything product-related."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'enterprise pricing SSO'",
                }
            },
            "required": ["query"],
        },
    ),
    FunctionDeclaration(
        name="get_user_memory",
        description=(
            "Retrieve the conversation history and facts about this specific user "
            "from the database. Call this at the start of every response to maintain "
            "continuity across sessions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user's unique identifier",
                }
            },
            "required": ["user_id"],
        },
    ),
    FunctionDeclaration(
        name="flag_for_human",
        description=(
            "Escalate this conversation for human review when you are uncertain, "
            "when the user seems frustrated, or when the question is outside your knowledge."
        ),
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "session_id": {"type": "string"},
                "reason": {
                    "type": "string",
                    "description": "Why this conversation needs human attention",
                },
            },
            "required": ["user_id", "session_id", "reason"],
        },
    ),
])


def run_agent(
    user_id: str,
    session_id: str,
    user_message: str,
    db: Session,
) -> tuple[str, list[str], str]:
    """
    Run the agent loop for a single user message.

    Returns:
        (response_text, tools_called, tool_data_used)
    """
    tools_called: list[str] = []
    tool_data_collected: list[str] = []

    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
        tools=[GEMINI_TOOLS],
    )

    # Build initial prompt — ask model to fetch memory first
    initial_prompt = (
        f"[Fetch user memory first, then answer]\n\n"
        f"User ID: {user_id}\n\n"
        f"User message: {user_message}"
    )

    # Start a chat session so history is tracked automatically
    chat = model.start_chat()

    # ── Agentic loop ─────────────────────────────────────────────
    max_iterations = 6
    current_message = initial_prompt

    for _ in range(max_iterations):
        response = chat.send_message(current_message)
        candidate = response.candidates[0]

        # Collect function calls from all parts
        function_calls = []
        text_parts = []

        for part in candidate.content.parts:
            if part.function_call.name:
                function_calls.append(part.function_call)
            elif part.text:
                text_parts.append(part.text)

        # No function calls → final answer
        if not function_calls:
            final_text = " ".join(text_parts).strip()
            return final_text or "I couldn't process that request.", tools_called, "\n\n".join(tool_data_collected)

        # ── Handle function calls ─────────────────────────────────
        function_responses = []
        for fc in function_calls:
            tool_name = fc.name
            tool_input = dict(fc.args)
            tools_called.append(tool_name)

            if tool_name == "search_catalog":
                result = search_catalog(tool_input.get("query", ""))
                tool_data_collected.append(f"[search_catalog] {result}")

            elif tool_name == "get_user_memory":
                result = get_user_memory(
                    user_id=tool_input.get("user_id", user_id),
                    db=db,
                )
                tool_data_collected.append(f"[get_user_memory] {result}")

            elif tool_name == "flag_for_human":
                result = flag_for_human(
                    user_id=tool_input.get("user_id", user_id),
                    session_id=tool_input.get("session_id", session_id),
                    reason=tool_input.get("reason", "Low confidence"),
                    db=db,
                )
            else:
                result = f"Unknown tool: {tool_name}"

            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result},
                    )
                )
            )

        # Feed all tool results back in one message
        current_message = function_responses

    # Fallback if we hit max iterations
    return (
        "I've gathered information but need more time to process this. Please try again.",
        tools_called,
        "\n\n".join(tool_data_collected),
    )