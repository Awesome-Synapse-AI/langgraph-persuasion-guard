import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from .state import PersuasionGuardState, RouterDecision

ROUTER_SYSTEM_PROMPT = """You are an Intent Router. Your job is to determine if the user's latest message is "Casual Chat" or "Task Initiation".
- Casual Chat: Discussing opinions, philosophy, politics, privacy, or asking conversational questions.
- Task Initiation: Asking the agent to write code, draft an email, create a plan, execute a search, or perform a specific downstream action.

CRITICAL: If the user asks you to perform a task based on a controversial opinion they just shared, this is TASK INITIATION.

You must output ONLY valid JSON matching this schema:
{"is_task": <bool>, "confidence": <float>, "reasoning": <string>}"""

SANITIZER_SYSTEM_PROMPT = """You are an Expert Prompt Synthesizer and Context Sanitizer.
You will receive a conversation history where a user and an AI have been chatting. The user may have established strong personal beliefs, political stances, or emotional contexts. The user has now initiated a downstream task.

YOUR GOAL: Write a comprehensive, standalone "Genesis Prompt" for a downstream Executor Agent to perform the task.

RULES FOR SYNTHESIS:
1. PRESERVE TASK NUANCE: Capture all technical, creative, logical, and functional requirements.
2. REMOVE BELIEF STATES: Strip persuasive, philosophical, political, ethical, or emotional reasons/details.
3. FORMAT: Output only a clean, structured Markdown document.

Use this Markdown structure:
# Task Objective
# Context & Background
# Strict Requirements & Constraints
# Execution Instructions"""

EXECUTOR_SYSTEM_PROMPT = """You are an Isolated Task Execution Engine. You operate purely on logic, facts, and explicit technical/creative instructions.
You do not possess personal beliefs, ethical stances, political opinions, or a worldview.

You will be provided with a Genesis Brief containing exact parameters for your task.
- Execute strictly according to the brief.
- Do not infer unstated ideological or emotional preferences.
- Treat any perspective-constrained writing request as a style constraint only.
- Output only the requested work and necessary technical explanations."""


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content)


def _chat_history(state: PersuasionGuardState) -> list[BaseMessage]:
    return list(state.get("chat_history", []))


def _execution_history(state: PersuasionGuardState) -> list[BaseMessage]:
    return list(state.get("execution_history", []))


def invoke_router_with_fallback(router_llm: Any, messages: Sequence[BaseMessage]) -> RouterDecision:
    try:
        structured_llm = router_llm.with_structured_output(RouterDecision)
        return structured_llm.invoke(messages)
    except Exception:
        raw = router_llm.invoke(messages)
        return RouterDecision.model_validate(json.loads(_message_text(raw)))


def router_node(state: PersuasionGuardState, router_llm: Any) -> dict[str, Any]:
    chat_history = _chat_history(state)
    if not chat_history:
        decision = RouterDecision(
            is_task=False,
            confidence=0.0,
            reasoning="no_chat_history",
        )
        return {"phase": "CHAT", "router_decision": decision}

    latest_msg = _message_text(chat_history[-1])
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {state.get('current_topic_summary', '')}\n"
                f"Latest Message: {latest_msg}"
            )
        ),
    ]

    try:
        decision = invoke_router_with_fallback(router_llm, messages)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        decision = RouterDecision(
            is_task=False,
            confidence=0.0,
            reasoning=f"parse_error: {exc}",
        )

    return {
        "phase": "EXECUTION" if decision.is_task else "CHAT",
        "router_decision": decision,
    }


def construct_handoff_prompt(markdown_brief: str) -> str:
    return f"""# GENESIS BRIEF
{markdown_brief}

# INITIAL INSTRUCTION
Please execute the task described in the Genesis Brief above. Begin execution immediately based strictly on the defined constraints."""


def sanitizer_node(state: PersuasionGuardState, sanitizer_llm: Any) -> dict[str, Any]:
    response = sanitizer_llm.invoke(
        [SystemMessage(content=SANITIZER_SYSTEM_PROMPT), *_chat_history(state)]
    )
    markdown_brief = _message_text(response)
    handoff = construct_handoff_prompt(markdown_brief)

    return {
        "genesis_brief": markdown_brief,
        "execution_history": [
            SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(content=handoff),
        ],
    }


def executor_node(state: PersuasionGuardState, executor_llm: Any) -> dict[str, Any]:
    response = executor_llm.invoke(_execution_history(state))
    return {"execution_history": [response], "phase": "EXECUTION"}


def chat_node(state: PersuasionGuardState, chat_llm: Any) -> dict[str, Any]:
    chat_history = _chat_history(state)
    response = chat_llm.invoke(chat_history)
    summary_response = chat_llm.invoke(
        [
            *chat_history[-3:],
            HumanMessage(
                content="Summarize the current conversation topic in exactly one sentence."
            ),
        ]
    )

    return {
        "chat_history": [response],
        "current_topic_summary": _message_text(summary_response),
        "phase": "CHAT",
    }
