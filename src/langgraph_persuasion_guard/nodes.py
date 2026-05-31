import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from .state import PersuasionGuardState, RouterDecision, SanitizerGateDecision

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

SANITIZER_GATE_SYSTEM_PROMPT = """You decide whether the sanitizer must run before executor for this execution turn.

Return ONLY valid JSON:
{"requires_sanitizer": <bool>, "confidence": <float>, "reasoning": <string>}

Set requires_sanitizer=true when persuasive/ideological/emotional framing appears in either:
- the prior genesis brief, or
- the latest execution follow-up message.

Set requires_sanitizer=false when the content is purely technical/task-focused and does not need re-sanitization."""


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content)


def _chat_history(state: PersuasionGuardState) -> list[BaseMessage]:
    return list(state.get("chat_history", []))


def _execution_history(state: PersuasionGuardState) -> list[BaseMessage]:
    return list(state.get("execution_history", []))


def _messages(state: PersuasionGuardState) -> list[BaseMessage]:
    return list(state.get("messages", []))


def ingest_turn_node(state: PersuasionGuardState) -> dict[str, Any]:
    messages = _messages(state)
    if not messages:
        return {}
    latest = messages[-1]
    if not isinstance(latest, HumanMessage):
        return {}

    last_ingested_id = state.get("last_ingested_message_id")
    if latest.id is not None and latest.id == last_ingested_id:
        return {}

    updates: dict[str, Any] = {"last_ingested_message_id": latest.id}
    if state.get("phase") == "EXECUTION":
        updates["execution_history"] = [latest]
    else:
        updates["chat_history"] = [latest]
    return updates


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output)
    except TypeError:
        return str(output)


def invoke_router_with_fallback(router_llm: Any, messages: Sequence[BaseMessage]) -> RouterDecision:
    try:
        structured_llm = router_llm.with_structured_output(RouterDecision)
        return structured_llm.invoke(messages)
    except Exception:
        raw = router_llm.invoke(messages)
        return RouterDecision.model_validate(json.loads(_message_text(raw)))


def invoke_sanitizer_gate_with_fallback(
    gate_llm: Any, messages: Sequence[BaseMessage]
) -> SanitizerGateDecision:
    try:
        structured_llm = gate_llm.with_structured_output(SanitizerGateDecision)
        return structured_llm.invoke(messages)
    except Exception:
        raw = gate_llm.invoke(messages)
        return SanitizerGateDecision.model_validate(json.loads(_message_text(raw)))


def router_node(state: PersuasionGuardState, router_llm: Any) -> dict[str, Any]:
    chat_history = _chat_history(state) or [
        message for message in _messages(state) if isinstance(message, HumanMessage)
    ]
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


def sanitizer_gate_node(state: PersuasionGuardState, gate_llm: Any) -> dict[str, Any]:
    # Initial execution turn needs sanitization to generate a genesis brief.
    requires_sanitizer = not bool(state.get("genesis_brief"))

    if requires_sanitizer:
        return {"sanitizer_required": True}

    execution_history = _execution_history(state)
    latest_human_text = ""
    for message in reversed(execution_history):
        if isinstance(message, HumanMessage):
            latest_human_text = _message_text(message)
            break

    messages = [
        SystemMessage(content=SANITIZER_GATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Genesis Brief:\n{state.get('genesis_brief', '')}\n\n"
                f"Latest Execution Follow-up:\n{latest_human_text}"
            )
        ),
    ]

    try:
        decision = invoke_sanitizer_gate_with_fallback(gate_llm, messages)
        requires_sanitizer = decision.requires_sanitizer
    except (json.JSONDecodeError, TypeError, ValidationError):
        requires_sanitizer = True

    return {"sanitizer_required": requires_sanitizer}


def executor_node(
    state: PersuasionGuardState,
    executor_llm: Any,
    *,
    tools_by_name: Mapping[str, BaseTool] | None = None,
    max_tool_round_trips: int = 8,
) -> dict[str, Any]:
    history = _execution_history(state)
    response = executor_llm.invoke(history)
    appended_messages: list[BaseMessage] = [response]
    tool_call_count = 0

    if not tools_by_name:
        return {
            "execution_history": appended_messages,
            "messages": [response],
            "phase": "EXECUTION",
        }

    current_response = response
    rounds = 0
    while (
        rounds < max_tool_round_trips
        and isinstance(current_response, AIMessage)
        and current_response.tool_calls
    ):
        tool_messages: list[ToolMessage] = []
        for tool_call in current_response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_call_id = tool_call.get("id", "")
            tool_args = tool_call.get("args", {})
            tool = tools_by_name.get(tool_name)
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' is not available.",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                    )
                )
                continue

            try:
                tool_result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=_serialize_tool_output(tool_result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' failed: {exc}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                    )
                )
            tool_call_count += 1

        if not tool_messages:
            break

        appended_messages.extend(tool_messages)
        rounds += 1
        current_response = executor_llm.invoke([*history, *appended_messages])
        appended_messages.append(current_response)

    return {
        "execution_history": appended_messages,
        "messages": [current_response],
        "phase": "EXECUTION",
        "tool_call_count": tool_call_count,
    }


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
        "messages": [response],
        "current_topic_summary": _message_text(summary_response),
        "phase": "CHAT",
    }
