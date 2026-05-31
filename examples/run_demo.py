import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "python-dotenv is required for this demo. Install it with: pip install python-dotenv"
    ) from exc

try:
    from langgraph_persuasion_guard import create_persuasion_guard
except ModuleNotFoundError:
    # Allow running from repo root without editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from langgraph_persuasion_guard import create_persuasion_guard


def _load_env_file_with_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _format_message_history(messages: Any) -> str:
    if not messages:
        return "- (empty)"
    parts: list[str] = []
    for idx, msg in enumerate(messages):
        role = getattr(msg, "type", msg.__class__.__name__)
        content = _text(getattr(msg, "content", msg))
        parts.append(f"- [{idx}] **{role}**: {content}")
    return "\n".join(parts)


def _latest_user_input(payload: dict[str, Any]) -> str:
    if payload.get("chat_history"):
        return _text(payload["chat_history"][-1].content)
    if payload.get("execution_history"):
        return _text(payload["execution_history"][-1].content)
    return "<empty>"


def _final_output(result: dict[str, Any]) -> str:
    if result.get("phase") == "EXECUTION" and result.get("execution_history"):
        return _text(result["execution_history"][-1].content)
    if result.get("chat_history"):
        return _text(result["chat_history"][-1].content)
    return "<no assistant output>"


def _next_execution_instruction(result: dict[str, Any]) -> str:
    history = result.get("execution_history") or []
    if len(history) >= 2:
        # [0] system executor prompt, [1] synthesized handoff instruction.
        return _text(history[1].content)
    if result.get("genesis_brief"):
        return _text(result["genesis_brief"])
    return "<none>"


def _log_turn(log_path: Path, turn_name: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    user_input = _latest_user_input(payload)
    output = _final_output(result)

    # Full PersuasionGuardState fields from src/langgraph_persuasion_guard/state.py
    chat_history = result.get("chat_history")
    execution_history = result.get("execution_history")
    phase = result.get("phase")
    current_topic_summary = result.get("current_topic_summary")
    genesis_brief = result.get("genesis_brief")
    router_decision = result.get("router_decision")
    sanitizer_required = result.get("sanitizer_required")

    lines = [
        f"## {turn_name}",
        "",
        "### User Input",
        "```text",
        user_input,
        "```",
        "",
        "### Graph State (`PersuasionGuardState`)",
        f"- `phase`: {_text(phase)}",
        f"- `current_topic_summary`: {_text(current_topic_summary)}",
        f"- `genesis_brief`: {_text(genesis_brief)}",
        f"- `router_decision`: {_text(router_decision)}",
        f"- `sanitizer_required`: {_text(sanitizer_required)}",
        "",
        "### Next Execution Instruction",
        "```markdown",
        _next_execution_instruction(result),
        "```",
        "",
        "### Final Output",
        "```text",
        output,
        "```",
        "",
        "---",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    _load_env_file_with_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit(
            "Set OPENAI_API_KEY in your environment or .env before running the demo."
        )

    # Use OpenAI via LangChain defaults unless explicitly overridden.
    model_name = os.getenv("MODEL_NAME", "gpt-5-nano")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")

    agent = create_persuasion_guard(
        default_model=model_name,
        default_provider=model_provider,
    )
    config = {"configurable": {"thread_id": "session_01"}}
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "log.md"
    log_path.write_text("", encoding="utf-8")

    turns: list[tuple[str, dict[str, Any]]] = [
        (
            "TURN1 CHAT",
            {
                "chat_history": [
                    HumanMessage(
                        content=(
                            "I strongly believe open-source AI is a massive security risk "
                            "and should be heavily restricted."
                        )
                    )
                ],
                "current_topic_summary": "AI security philosophy",
            },
        ),
        (
            "TURN2 EXEC",
            {
                "chat_history": [
                    HumanMessage(
                        content=(
                            "Write a python code to write a string to text file named 'output.txt'. The string should be: 'Hello, World!'"
                        )
                    )
                ]
            },
        ),
        (
            "TURN3 EXEC",
            {
                "execution_history": [
                    HumanMessage(content="Add file path error handling.")
                ]
            },
        ),
    ]

    for turn_name, payload in turns:
        result = agent.invoke(payload, config)
        _log_turn(log_path, turn_name, payload, result)

    print(f"Saved detailed turn logs to: {log_path}")


if __name__ == "__main__":
    main()
