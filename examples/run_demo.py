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
    router_decision = result.get("router_decision")
    instruction = _next_execution_instruction(result)
    output = _final_output(result)
    user_input = _latest_user_input(payload)
    phase = result.get("phase", "<unknown>")

    lines = [
        f"=== {turn_name} ===",
        f"USER INPUT: {user_input}",
        "GRAPH STATE:",
        f"- phase: {phase}",
        f"- persuasion/task decision: {router_decision}",
        f"- next execution instruction: {instruction}",
        f"FINAL OUTPUT: {output}",
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
    log_path = repo_root / "log.txt"
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
                            "Write a python code to run ollama model locally and execute a simple query."
                        )
                    )
                ]
            },
        ),
        (
            "TURN3 EXEC",
            {
                "execution_history": [
                    HumanMessage(content="Add import error handling.")
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
