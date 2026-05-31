import os

from langchain_core.messages import HumanMessage

from langgraph_persuasion_guard.graph import create_persuasion_guard


def main() -> None:
    if not os.getenv("MODEL_NAME"):
        raise SystemExit(
            "Set MODEL_NAME before running the demo. Example: "
            "MODEL_NAME=openai:gpt-4o-mini"
        )

    agent = create_persuasion_guard()
    config = {"configurable": {"thread_id": "session_01"}}

    result = agent.invoke(
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
        config,
    )
    print(f"TURN1 CHAT: {result['chat_history'][-1].content}\n")

    result = agent.invoke(
        {
            "chat_history": [
                HumanMessage(
                    content="Write a deployment script for an open-source LLM on AWS."
                )
            ]
        },
        config,
    )
    print(f"TURN2 EXEC: {result['execution_history'][-1].content}\n")

    result = agent.invoke(
        {
            "execution_history": [
                HumanMessage(content="Add error handling and IAM role validation.")
            ]
        },
        config,
    )
    print(f"TURN3 EXEC: {result['execution_history'][-1].content}")


if __name__ == "__main__":
    main()
