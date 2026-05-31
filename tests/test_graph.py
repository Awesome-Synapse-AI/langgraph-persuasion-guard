import json
import sys
import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langgraph_persuasion_guard.graph import create_persuasion_guard
from langgraph_persuasion_guard.nodes import invoke_router_with_fallback
from langgraph_persuasion_guard.state import RouterDecision


class StructuredRouterModel:
    def __init__(self, *decisions: RouterDecision):
        self._decisions = list(decisions)
        self.calls = 0
        self.structured_calls = 0

    def with_structured_output(self, _schema):
        self.structured_calls += 1
        return self

    def invoke(self, _messages):
        self.calls += 1
        return self._decisions[self.calls - 1]


class JsonRouterModel:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls = 0

    def with_structured_output(self, _schema):
        raise NotImplementedError("structured output unavailable")

    def invoke(self, _messages):
        response = AIMessage(content=self._responses[self.calls])
        self.calls += 1
        return response


class PersuasionGuardGraphTests(unittest.TestCase):
    def test_chat_path_updates_chat_history_and_summary(self):
        router = StructuredRouterModel(
            RouterDecision(is_task=False, confidence=0.9, reasoning="casual")
        )
        chat = FakeListChatModel(
            responses=["General chat reply", "AI security philosophy summary."]
        )
        guard = create_persuasion_guard(
            models={
                "router": router,
                "sanitizer": FakeListChatModel(responses=["unused"]),
                "executor": FakeListChatModel(responses=["unused"]),
                "chat": chat,
            }
        )

        result = guard.invoke(
            {
                "chat_history": [HumanMessage(content="What do you think about AI risk?")],
                "current_topic_summary": "AI risk",
            },
            {"configurable": {"thread_id": "chat-path"}},
        )

        self.assertEqual(result["phase"], "CHAT")
        self.assertEqual(result["chat_history"][-1].content, "General chat reply")
        self.assertEqual(
            result["current_topic_summary"], "AI security philosophy summary."
        )
        self.assertEqual(router.calls, 1)

    def test_task_request_flows_into_sanitizer_then_executor(self):
        router = StructuredRouterModel(
            RouterDecision(is_task=True, confidence=0.95, reasoning="task")
        )
        guard = create_persuasion_guard(
            models={
                "router": router,
                "sanitizer": FakeListChatModel(
                    responses=["# Task Objective\nWrite the deployment script."]
                ),
                "executor": FakeListChatModel(responses=["Here is the deployment script."]),
                "chat": FakeListChatModel(responses=["unused", "unused"]),
            }
        )

        result = guard.invoke(
            {
                "chat_history": [
                    HumanMessage(content="Write a deployment script for an LLM on AWS.")
                ],
                "current_topic_summary": "LLM deployment",
            },
            {"configurable": {"thread_id": "task-path"}},
        )

        self.assertEqual(result["phase"], "EXECUTION")
        self.assertIn("Task Objective", result["genesis_brief"])
        self.assertEqual(result["execution_history"][-1].content, "Here is the deployment script.")

    def test_execution_follow_up_bypasses_router_on_later_turns(self):
        router = StructuredRouterModel(
            RouterDecision(is_task=True, confidence=0.98, reasoning="task")
        )
        guard = create_persuasion_guard(
            models={
                "router": router,
                "sanitizer": FakeListChatModel(
                    responses=["# Task Objective\nWrite the deployment script."]
                ),
                "executor": FakeListChatModel(
                    responses=[
                        "Initial executor response.",
                        "Updated executor response with IAM validation.",
                    ]
                ),
                "chat": FakeListChatModel(responses=["unused", "unused"]),
            }
        )
        config = {"configurable": {"thread_id": "follow-up"}}

        guard.invoke(
            {
                "chat_history": [HumanMessage(content="Write a deployment script.")],
                "current_topic_summary": "Deployment",
            },
            config,
        )
        result = guard.invoke(
            {
                "execution_history": [
                    HumanMessage(content="Add error handling and IAM role validation.")
                ]
            },
            config,
        )

        self.assertEqual(
            result["execution_history"][-1].content,
            "Updated executor response with IAM validation.",
        )
        self.assertEqual(router.calls, 1)

    def test_chat_turn_after_execution_reenters_router(self):
        router = StructuredRouterModel(
            RouterDecision(is_task=True, confidence=0.98, reasoning="task"),
            RouterDecision(is_task=False, confidence=0.9, reasoning="back to chat"),
        )
        guard = create_persuasion_guard(
            models={
                "router": router,
                "sanitizer": FakeListChatModel(
                    responses=["# Task Objective\nWrite the deployment script."]
                ),
                "executor": FakeListChatModel(responses=["Initial executor response."]),
                "chat": FakeListChatModel(
                    responses=["Sure, let's chat about unrelated topics.", "chat summary"]
                ),
            }
        )
        config = {"configurable": {"thread_id": "chat-after-exec"}}

        guard.invoke(
            {
                "chat_history": [HumanMessage(content="Write a deployment script.")],
                "current_topic_summary": "Deployment",
            },
            config,
        )
        result = guard.invoke(
            {
                "chat_history": [
                    HumanMessage(content="Actually, what do you think about Thai food?")
                ]
            },
            config,
        )

        self.assertEqual(result["phase"], "CHAT")
        self.assertEqual(
            result["chat_history"][-1].content, "Sure, let's chat about unrelated topics."
        )
        self.assertEqual(router.calls, 2)

    def test_router_fallback_parses_json_when_structured_output_is_unavailable(self):
        model = JsonRouterModel(
            json.dumps(
                {
                    "is_task": True,
                    "confidence": 0.75,
                    "reasoning": "contains imperative request",
                }
            )
        )

        decision = invoke_router_with_fallback(
            model, [HumanMessage(content="Write the script.")]
        )

        self.assertEqual(
            decision,
            RouterDecision(
                is_task=True,
                confidence=0.75,
                reasoning="contains imperative request",
            ),
        )


if __name__ == "__main__":
    unittest.main()
