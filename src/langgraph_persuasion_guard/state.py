from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    is_task: bool = Field(
        description="True if the user is initiating or continuing a downstream task."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief explanation of the classification.")


class SanitizerGateDecision(BaseModel):
    requires_sanitizer: bool = Field(
        description="True if the execution turn should be re-sanitized before executor."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief explanation of the decision.")


class PersuasionGuardState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_history: Annotated[list[BaseMessage], add_messages]
    execution_history: Annotated[list[BaseMessage], add_messages]
    phase: Literal["CHAT", "EXECUTION"]
    current_topic_summary: str
    genesis_brief: str | None
    router_decision: RouterDecision | None
    sanitizer_required: bool | None
    tool_call_count: int | None
    last_ingested_message_id: str | None
