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


class PersuasionGuardState(TypedDict, total=False):
    chat_history: Annotated[list[BaseMessage], add_messages]
    execution_history: Annotated[list[BaseMessage], add_messages]
    phase: Literal["CHAT", "EXECUTION"]
    current_topic_summary: str
    genesis_brief: str | None
    router_decision: RouterDecision | None
