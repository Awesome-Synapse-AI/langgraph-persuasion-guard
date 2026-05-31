from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .modeling import RoleModelConfig, build_models
from .nodes import chat_node, executor_node, router_node, sanitizer_node
from .state import PersuasionGuardState

ModelMap = Mapping[str, BaseChatModel | Any]


def route_from_start(state: PersuasionGuardState) -> Literal["router_node", "executor_node"]:
    execution_history = state.get("execution_history") or []
    # Continue directly in execution mode only when this turn provides
    # a human follow-up in the execution timeline.
    if (
        state.get("phase") == "EXECUTION"
        and execution_history
        and isinstance(execution_history[-1], HumanMessage)
    ):
        return "executor_node"
    return "router_node"


def route_after_router(state: PersuasionGuardState) -> Literal["sanitizer_node", "chat_node"]:
    if state.get("phase") == "EXECUTION":
        return "sanitizer_node"
    return "chat_node"


def build_persuasion_guard_graph(
    models: ModelMap | None = None,
    *,
    default_model: str | None = None,
    default_provider: str | None = None,
    role_model_overrides: Mapping[
        Literal["router", "sanitizer", "executor", "chat"], RoleModelConfig
    ]
    | None = None,
    use_env: bool = True,
    checkpointer: Any | None = None,
):
    if models is not None:
        active_models = dict(models)
    else:
        active_models = dict(
            build_models(
                default_model=default_model,
                default_provider=default_provider,
                role_overrides=role_model_overrides,
                use_env=use_env,
            )
        )
    required = {"router", "sanitizer", "executor", "chat"}
    missing = required.difference(active_models)
    if missing:
        raise ValueError(f"Missing required models: {', '.join(sorted(missing))}")

    graph = StateGraph(PersuasionGuardState)
    graph.add_node("router_node", lambda state: router_node(state, active_models["router"]))
    graph.add_node(
        "sanitizer_node",
        lambda state: sanitizer_node(state, active_models["sanitizer"]),
    )
    graph.add_node(
        "executor_node",
        lambda state: executor_node(state, active_models["executor"]),
    )
    graph.add_node("chat_node", lambda state: chat_node(state, active_models["chat"]))

    graph.add_conditional_edges(START, route_from_start)
    graph.add_conditional_edges("router_node", route_after_router)
    graph.add_edge("sanitizer_node", "executor_node")
    graph.add_edge("executor_node", END)
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def create_persuasion_guard(
    models: ModelMap | None = None,
    *,
    default_model: str | None = None,
    default_provider: str | None = None,
    role_model_overrides: Mapping[
        Literal["router", "sanitizer", "executor", "chat"], RoleModelConfig
    ]
    | None = None,
    use_env: bool = True,
    checkpointer: Any | None = None,
):
    return build_persuasion_guard_graph(
        models=models,
        default_model=default_model,
        default_provider=default_provider,
        role_model_overrides=role_model_overrides,
        use_env=use_env,
        checkpointer=checkpointer,
    )
