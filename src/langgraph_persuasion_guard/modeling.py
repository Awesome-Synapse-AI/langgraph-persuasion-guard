import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class RoleModelConfig:
    model: str
    model_provider: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


def build_model(cfg: RoleModelConfig) -> BaseChatModel:
    kwargs: dict[str, object] = {"temperature": cfg.temperature}
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    if cfg.model_provider:
        kwargs["model_provider"] = cfg.model_provider
    return init_chat_model(cfg.model, **kwargs)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable {name}. Set it to a model id such as "
            "'openai:gpt-4o-mini' or pair MODEL_PROVIDER with an unprefixed MODEL_NAME."
        )
    return value


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _build_role_config(
    role_prefix: str,
    *,
    default_model: str,
    default_provider: str | None,
    default_temperature: float,
) -> RoleModelConfig:
    return RoleModelConfig(
        model=os.getenv(f"{role_prefix}_MODEL_NAME", default_model),
        model_provider=os.getenv(f"{role_prefix}_MODEL_PROVIDER", default_provider),
        temperature=float(os.getenv(f"{role_prefix}_TEMPERATURE", str(default_temperature))),
        max_tokens=_optional_int_env(f"{role_prefix}_MAX_TOKENS"),
    )


def build_models() -> dict[str, BaseChatModel]:
    default_model = _require_env("MODEL_NAME")
    default_provider = os.getenv("MODEL_PROVIDER")

    return {
        "router": build_model(
            _build_role_config(
                "ROUTER",
                default_model=default_model,
                default_provider=default_provider,
                default_temperature=0.0,
            )
        ),
        "sanitizer": build_model(
            _build_role_config(
                "SANITIZER",
                default_model=default_model,
                default_provider=default_provider,
                default_temperature=0.2,
            )
        ),
        "executor": build_model(
            _build_role_config(
                "EXECUTOR",
                default_model=default_model,
                default_provider=default_provider,
                default_temperature=0.0,
            )
        ),
        "chat": build_model(
            _build_role_config(
                "CHAT",
                default_model=default_model,
                default_provider=default_provider,
                default_temperature=0.7,
            )
        ),
    }
