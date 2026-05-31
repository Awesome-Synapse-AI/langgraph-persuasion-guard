import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

RoleName = Literal["router", "sanitizer", "executor", "chat"]
ROLE_DEFAULT_TEMPERATURES: dict[RoleName, float] = {
    "router": 0.0,
    "sanitizer": 0.2,
    "executor": 0.0,
    "chat": 0.7,
}


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


def _read_env(name: str, *, use_env: bool) -> str | None:
    if not use_env:
        return None
    return os.getenv(name)


def _optional_int_env(name: str, *, use_env: bool) -> int | None:
    value = _read_env(name, use_env=use_env)
    return int(value) if value else None


def _build_role_config(role: RoleName, *, default_model: str | None, default_provider: str | None, use_env: bool) -> RoleModelConfig:
    role_prefix = role.upper()
    role_model = _read_env(f"{role_prefix}_MODEL_NAME", use_env=use_env) or default_model
    if not role_model:
        raise ValueError(
            f"Unable to resolve model for role '{role}'. Provide `default_model`, "
            f"`role_overrides['{role}']`, or set MODEL_NAME/{role_prefix}_MODEL_NAME."
        )

    return RoleModelConfig(
        model=role_model,
        model_provider=(
            _read_env(f"{role_prefix}_MODEL_PROVIDER", use_env=use_env)
            or default_provider
        ),
        temperature=float(
            _read_env(f"{role_prefix}_TEMPERATURE", use_env=use_env)
            or str(ROLE_DEFAULT_TEMPERATURES[role])
        ),
        max_tokens=_optional_int_env(f"{role_prefix}_MAX_TOKENS", use_env=use_env),
    )


def build_models(
    *,
    default_model: str | None = None,
    default_provider: str | None = None,
    role_overrides: Mapping[RoleName, RoleModelConfig] | None = None,
    use_env: bool = True,
) -> dict[str, BaseChatModel]:
    resolved_default_model = default_model or _read_env("MODEL_NAME", use_env=use_env)
    resolved_default_provider = default_provider or _read_env(
        "MODEL_PROVIDER", use_env=use_env
    )
    overrides = dict(role_overrides or {})

    return {
        role: build_model(
            overrides.get(role)
            or _build_role_config(
                role,
                default_model=resolved_default_model,
                default_provider=resolved_default_provider,
                use_env=use_env,
            )
        )
        for role in ("router", "sanitizer", "executor", "chat")
    }
