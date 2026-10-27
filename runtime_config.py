"""Local runtime policy for provider access and optional LangSmith tracing.

This module only reads configuration and changes tracing-related environment
variables. It never creates a provider or LangSmith client and never reads a
secret value into logs or returned configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_TRACE_VARS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    offline: bool
    tracing_requested: bool
    tracing_enabled: bool
    langsmith_project: str


def read_runtime_config() -> RuntimeConfig:
    offline = _truthy(os.getenv("TISW_OFFLINE"))
    tracing_requested = any(_truthy(os.getenv(name)) for name in _TRACE_VARS)
    project = (
        os.getenv("LANGSMITH_PROJECT", "").strip()
        or os.getenv("LANGCHAIN_PROJECT", "").strip()
        or "tisw"
    )
    return RuntimeConfig(
        offline=offline,
        tracing_requested=tracing_requested,
        tracing_enabled=tracing_requested and not offline,
        langsmith_project=project,
    )


def configure_runtime(*, offline: bool | None = None) -> RuntimeConfig:
    """Apply the local provider/tracing policy and return safe status flags.

    ``offline=True`` is authoritative and disables both current and legacy
    LangSmith tracing environment names. When offline is false, tracing still
    requires an explicit true value in one of the supported variables.
    """
    if offline is not None:
        os.environ["TISW_OFFLINE"] = "true" if offline else "false"

    config = read_runtime_config()
    enabled = config.tracing_enabled
    tracing_value = "true" if enabled else "false"
    for name in _TRACE_VARS:
        os.environ[name] = tracing_value

    if enabled:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", config.langsmith_project)

    return RuntimeConfig(
        offline=config.offline,
        tracing_requested=config.tracing_requested,
        tracing_enabled=enabled,
        langsmith_project=config.langsmith_project,
    )


def is_offline_mode() -> bool:
    return read_runtime_config().offline
