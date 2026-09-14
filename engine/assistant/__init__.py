"""Manual per-message Assist: context selection, prompt policy, and the application service.

Exports resolve lazily so importing this package never loads provider adapters, HTTP
clients, or the conversation store.
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "ASSIST_RESPONSE_JSON_SCHEMA": "engine.assistant.prompt_builder",
    "AssistContextWindow": "engine.assistant.context_window",
    "AssistError": "engine.assistant.service",
    "AssistErrorCode": "engine.assistant.service",
    "AssistOutcome": "engine.assistant.service",
    "AssistPromptPolicy": "engine.assistant.prompt_builder",
    "AssistResponseSchemaError": "engine.assistant.prompt_builder",
    "AssistantService": "engine.assistant.service",
    "ContextBudget": "engine.assistant.context_window",
    "ContextSelectionError": "engine.assistant.context_window",
    "select_assist_context": "engine.assistant.context_window",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
