"""Assistant provider adapters.

Exports resolve lazily so importing this package never loads adapter modules or HTTP
clients; the registry imports the adapter module only when a session needs it.
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "AssistantPromptPolicy": "engine.providers.assistant.openai_compatible",
    "OpenAICompatibleAssistantProvider": "engine.providers.assistant.openai_compatible",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
