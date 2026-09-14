"""Streaming STT provider adapters.

Nothing is imported eagerly: loading one adapter must never pull in another adapter, a
network client, or an ML runtime. The local adapter is reachable lazily by name; every other
adapter is imported from its own module.
"""

import importlib
from typing import Any, Dict

_LAZY_EXPORTS: Dict[str, str] = {
    "LocalSTTProvider": "engine.providers.stt.local",
    "LocalSTTSettings": "engine.providers.stt.local",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
