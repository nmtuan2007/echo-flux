"""Realtime session orchestration.

Exports resolve lazily (PEP 562), so importing one session module, such as the ring buffer,
never pulls in the orchestrator, the provider registry, or any adapter.
"""

import importlib
from typing import Any

_ORCHESTRATOR = "engine.session.session_orchestrator"

__all__ = [
    "STTSessionConfig",
    "STTSessionError",
    "STTSessionOrchestrator",
    "STTStatusEvent",
    "STTStatusKind",
    "STTTranscriptEvent",
    "legacy_stt_service",
    "resolve_session_config",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_ORCHESTRATOR), name)
    globals()[name] = value
    return value
