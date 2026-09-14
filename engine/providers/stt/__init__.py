"""Streaming STT provider adapters.

Nothing is imported eagerly: loading one adapter must never pull in another adapter, a
network client, an optional SDK, or an ML runtime. Every name below resolves lazily to its
own module on first attribute access; the registry resolves adapters by module path instead.
"""

import importlib
from typing import Any, Dict, List

_LAZY_EXPORTS: Dict[str, str] = {
    "LocalSTTProvider": "engine.providers.stt.local",
    "LocalSTTSettings": "engine.providers.stt.local",
    "OpenAIRealtimeSTTProvider": "engine.providers.stt.openai_realtime",
    "OpenAIRealtimeSettings": "engine.providers.stt.openai_protocol",
    "GeminiLiveSTTProvider": "engine.providers.stt.gemini_live",
    "GeminiLiveSTTConfig": "engine.providers.stt.gemini_live",
    "DeepgramSTTProvider": "engine.providers.stt.deepgram",
    "DeepgramSettings": "engine.providers.stt.deepgram",
    "AzureSpeechSTTProvider": "engine.providers.stt.azure_speech",
    "AzureSpeechSTTConfig": "engine.providers.stt.azure_speech",
    "AssemblyAISTTProvider": "engine.providers.stt.assemblyai",
    "AssemblyAISettings": "engine.providers.stt.assemblyai",
    "CustomSTTProvider": "engine.providers.stt.custom",
    "CustomSTTSettings": "engine.providers.stt.custom",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
