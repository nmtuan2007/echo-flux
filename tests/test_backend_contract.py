"""Pins .agent/skills/add-engine-backend/SKILL.md against the real ABCs.

The skill file tells an agent which methods to implement when adding an ASR or
translation backend. When it drifts from the ABC, the agent writes a class that cannot
be instantiated -- and nothing catches it until runtime. This compares the documented
signatures to ``__abstractmethods__`` directly, so drift fails here instead.

Heavy ML imports (torch, faster_whisper, transformers) live inside ``load_model()``, not
at module scope, so the real backend classes import cheaply and need no skip guards.
"""

import inspect
import re
from pathlib import Path

import pytest

from engine.asr.base import ASRBackend
from engine.translation.base import TranslationBackend

SKILL_FILE = (
    Path(__file__).resolve().parent.parent / ".agent" / "skills" / "add-engine-backend" / "SKILL.md"
)

# Each python fence in the skill documents one backend type, in this order.
_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)


def _documented_methods() -> list:
    """Return the method-name set from each python fence in the skill, in order."""
    text = SKILL_FILE.read_text(encoding="utf-8")
    fences = _FENCE_RE.findall(text)
    return [set(_DEF_RE.findall(fence)) for fence in fences]


@pytest.fixture(scope="module")
def documented():
    blocks = _documented_methods()
    assert len(blocks) >= 2, (
        f"{SKILL_FILE.name} should contain a python fence for ASR and one for "
        f"Translation; found {len(blocks)}."
    )
    return blocks


def test_asr_abstract_methods_match_the_skill(documented):
    actual = set(ASRBackend.__abstractmethods__)
    assert documented[0] == actual, (
        "add-engine-backend/SKILL.md is out of sync with ASRBackend.\n"
        f"  documented but not abstract: {sorted(documented[0] - actual)}\n"
        f"  abstract but undocumented:   {sorted(actual - documented[0])}"
    )


def test_translation_abstract_methods_match_the_skill(documented):
    actual = set(TranslationBackend.__abstractmethods__)
    assert documented[1] == actual, (
        "add-engine-backend/SKILL.md is out of sync with TranslationBackend.\n"
        f"  documented but not abstract: {sorted(documented[1] - actual)}\n"
        f"  abstract but undocumented:   {sorted(actual - documented[1])}"
    )


def test_asr_stream_methods_accept_a_stream_id():
    # The engine runs 'mic' and 'system' concurrently; per-stream state is mandatory.
    for name in ("transcribe_stream", "finalize_current", "reset_stream"):
        params = inspect.signature(getattr(ASRBackend, name)).parameters
        assert "stream_id" in params, f"ASRBackend.{name} must accept stream_id"


def test_translate_is_concrete_and_translate_raw_is_abstract():
    # Backends override translate_raw; translate() adds post-processing around it.
    assert "translate_raw" in TranslationBackend.__abstractmethods__
    assert "translate" not in TranslationBackend.__abstractmethods__


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("engine.asr.faster_whisper_backend", "FasterWhisperBackend"),
        ("engine.translation.marian_backend", "MarianBackend"),
        ("engine.translation.online_backend", "OnlineBackend"),
        ("engine.translation.fallback_backend", "FallbackTranslationBackend"),
    ],
)
def test_shipped_backends_satisfy_their_abc(module_path, class_name):
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    missing = {
        name
        for name in getattr(cls.__mro__[-2], "__abstractmethods__", set())
        if getattr(cls, name, None) is None
    }
    assert not missing, f"{class_name} is missing {sorted(missing)}"

    # Instantiation is the real proof: Python raises TypeError on an unimplemented
    # abstract member. Constructors must not load models.
    instance = cls()
    assert instance.is_loaded is False


def test_result_dataclasses_expose_the_documented_fields():
    from engine.asr.base import TranscriptResult
    from engine.translation.base import TranslationResult

    result = TranscriptResult(text="hi", is_final=True)
    assert result.stream_id == "default"
    assert result.confidence == 0.0
    assert result.language is None

    translation = TranslationResult(
        source_text="hi", translated_text="chao", source_lang="en", target_lang="vi"
    )
    assert translation.confidence is None
