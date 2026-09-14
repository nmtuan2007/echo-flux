"""Local STT and Translation provider adapters (EF-009).

Every test uses fake runtimes: faster_whisper, ctranslate2, and transformers are never
imported for real, and no model is downloaded or loaded.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, List, Optional

import pytest

from engine.local_models.catalog import (
    CommercialUse,
    HardwareTier,
    LicenseInfo,
    ModelPack,
    PackArtifact,
    PackCatalog,
    PackKind,
    PackSignature,
    PackSource,
)
from engine.local_models.installer import InstalledPack
from engine.local_models.selector import HardwareProbe, HardwareProfile
from engine.providers.contracts import StreamingSTTProvider, TranslationProvider
from engine.providers.registry import build_default_registry
from engine.providers.stt.local import LocalSTTProvider, LocalSTTSettings
from engine.providers.translation.local import (
    LocalTranslationProvider,
    LocalTranslationSettings,
)
from engine.providers.types import (
    AudioFrame,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    TranslationRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_100MS = b"\x01\x00" * 1600


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeProbe(HardwareProbe):
    def __init__(self) -> None:
        self.calls = 0

    def probe(self) -> HardwareProfile:
        self.calls += 1
        return HardwareProfile(cpu_cores=8, ram_gb=16.0, platform="windows", architecture="x86_64")


@dataclass
class FakeTranscript:
    text: str
    is_final: bool
    language: Optional[str] = None
    stream_id: str = "default"


class FakeWhisper:
    """Records calls and threads; returns a partial per chunk and a final on finalize."""

    def __init__(self) -> None:
        self.config: Any = None
        self.load_calls = 0
        self.unload_calls = 0
        self.fail_load = False
        self.fail_audio = False
        self.finalize_none = False
        self.gate: Optional[threading.Event] = None
        self.device = "cpu"
        self.threads: List[str] = []
        self.calls: List[tuple] = []
        self.buffers: Dict[str, int] = {}
        self.languages = {"mic": "en", "system": "ja"}

    def load_model(self, config: Any) -> None:
        self.load_calls += 1
        self.threads.append(threading.current_thread().name)
        if self.fail_load:
            raise RuntimeError("load failed")
        self.config = config

    def transcribe_stream(self, chunk: bytes, stream_id: str = "default") -> Any:
        self.calls.append(("audio", stream_id))
        self.threads.append(threading.current_thread().name)
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_audio:
            raise RuntimeError("inference failed")
        self.buffers[stream_id] = self.buffers.get(stream_id, 0) + len(chunk)
        text = f"{stream_id} {self.buffers[stream_id]}"
        return FakeTranscript(text, False, self.languages.get(stream_id), stream_id)

    def finalize_current(self, stream_id: str = "default") -> Any:
        self.calls.append(("finalize", stream_id))
        size = self.buffers.pop(stream_id, 0)
        if not size or self.finalize_none:
            return None
        return FakeTranscript(
            f"{stream_id} final {size}", True, self.languages.get(stream_id), stream_id
        )

    def reset_stream(self, stream_id: str = "default") -> None:
        self.calls.append(("reset", stream_id))
        self.buffers.pop(stream_id, None)

    def unload_model(self) -> None:
        self.unload_calls += 1


class Factory:
    def __init__(self, make: Any) -> None:
        self.make = make
        self.created: List[Any] = []

    def __call__(self) -> Any:
        instance = self.make()
        self.created.append(instance)
        return instance


@dataclass
class FakeTranslationResult:
    translated_text: str


class FakeMarian:
    fail_load = False
    fail_translate = False

    def __init__(self) -> None:
        self.config: Optional[dict] = None
        self.unload_calls = 0
        self.threads: List[int] = []

    def load_model(self, config: dict) -> None:
        self.threads.append(threading.get_ident())
        if FakeMarian.fail_load:
            raise RuntimeError("load failed")
        self.config = config

    def translate(self, text: str, source_lang: str, target_lang: str) -> Any:
        self.threads.append(threading.get_ident())
        if FakeMarian.fail_translate:
            raise RuntimeError(f"could not translate {text}")
        return FakeTranslationResult(f"[{source_lang}->{target_lang}] {text}")

    def unload_model(self) -> None:
        self.unload_calls += 1


@pytest.fixture(autouse=True)
def reset_fake_marian():
    FakeMarian.fail_load = False
    FakeMarian.fail_translate = False
    yield


class FakeInstaller:
    def __init__(self, installed: Optional[InstalledPack]) -> None:
        self.installed = installed

    def active(self, pack_id: str) -> Optional[InstalledPack]:
        return self.installed


def frame(stream_id: str, sequence: int = 0, pcm: bytes = FRAME_100MS, **kwargs: Any) -> AudioFrame:
    return AudioFrame(stream_id=stream_id, sequence=sequence, pcm_data=pcm, **kwargs)


def make_stt(backend: Optional[FakeWhisper] = None, **kwargs: Any):
    backend = backend or FakeWhisper()
    factory = Factory(lambda: backend)
    kwargs.setdefault("stream_ids", ("mic", "system"))
    kwargs.setdefault("hardware_probe", FakeProbe())
    provider = LocalSTTProvider(backend_factory=factory, **kwargs)
    return provider, backend, factory


def make_translation(**kwargs: Any):
    factory = Factory(FakeMarian)
    kwargs.setdefault("hardware_probe", FakeProbe())
    kwargs.setdefault("preset_resolver", _presets)
    provider = LocalTranslationProvider(backend_factory=factory, **kwargs)
    return provider, factory


_PRESETS = {
    ("en", "vi"): "preset-en-vi",
    ("vi", "en"): "preset-vi-en",
    ("en", "fr"): "preset-en-fr",
}


def _presets(source: str, target: str) -> Optional[str]:
    return _PRESETS.get((source, target))


async def collect(provider: LocalSTTProvider, count: int, timeout: float = 5.0) -> list:
    events: list = []

    async def run() -> None:
        async for event in provider.receive_events():
            events.append(event)
            if len(events) >= count:
                return

    await asyncio.wait_for(run(), timeout)
    return events


async def drain(provider: LocalSTTProvider, timeout: float = 5.0) -> list:
    """Every remaining event until the provider ends iteration."""
    events: list = []

    async def run() -> None:
        async for event in provider.receive_events():
            events.append(event)

    await asyncio.wait_for(run(), timeout)
    return events


def _signed_pack(kind: PackKind, **kwargs: Any) -> ModelPack:
    return ModelPack(
        pack_id=f"{kind.value}-signed",
        version="1.0.0",
        kind=kind,
        display_name="Signed test pack",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED, reviewed=True),
        tiers=frozenset({HardwareTier.CPU_16GB, HardwareTier.CPU_8GB}),
        artifacts=(PackArtifact("model.bin", "https://example.invalid/model.bin", 10, "0" * 64),),
        signature=PackSignature("test-key", "hmac-sha256", "ab" * 16),
        benchmark_evidence="bench-test",
        **kwargs,
    )


def _legacy_pack(kind: PackKind, **kwargs: Any) -> ModelPack:
    return ModelPack(
        pack_id=f"legacy-{kind.value}",
        version="1.0.0",
        kind=kind,
        display_name="Legacy test pack",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED),
        source=PackSource.LEGACY_RUNTIME,
        tiers=frozenset(HardwareTier),
        compatibility_fallback=True,
        **kwargs,
    )


# ── Contracts and imports ────────────────────────────────────────────────────


def test_wrappers_satisfy_capability_contracts_and_manifest_factories():
    assert issubclass(LocalSTTProvider, StreamingSTTProvider)
    assert issubclass(LocalTranslationProvider, TranslationProvider)
    stt, _, _ = make_stt()
    translation, _ = make_translation()
    assert stt.provider_id == "local" and stt.capability.value == "stt"
    assert translation.capability.value == "translation"

    registry = build_default_registry()
    assert registry.load_factory("stt", "local") is LocalSTTProvider
    assert registry.load_factory("translation", "local") is LocalTranslationProvider


def test_settings_build_from_registry_options():
    registry = build_default_registry()
    stt_options = registry.validate_options("stt", "local", {})
    translation_options = registry.validate_options("translation", "local", {})
    stt = LocalSTTSettings.from_options(stt_options)
    translation = LocalTranslationSettings.from_options(translation_options)
    assert stt.language == "auto"
    assert translation.source_lang == "auto" and translation.target_lang == "vi"


def test_packages_import_without_ml_runtimes_or_other_adapters(tmp_path):
    script = """
import json, sys
import engine.providers.stt as stt_pkg
import engine.providers.translation as translation_pkg
import engine.providers.stt.local
import engine.providers.translation.local
assert stt_pkg.LocalSTTProvider.__name__ == "LocalSTTProvider"
assert translation_pkg.LocalTranslationProvider.__name__ == "LocalTranslationProvider"
print(json.dumps(sorted(sys.modules)))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["ECHOFLUX_DATA_DIR"] = str(tmp_path / "data")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    forbidden_roots = {"faster_whisper", "ctranslate2", "transformers", "torch", "numpy"}
    forbidden_roots |= {"websockets", "azure"}
    assert not {name for name in loaded if name.split(".")[0] in forbidden_roots}
    forbidden_modules = {
        "engine.asr",
        "engine.translation",
        "engine.providers.stt.deepgram",
        "engine.providers.stt.streaming_ws",
        "engine.providers.stt.openai_realtime",
        "engine.providers.translation.openai_compatible",
    }
    assert not loaded & forbidden_modules


def test_invalid_settings_are_rejected_before_any_work():
    for settings in (
        LocalSTTSettings(language="english please"),
        LocalSTTSettings(device="gpu"),
        LocalSTTSettings(compute_type="int4"),
    ):
        with pytest.raises(ProviderRuntimeError) as exc:
            make_stt(settings=settings)
        assert exc.value.error.code is ProviderErrorCode.INVALID_REQUEST
    for translation_settings in (
        LocalTranslationSettings(target_lang="auto"),
        LocalTranslationSettings(source_lang="??"),
    ):
        with pytest.raises(ProviderRuntimeError) as exc:
            make_translation(settings=translation_settings)
        assert exc.value.error.code is ProviderErrorCode.INVALID_REQUEST


# ── Local STT ────────────────────────────────────────────────────────────────


async def test_stt_resolves_default_pack_with_auto_language_off_the_loop():
    provider, backend, _ = make_stt()
    await provider.start()
    try:
        assert backend.config.model_size == "small"
        assert backend.config.language == "auto"
        assert backend.config.model_path is None
        assert provider.resolved_pack.pack_id == "legacy-faster-whisper-small"
        assert provider.resolved_pack.reason == "compatibility_fallback"
        assert backend.threads and backend.threads[0] != threading.current_thread().name
    finally:
        await provider.stop()


async def test_stt_explicit_language_hint_is_passed_through():
    provider, backend, _ = make_stt(settings=LocalSTTSettings(language="en"))
    await provider.start()
    await provider.stop()
    assert backend.config.language == "en"


async def test_stt_streams_keep_independent_state_and_detected_language():
    provider, backend, _ = make_stt()
    await provider.start()
    try:
        await provider.send_audio(frame("mic", 0))
        await provider.send_audio(frame("system", 0))
        await provider.send_audio(frame("mic", 1))
        await provider.finalize_stream("mic")
        events = await collect(provider, 4)

        mic = [e for e in events if e.stream_id == "mic"]
        system = [e for e in events if e.stream_id == "system"]
        assert [(e.is_final, e.revision, e.sequence) for e in mic] == [
            (False, 1, 0),
            (False, 2, 0),
            (True, 3, 0),
        ]
        assert mic[-1].text == "mic final 6400" and mic[-1].is_endpoint
        assert {e.detected_language for e in mic} == {"en"}
        assert mic[-1].provider_event_id == "local:mic:0"
        assert mic[-1].end_time_ms == pytest.approx(200.0)
        assert [(e.revision, e.sequence, e.detected_language) for e in system] == [(1, 0, "ja")]

        # Resetting system abandons its utterance without touching mic.
        await provider.reset_stream("system")
        await provider.send_audio(frame("system", 1))
        await provider.finalize_stream("system")
        await provider.send_audio(frame("mic", 2))
        later = await collect(provider, 3)
        system_later = [e for e in later if e.stream_id == "system"]
        mic_later = [e for e in later if e.stream_id == "mic"]
        assert [(e.is_final, e.sequence, e.revision) for e in system_later] == [
            (False, 1, 1),
            (True, 1, 2),
        ]
        assert [(e.is_final, e.sequence, e.revision) for e in mic_later] == [(False, 1, 1)]
        assert ("reset", "system") in backend.calls and ("reset", "mic") not in backend.calls
        audio_threads = {name for name in backend.threads[1:]}
        assert audio_threads == {"LocalSTT-mic", "LocalSTT-system"}
    finally:
        await provider.stop()


async def test_stt_retracted_partial_closes_with_empty_final():
    provider, backend, _ = make_stt()
    backend.finalize_none = True
    await provider.start()
    try:
        await provider.send_audio(frame("mic"))
        await provider.finalize_stream("mic")
        partial, final = await collect(provider, 2)
        assert not partial.is_final and final.is_final
        assert final.text == "" and final.revision == 2 and final.sequence == 0
    finally:
        await provider.stop()


async def test_stt_stop_finalizes_open_utterance_and_ends_iteration():
    provider, backend, _ = make_stt()
    await provider.start()
    await provider.send_audio(frame("mic"))
    await collect(provider, 1)
    await provider.stop()
    remaining = await drain(provider)
    assert [(e.stream_id, e.is_final, e.text) for e in remaining] == [
        ("mic", True, "mic final 3200")
    ]
    assert backend.unload_calls == 1


async def test_stt_lifecycle_is_idempotent_and_restartable():
    provider, backend, _ = make_stt()
    await provider.stop()
    await provider.start()
    await provider.start()
    assert backend.load_calls == 1
    await provider.stop()
    await provider.stop()
    assert backend.unload_calls == 1
    with pytest.raises(ProviderRuntimeError) as exc:
        await provider.send_audio(frame("mic"))
    assert exc.value.error.code is ProviderErrorCode.INVALID_REQUEST

    await provider.start()
    assert backend.load_calls == 2
    await provider.send_audio(frame("mic"))
    (event,) = await collect(provider, 1)
    assert event.stream_id == "mic"
    await provider.stop()
    assert backend.unload_calls == 2


async def test_stt_load_failure_is_categorized_and_releases_the_backend():
    provider, backend, _ = make_stt()
    backend.fail_load = True
    with pytest.raises(ProviderRuntimeError) as exc:
        await provider.start()
    assert exc.value.error.code is ProviderErrorCode.MODEL_UNAVAILABLE
    assert "load failed" not in str(exc.value)
    assert backend.unload_calls == 1
    with pytest.raises(ProviderRuntimeError):
        await provider.send_audio(frame("mic"))


async def test_stt_inference_failure_surfaces_from_receive_events():
    provider, backend, _ = make_stt()
    backend.fail_audio = True
    await provider.start()
    await provider.send_audio(frame("mic"))
    with pytest.raises(ProviderRuntimeError) as exc:
        await collect(provider, 1)
    assert exc.value.error.code is ProviderErrorCode.INTERNAL_ERROR
    with pytest.raises(ProviderRuntimeError) as again:
        await provider.send_audio(frame("system"))
    assert again.value.error.code is ProviderErrorCode.INTERNAL_ERROR
    await provider.stop()
    assert backend.unload_calls == 1


async def test_stt_rejects_bad_audio_and_unknown_streams():
    provider, _, _ = make_stt()
    await provider.start()
    try:
        for bad in (
            frame("mic", sample_rate=8000),
            frame("mic", channels=2),
            frame("mic", pcm=b"\x00"),
            frame("unknown"),
        ):
            with pytest.raises(ProviderRuntimeError) as exc:
                await provider.send_audio(bad)
            assert exc.value.error.code is ProviderErrorCode.INVALID_REQUEST
    finally:
        await provider.stop()


async def test_stt_drops_frames_instead_of_blocking_when_behind():
    backend = FakeWhisper()
    backend.gate = threading.Event()
    provider, _, _ = make_stt(backend, max_pending_frames=2)
    await provider.start()
    try:
        for sequence in range(10):
            await provider.send_audio(frame("mic", sequence))
        assert provider.dropped_frames("mic") >= 6
        assert provider.dropped_frames("system") == 0
    finally:
        backend.gate.set()
        await provider.stop()


async def test_stt_health_check_never_loads_a_model():
    provider, _, factory = make_stt(runtime_available=lambda: True)
    healthy = await provider.health_check()
    assert healthy.status is ProviderHealthStatus.HEALTHY
    assert healthy.details["loaded"] is False
    assert healthy.details["pack_id"] == "legacy-faster-whisper-small"

    missing, _, missing_factory = make_stt(runtime_available=lambda: False)
    unhealthy = await missing.health_check()
    assert unhealthy.status is ProviderHealthStatus.UNHEALTHY
    assert unhealthy.details["code"] == "model_unavailable"
    assert not factory.created and not missing_factory.created


async def test_stt_unsupported_language_hint_is_categorized():
    catalog = PackCatalog([_legacy_pack(PackKind.STT, languages=frozenset({"en"}))])
    provider, backend, _ = make_stt(
        settings=LocalSTTSettings(language="fr"), catalog=catalog, runtime_available=lambda: True
    )
    with pytest.raises(ProviderRuntimeError) as exc:
        await provider.start()
    assert exc.value.error.code is ProviderErrorCode.UNSUPPORTED_LANGUAGE
    assert backend.load_calls == 0
    health = await provider.health_check()
    assert health.details["code"] == "unsupported_language"


async def test_stt_catalog_pack_must_be_installed(tmp_path):
    catalog = PackCatalog([_signed_pack(PackKind.STT, multilingual=True)])
    missing, backend, _ = make_stt(
        catalog=catalog, installer_factory=lambda c, p: FakeInstaller(None)
    )
    with pytest.raises(ProviderRuntimeError) as exc:
        await missing.start()
    assert exc.value.error.code is ProviderErrorCode.MODEL_UNAVAILABLE
    assert backend.load_calls == 0

    installed = InstalledPack("stt-signed", "1.0.0", tmp_path / "stt-signed")
    provider, backend, _ = make_stt(
        catalog=catalog, installer_factory=lambda c, p: FakeInstaller(installed)
    )
    await provider.start()
    await provider.stop()
    assert backend.config.model_path == str(tmp_path / "stt-signed")
    assert provider.resolved_pack.reason == "recommended"


# ── Local Translation ────────────────────────────────────────────────────────


def _request(text: str = "hello", source: str = "auto", target: str = "vi", **kw: Any):
    return TranslationRequest(text=text, source_lang=source, target_lang=target, **kw)


async def test_translation_explicit_pair_preloads_resolved_preset_off_the_loop():
    provider, factory = make_translation(
        settings=LocalTranslationSettings(source_lang="en", target_lang="vi")
    )
    await provider.start()
    try:
        (backend,) = factory.created
        assert backend.config == {
            "translation.source_lang": "en",
            "translation.target_lang": "vi",
            "translation.model": "preset-en-vi",
            "device": "auto",
        }
        result = await provider.translate(
            _request("hello", "en", "vi", stream_id="mic", sequence=4, revision=2)
        )
        assert result.translated_text == "[en->vi] hello"
        assert (result.source_lang, result.target_lang) == ("en", "vi")
        assert result.detected_source_lang is None
        assert (result.stream_id, result.sequence, result.revision) == ("mic", 4, 2)
        assert result.latency_ms is not None
        assert threading.get_ident() not in backend.threads
        assert len(factory.created) == 1
    finally:
        await provider.stop()


async def test_translation_auto_source_resolves_deterministically():
    provider, factory = make_translation()
    await provider.start()
    try:
        assert not factory.created  # auto source loads nothing until a request arrives
        result = await provider.translate(_request("hello"))
        assert result.source_lang == "en" and result.detected_source_lang == "en"
        again = await provider.translate(_request("again"))
        assert again.source_lang == "en" and len(factory.created) == 1

        reverse = await provider.translate(_request("xin chao", "vi", "en"))
        assert reverse.translated_text == "[vi->en] xin chao"
    finally:
        await provider.stop()


async def test_translation_ambiguous_auto_source_requires_a_hint():
    pairs = frozenset({("en", "fr"), ("de", "fr")})
    catalog = PackCatalog([_legacy_pack(PackKind.TRANSLATION, language_pairs=pairs)])
    provider, _ = make_translation(catalog=catalog)
    await provider.start()
    try:
        with pytest.raises(ProviderRuntimeError) as exc:
            await provider.translate(_request(target="fr"))
        assert exc.value.error.code is ProviderErrorCode.UNSUPPORTED_LANGUAGE
        assert exc.value.error.details["reason"] == "source_language_required"
        result = await provider.translate(_request("hi", "en", "fr"))
        assert result.translated_text == "[en->fr] hi"
        with pytest.raises(ProviderRuntimeError) as no_preset:
            await provider.translate(_request("hallo", "de", "fr"))
        assert no_preset.value.error.details["reason"] == "no_preset_model"
    finally:
        await provider.stop()


async def test_unsupported_pair_is_safe_and_does_not_stop_stt():
    stt, _, _ = make_stt()
    translation, factory = make_translation()
    await stt.start()
    await translation.start()
    try:
        with pytest.raises(ProviderRuntimeError) as exc:
            await translation.translate(_request("secret words", "fr", "vi"))
        error = exc.value.error
        assert error.code is ProviderErrorCode.UNSUPPORTED_LANGUAGE
        assert error.capability.value == "translation" and not error.retryable
        assert "secret words" not in str(exc.value)
        assert not factory.created

        await stt.send_audio(frame("mic"))
        (event,) = await collect(stt, 1)
        assert event.text == "mic 3200"
        assert (await translation.translate(_request("hello", "en", "vi"))).translated_text
    finally:
        await translation.stop()
        await stt.stop()


async def test_translation_model_load_failure_is_categorized_and_cached():
    FakeMarian.fail_load = True
    provider, factory = make_translation()
    await provider.start()
    try:
        for _ in range(2):
            with pytest.raises(ProviderRuntimeError) as exc:
                await provider.translate(_request("hello", "en", "vi"))
            assert exc.value.error.code is ProviderErrorCode.MODEL_UNAVAILABLE
        assert len(factory.created) == 1
        assert factory.created[0].unload_calls == 1
    finally:
        await provider.stop()


async def test_translation_inference_failure_is_retryable_and_not_cached():
    provider, factory = make_translation()
    await provider.start()
    try:
        FakeMarian.fail_translate = True
        with pytest.raises(ProviderRuntimeError) as exc:
            await provider.translate(_request("private text", "en", "vi"))
        assert exc.value.error.code is ProviderErrorCode.INTERNAL_ERROR
        assert exc.value.error.retryable
        assert "private text" not in str(exc.value)
        FakeMarian.fail_translate = False
        result = await provider.translate(_request("private text", "en", "vi"))
        assert result.translated_text == "[en->vi] private text"
        assert len(factory.created) == 1
    finally:
        await provider.stop()


async def test_translation_skips_models_for_empty_or_same_language_text():
    provider, factory = make_translation()
    await provider.start()
    try:
        empty = await provider.translate(_request("   ", "en", "vi"))
        assert empty.translated_text == ""
        same = await provider.translate(_request("xin chao", "vi", "vi"))
        assert same.translated_text == "xin chao"
        assert not factory.created
    finally:
        await provider.stop()


async def test_translation_evicts_least_recent_pair_and_stop_releases_everything():
    provider, factory = make_translation(max_loaded_pairs=1)
    await provider.start()
    await provider.translate(_request("hello", "en", "vi"))
    await provider.translate(_request("xin chao", "vi", "en"))
    first, second = factory.created
    assert first.unload_calls == 1 and second.unload_calls == 0

    await provider.stop()
    await provider.stop()
    assert second.unload_calls == 1
    with pytest.raises(ProviderRuntimeError) as exc:
        await provider.translate(_request("hello", "en", "vi"))
    assert exc.value.error.code is ProviderErrorCode.INVALID_REQUEST

    await provider.start()
    await provider.translate(_request("hello", "en", "vi"))
    assert len(factory.created) == 3
    await provider.stop()


async def test_translation_start_tolerates_unsupported_configured_pair():
    provider, factory = make_translation(
        settings=LocalTranslationSettings(source_lang="fr", target_lang="vi")
    )
    await provider.start()
    try:
        with pytest.raises(ProviderRuntimeError) as exc:
            await provider.translate(_request("bonjour"))
        assert exc.value.error.code is ProviderErrorCode.UNSUPPORTED_LANGUAGE
        assert not factory.created
    finally:
        await provider.stop()


async def test_translation_catalog_pack_without_runtime_is_model_unavailable():
    pairs = frozenset({("en", "vi")})
    catalog = PackCatalog([_signed_pack(PackKind.TRANSLATION, language_pairs=pairs)])
    provider, factory = make_translation(catalog=catalog)
    await provider.start()
    try:
        with pytest.raises(ProviderRuntimeError) as exc:
            await provider.translate(_request("hello", "en", "vi"))
        assert exc.value.error.code is ProviderErrorCode.MODEL_UNAVAILABLE
        assert not factory.created
    finally:
        await provider.stop()


async def test_translation_health_check_never_loads_a_model():
    provider, factory = make_translation(
        settings=LocalTranslationSettings(source_lang="en", target_lang="vi"),
        runtime_available=lambda: True,
    )
    healthy = await provider.health_check()
    assert healthy.status is ProviderHealthStatus.HEALTHY
    assert healthy.details["pack_id"] == "legacy-marian-opus-mt"

    missing, missing_factory = make_translation(runtime_available=lambda: False)
    unhealthy = await missing.health_check()
    assert unhealthy.status is ProviderHealthStatus.UNHEALTHY
    assert unhealthy.details["code"] == "model_unavailable"
    assert not factory.created and not missing_factory.created


def test_default_preset_resolver_uses_marian_presets_only():
    from engine.providers.translation.local import _default_preset_resolver

    assert _default_preset_resolver("en", "vi") == "Helsinki-NLP/opus-mt-en-vi"
    assert _default_preset_resolver("fr", "vi") is None


# ── Legacy backend compatibility ─────────────────────────────────────────────


class _FakeWhisperModel:
    fail_cuda = False
    transcribe_kwargs: ClassVar[List[dict]] = []

    def __init__(self, path: str, device: str, **kwargs: Any) -> None:
        if device == "cuda" and _FakeWhisperModel.fail_cuda:
            raise RuntimeError("cublas64_12.dll missing")
        self.device = device

    def transcribe(self, audio: Any, **kwargs: Any):
        _FakeWhisperModel.transcribe_kwargs.append(kwargs)
        return iter(()), SimpleNamespace(language="en")


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    _FakeWhisperModel.fail_cuda = False
    _FakeWhisperModel.transcribe_kwargs = []
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_FakeWhisperModel)
    )
    return _FakeWhisperModel


@pytest.mark.parametrize("language, expected", [("auto", None), ("", None), ("en", "en")])
def test_faster_whisper_maps_auto_language_to_detection(fake_faster_whisper, language, expected):
    import numpy as np

    from engine.asr.faster_whisper_backend import FasterWhisperBackend
    from engine.core.config import TranscriptionConfig

    backend = FasterWhisperBackend()
    backend.load_model(TranscriptionConfig("small", language, "cpu", "int8"))
    backend._get_stream("mic")["audio_buffer"] = np.zeros(8000, dtype=np.float32)
    assert backend.finalize_current("mic") is None
    assert fake_faster_whisper.transcribe_kwargs[-1]["language"] == expected


def test_faster_whisper_reports_device_after_fallback_and_unloads_idempotently(
    fake_faster_whisper,
):
    from engine.asr.faster_whisper_backend import FasterWhisperBackend
    from engine.core.config import TranscriptionConfig

    backend = FasterWhisperBackend()
    assert backend.device is None
    backend.load_model(TranscriptionConfig("small", "auto", "cuda", "float16"))
    assert backend.device == "cuda"

    fake_faster_whisper.fail_cuda = True
    fallback = FasterWhisperBackend()
    fallback.load_model(TranscriptionConfig("small", "auto", "auto", "float16"))
    assert fallback.device == "cpu" and fallback.is_loaded

    fallback.unload_model()
    fallback.unload_model()
    assert fallback.device is None and not fallback.is_loaded


def test_marian_resolves_presets_without_guessing():
    from engine.translation.marian_backend import MarianBackend

    assert MarianBackend.resolve_preset_model("en", "vi") == "Helsinki-NLP/opus-mt-en-vi"
    assert MarianBackend.resolve_preset_model("fr", "vi") is None
    assert MarianBackend.resolve_preset_model("auto", "vi") is None


def test_marian_lifecycle_is_idempotent_and_reload_releases_previous(monkeypatch):
    from engine.translation.marian_backend import MarianBackend

    class FakeTranslator:
        def __init__(self, path: str, device: str, compute_type: str) -> None:
            self.device = device

        def translate_batch(self, batch: Any, **kwargs: Any) -> list:
            return []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name: str) -> "FakeTokenizer":
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 0, Translator=FakeTranslator),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(MarianTokenizer=FakeTokenizer))
    monkeypatch.setattr(MarianBackend, "_is_valid_ct2_model", lambda self, path: True)

    backend = MarianBackend()
    backend.unload_model()
    backend.unload_model()
    assert not backend.is_loaded

    unloads: List[bool] = []
    original_unload = backend.unload_model

    def counting_unload() -> None:
        unloads.append(True)
        original_unload()

    monkeypatch.setattr(backend, "unload_model", counting_unload)
    config = {"translation.source_lang": "en", "translation.target_lang": "vi"}
    config["translation.model"] = "Helsinki-NLP/opus-mt-en-vi"
    backend.load_model(config)
    assert backend.is_loaded and not unloads
    backend.load_model(config)
    assert backend.is_loaded and len(unloads) == 1
    backend.unload_model()
    assert not backend.is_loaded
