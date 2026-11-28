"""Unit tests for Phase 2 & 3 backend improvements."""

import asyncio
import json
import sys
import types
from pathlib import Path
from queue import Queue
from typing import Any, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from engine.asr.faster_whisper_backend import FasterWhisperBackend
from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import ConversationRecord
from engine.conversation.repository import ConversationRepository
from engine.core.config import TranscriptionConfig
from engine.local_models.selector import GpuInfo, SystemHardwareProbe
from engine.main import EchoFluxEngine
from engine.providers.manifests import BUILTIN_MANIFESTS
from engine.providers.stt.local import LocalSTTProvider, LocalSTTSettings
from engine.providers.types import ProviderHealthCheck, ProviderHealthStatus
from engine.server.control_dispatcher import ControlCommandDispatcher
from engine.server.control_helpers import probe_hardware_with_gpu
from engine.session.session_orchestrator import resolve_session_config


def test_probe_hardware_with_gpu_caches_per_instance(monkeypatch):
    calls: List[Any] = []

    def fake_check_output(cmd, timeout=None, text=None):
        calls.append(cmd)
        return "8192\n"

    fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    probe1 = SystemHardwareProbe()
    p1_first = probe_hardware_with_gpu(probe1)
    p1_second = probe_hardware_with_gpu(probe1)
    assert isinstance(p1_first.gpu, GpuInfo)
    assert p1_first.gpu.vram_gb == 8.0
    assert p1_second.gpu == p1_first.gpu
    assert len(calls) == 1

    # Fresh SystemHardwareProbe() instances remain isolated.
    probe2 = SystemHardwareProbe()
    p2_first = probe_hardware_with_gpu(probe2)
    assert p2_first.gpu == p1_first.gpu
    assert len(calls) == 2

    # ControlCommandDispatcher reuses its internal SystemHardwareProbe across polls.
    dispatcher = ControlCommandDispatcher()
    d_probe_a = dispatcher._get_probe()
    d_probe_b = dispatcher._get_probe()
    assert d_probe_a is d_probe_b


def test_result_queue_bounded_and_drops_audio_level_under_backpressure(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOFLUX_DATA_DIR", str(tmp_path))
    repo = ConversationRepository(tmp_path / "q.sqlite3")
    try:
        engine = EchoFluxEngine(insecure_dev=True, conversation_repo=repo)
        assert isinstance(engine._result_queue, Queue)
        assert engine._result_queue.maxsize == 1000

        for i in range(801):
            engine._enqueue_result({"type": "final", "text": f"m{i}"})
        assert engine._result_queue.qsize() == 801

        # Audio level frames are dropped when queue size exceeds 800.
        engine._enqueue_result({"type": "audio_level", "source": "mic", "level": 0.4})
        assert engine._result_queue.qsize() == 801

        # Fill up to capacity and verify oldest item is evicted on Full.
        for i in range(801, 1000):
            engine._enqueue_result({"type": "final", "text": f"m{i}"})
        assert engine._result_queue.qsize() == 1000

        engine._enqueue_result({"type": "final", "text": "newest"})
        assert engine._result_queue.qsize() == 1000
        first = engine._result_queue.get_nowait()
        assert first["text"] == "m1"
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_translation_configure_hot_toggles_live_translation(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOFLUX_DATA_DIR", str(tmp_path))
    repo = ConversationRepository(tmp_path / "trans.sqlite3")
    try:
        engine = EchoFluxEngine(insecure_dev=True, conversation_repo=repo)
        engine._running = True
        engine._current_config = {
            "services.translation": {
                "enabled": False,
                "providerId": "local",
                "options": {"source_lang": "en", "target_lang": "vi"},
            }
        }

        created = []

        class FakeLocalTransProvider:
            def __init__(self, resolved):
                self.provider_id = resolved.provider_id
                self.target_lang = resolved.options.get("target_lang")
                self.started = False
                self.stopped = False
                created.append(self)

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

            async def health_check(self):
                return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

            async def translate(self, req):
                return types.SimpleNamespace(translated_text=req.text, detected_source_lang="en")

        engine._translation_local_builder = FakeLocalTransProvider

        sent = []

        class FakeWS:
            async def send(self, raw):
                sent.append(json.loads(raw))

        ws = FakeWS()

        # 1. Hot-enable translation via dispatcher with target_lang="ja".
        await engine.control_dispatcher.dispatch(
            "req-1",
            "translation_configure",
            {"enabled": True, "target_lang": "ja"},
            ws,
            None,
        )
        res1 = sent[0]["data"]
        assert res1["ok"] is True
        assert res1["enabled"] is True
        assert res1["target_lang"] == "ja"
        assert res1["active"] is True
        assert engine._translation_session is not None
        assert len(created) == 1
        assert created[0].started is True
        assert created[0].target_lang == "ja"

        # 2. Hot-disable translation stops the active session.
        sent.clear()
        await engine.control_dispatcher.dispatch(
            "req-2",
            "translation_configure",
            {"enabled": False},
            ws,
            None,
        )
        res2 = sent[0]["data"]
        assert res2["ok"] is True
        assert res2["enabled"] is False
        assert res2["active"] is False
        assert engine._translation_session is None
        assert created[0].stopped is True
    finally:
        if engine._translation_session is not None:
            await engine._translation_session.stop()
        repo.close()


def test_context_vocabulary_extraction_and_stt_injection(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOFLUX_DATA_DIR", str(tmp_path))
    repo = ConversationRepository(tmp_path / "vocab.sqlite3")
    try:
        profiles = ContextProfileService(repo)
        repo.create_conversation(
            ConversationRecord(
                conversation_id="conv-vocab",
                title="Vocab Test",
                created_at_ms=1000.0,
                updated_at_ms=1000.0,
            )
        )
        profiles.create_profile(
            "Tech Glossary",
            (
                "# Glossary\n"
                "- EchoFlux: realtime captioning app\n"
                "- Kubernetes = container orchestrator\n"
                "- FastAPI - async Python framework\n"
                "PyTorch, CTranslate2, EchoFlux\n"
            ),
            profile_id="prof-vocab",
        )
        profiles.set_attached_profiles("conv-vocab", ["prof-vocab"])

        engine = EchoFluxEngine(insecure_dev=True, conversation_repo=repo)
        engine._active_conversation_id = "conv-vocab"

        terms = engine._extract_context_vocabulary({})
        assert terms == ["EchoFlux", "Kubernetes", "FastAPI", "PyTorch", "CTranslate2"]

        # Verify injection for local STT without overwriting explicit user option.
        local_cfg = resolve_session_config({}, ("mic",))
        injected_local = engine._inject_context_vocabulary(local_cfg, {})
        assert injected_local.primary.options["initial_prompt"] == (
            "EchoFlux, Kubernetes, FastAPI, PyTorch, CTranslate2"
        )

        # Verify injection for cloud providers (Deepgram, OpenAI Realtime).
        dg_cfg = resolve_session_config(
            {
                "services.stt": {
                    "enabled": True,
                    "providerId": "deepgram",
                    "credentialRefs": {"api_key": {"configId": "dg", "slot": "api_key"}},
                }
            },
            ("mic",),
        )
        injected_dg = engine._inject_context_vocabulary(dg_cfg, {})
        assert injected_dg.primary.options["keyterms"] == (
            "EchoFlux, Kubernetes, FastAPI, PyTorch, CTranslate2"
        )
        assert injected_dg.fallback.options["initial_prompt"] == (
            "EchoFlux, Kubernetes, FastAPI, PyTorch, CTranslate2"
        )

        # LocalSTTSettings -> TranscriptionConfig -> FasterWhisperBackend.transcribe
        stt_settings = LocalSTTSettings.from_options(dict(injected_local.primary.options))
        assert stt_settings.initial_prompt == "EchoFlux, Kubernetes, FastAPI, PyTorch, CTranslate2"

        seen_kwargs = {}

        class FakeWhisper:
            def transcribe(self, _audio, **kwargs):
                seen_kwargs.update(kwargs)
                return [], types.SimpleNamespace(language="en", language_probability=0.99)

        backend = FasterWhisperBackend()
        backend._config = TranscriptionConfig(
            model_size="small",
            language="en",
            device="cpu",
            compute_type="int8",
            initial_prompt=stt_settings.initial_prompt,
        )
        backend._model = FakeWhisper()
        stream = backend._get_stream("mic")
        stream["pending_chunks"].append(np.ones(1600, dtype=np.float32))
        stream["pending_samples"] = 1600
        backend._run_inference("mic", stream, is_final=True)
        assert seen_kwargs["initial_prompt"] == "EchoFlux, Kubernetes, FastAPI, PyTorch, CTranslate2"

        # Unused LocalSTTProvider import check (verify constructor accepts settings cleanly).
        provider = LocalSTTProvider(stt_settings, hardware_probe=SystemHardwareProbe())
        assert provider._settings.initial_prompt == stt_settings.initial_prompt
    finally:
        repo.close()


def test_local_stt_quality_preset_not_advanced():
    local_manifest = next(m for m in BUILTIN_MANIFESTS if m.provider_id == "local")
    field = local_manifest.get_field("quality_preset")
    assert field is not None
    assert field.advanced is False


def test_control_bridge_allows_translation_configure():
    src = (
        Path(__file__).resolve().parent.parent
        / "apps"
        / "desktop"
        / "src-tauri"
        / "src"
        / "control_bridge.rs"
    ).read_text(encoding="utf-8")
    assert '"translation_configure" => (None, 15, false),' in src
    assert asyncio.iscoroutinefunction(EchoFluxEngine.configure_translation_live)
    assert isinstance(MagicMock(), MagicMock)
