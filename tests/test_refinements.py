"""Targeted unit tests for Phase 1-3 refinements (VAD, summary streaming/cancel, pack install lock, GPU probe)."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from engine.audio.vad import VAD, WINDOW_SIZE_SAMPLES
from engine.local_models.selector import (
    GpuInfo,
    HardwareProbe,
    HardwareProfile,
    SystemHardwareProbe,
)
from engine.main import EchoFluxEngine
from engine.server.control_dispatcher import ControlCommandDispatcher, DispatcherError
from engine.server.control_helpers import probe_hardware_with_gpu


class _FakeSession:
    def __init__(self, probs):
        self._probs = list(probs)

    def run(self, _outputs, _inputs):
        prob = self._probs.pop(0) if self._probs else 0.0
        return np.array([[prob]], dtype=np.float32), np.zeros((2, 1, 64)), np.zeros((2, 1, 64))


def test_vad_preserves_speech_state_across_subwindow_chunks(monkeypatch):
    monkeypatch.setattr(VAD, "_init_model", lambda self: None)
    vad = VAD({"enabled": True, "threshold": 0.5, "sample_rate": 16000})
    vad._session = _FakeSession([0.9, 0.9])

    # 320 samples (20ms at 16kHz): first chunk buffers, second reaches 640 >= 512 -> speech=True
    chunk_20ms = np.zeros(320, dtype=np.int16).tobytes()
    assert vad.process(chunk_20ms) is False
    assert vad.process(chunk_20ms) is True

    # Next 320-sample chunk leaves buffer at 128 + 320 = 448 < 512: must stay True (no flicker)
    assert len(vad._buffer) < WINDOW_SIZE_SAMPLES
    assert vad.process(chunk_20ms) is True


@pytest.mark.asyncio
async def test_pack_install_rejects_concurrent_install(tmp_path):
    dispatcher = ControlCommandDispatcher()
    dispatcher._pack_install_active = "legacy-faster-whisper-small"
    with pytest.raises(DispatcherError) as exc_info:
        await dispatcher._handle_pack_install({"pack_id": "legacy-marian-opus-mt"})
    assert exc_info.value.code == "install_in_progress"


@pytest.mark.asyncio
async def test_summary_cancel_by_conversation_id_cancels_streaming_task():
    dispatcher = ControlCommandDispatcher()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_summary():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(slow_summary())
    dispatcher._summary_tasks_by_conv["conv-123"] = task
    await started.wait()

    res = await dispatcher._handle_summary_cancel({"conversation_id": "conv-123"})
    assert res["cancelled"] is True
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_engine_run_summary_broadcasts_chunks_when_authenticated(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOFLUX_DATA_DIR", str(tmp_path))
    engine = EchoFluxEngine.__new__(EchoFluxEngine)
    engine._ws_server = MagicMock()
    engine._ws_server.is_authenticated = True
    engine._ws_server.broadcast = AsyncMock()

    ws = MagicMock()
    ws.send = AsyncMock()

    async def fake_request_summary(cid, on_chunk=None):
        await on_chunk("Hello ")
        await on_chunk("world")
        stored = MagicMock(summary_id="sum-1")
        return MagicMock(
            ok=True,
            warning=None,
            stored=stored,
            provider=None,
            included_message_count=2,
            omitted_message_count=0,
        )

    service = MagicMock()
    service.request_summary = fake_request_summary
    cmd = MagicMock(conversation_id="conv-bc")

    await engine._run_summary(service, cmd, ws)
    assert engine._ws_server.broadcast.await_count == 2
    first_chunk = engine._ws_server.broadcast.await_args_list[0].args[0]
    assert first_chunk == {"type": "llm_chunk", "text": "Hello ", "conversation_id": "conv-bc"}


def test_probe_hardware_with_gpu_respects_fake_and_enriches_system(monkeypatch):
    class FakeProbe(HardwareProbe):
        def probe(self) -> HardwareProfile:
            return HardwareProfile(4, 8.0, "windows", "x86_64", gpu=None)

    assert probe_hardware_with_gpu(FakeProbe()).gpu is None

    fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    enriched = probe_hardware_with_gpu(SystemHardwareProbe())
    assert isinstance(enriched.gpu, GpuInfo)
    assert enriched.gpu.backend == "cuda"


def test_microphone_resolves_shifted_device_index_by_name():
    from engine.audio.microphone import MicrophoneInput

    class FakePA:
        def __init__(self, devices):
            self._devices = devices

        def get_device_count(self):
            return len(self._devices)

        def get_device_info_by_index(self, idx):
            return self._devices[idx]

    mic = MicrophoneInput({}, device_id=0, device_name="USB Yeti")
    pa = FakePA(
        [
            {"name": "Built-in Mic", "maxInputChannels": 1},
            {"name": "USB Yeti Stereo", "maxInputChannels": 2},
        ]
    )
    assert mic._resolve_device_index(pa) == 1


def test_local_stt_quality_preset_maps_to_model_size():
    from engine.local_models import PackKind, builtin_catalog
    from engine.local_models.selector import HardwareTier
    from engine.providers.stt.local import LocalSTTSettings, resolve_stt_pack, validate_settings

    validate_settings(LocalSTTSettings(quality_preset="fast"))
    catalog = builtin_catalog()
    profile = HardwareProfile(4, 16.0, "windows", "x86_64")
    profile_probe = type("P", (HardwareProbe,), {"probe": lambda self: profile})()
    resolved_fast = resolve_stt_pack(catalog, profile_probe, "en", quality_preset="fast")
    resolved_acc = resolve_stt_pack(catalog, profile_probe, "en", quality_preset="accurate")
    assert resolved_fast.model_size == "tiny"
    assert resolved_fast.pack_id == "legacy-faster-whisper-tiny"
    assert resolved_acc.model_size == "medium"
    assert resolved_acc.pack_id == "legacy-faster-whisper-medium"
    assert HardwareTier.CPU_16GB is not None and PackKind.STT is not None


@pytest.mark.asyncio
async def test_retranslate_message_post_stop_uses_local_provider_and_persists(monkeypatch):
    import engine.providers.translation.local as local_mod

    class FakeLocalTrans:
        def __init__(self, settings, provider_id="local"):
            self.provider_id = provider_id

        async def start(self):
            return None

        async def translate(self, req):
            return types.SimpleNamespace(translated_text=f"VI:{req.text}")

        async def stop(self):
            return None

    monkeypatch.setattr(local_mod, "LocalTranslationProvider", FakeLocalTrans)

    engine = EchoFluxEngine.__new__(EchoFluxEngine)
    engine._translation_session = None
    engine._conversation_repo = MagicMock()
    engine._last_translation_config = types.SimpleNamespace(
        enabled=True,
        source_lang="en",
        target_lang="vi",
        fallback=types.SimpleNamespace(provider_id="local", options={"source_lang": "en", "target_lang": "vi"}),
        primary=None,
    )

    out = await engine.retranslate_message("conv-1", "msg-1", "Hello team")
    assert out == "VI:Hello team"
    engine._conversation_repo.set_translation.assert_called_once()


def test_faster_whisper_lazy_buffer_and_concurrent_streams():
    import threading
    import time

    from engine.asr.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend()
    backend._config = types.SimpleNamespace(language="en")
    backend._max_buffer_duration = 4.0

    class SlowFakeModel:
        def __init__(self):
            self.active = 0
            self.overlap = False

        def transcribe(self, audio, **_kwargs):
            self.active += 1
            if self.active > 1:
                self.overlap = True
            time.sleep(0.02)
            self.active -= 1
            seg = types.SimpleNamespace(
                text=f"n={len(audio)}",
                start=0.0,
                end=0.5,
                no_speech_prob=0.0,
                avg_logprob=-0.1,
                compression_ratio=1.0,
            )
            return [seg], types.SimpleNamespace(language="en", language_probability=0.99)

    fake_model = SlowFakeModel()
    backend._model = fake_model

    s1 = backend._get_stream("mic")
    s2 = backend._get_stream("system")
    assert s1 is not s2
    chunk = np.ones(1600, dtype=np.int16).tobytes()
    backend.transcribe_stream(chunk, "mic")
    backend.transcribe_stream(chunk, "mic")
    assert backend._buffer_samples(s1) == 3200
    assert len(s1["pending_chunks"]) == 2
    materialized = backend._materialize_buffer(s1)
    assert len(materialized) == 3200
    assert len(s1["audio_buffer"]) == 3200
    assert len(s1["pending_chunks"]) == 0

    s2["pending_chunks"].append(np.ones(3200, dtype=np.float32))
    s2["pending_samples"] = 3200

    errors = []

    def run_once(sid, stream):
        try:
            res = backend._run_inference(sid, stream, is_final=True)
            assert res is not None and res.text == "n=3200" and res.language == "en"
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=run_once, args=("mic", s1))
    t2 = threading.Thread(target=run_once, args=("system", s2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    assert fake_model.overlap is False


def test_local_stt_honors_legacy_model_size():
    from engine.local_models import builtin_catalog
    from engine.providers.stt.local import resolve_stt_pack

    catalog = builtin_catalog()
    profile = HardwareProfile(8, 32.0, "windows", "x86_64")
    probe = type("P", (HardwareProbe,), {"probe": lambda self: profile})()
    resolved = resolve_stt_pack(catalog, probe, "en", legacy_model_size="tiny")
    assert resolved.model_size == "tiny"
    assert resolved.pack_id == "legacy-faster-whisper-tiny"


def test_repository_ask_turns_v2_roundtrip(tmp_path):
    from engine.conversation.models import ConversationRecord, StoredAskTurn
    from engine.conversation.repository import ConversationRepository

    db_path = tmp_path / "conv.db"
    repo = ConversationRepository(db_path=str(db_path))
    conv_id = "conv-ask-1"
    repo.create_conversation(
        ConversationRecord(
            conversation_id=conv_id,
            title="Ask Test",
            created_at_ms=1700000000000,
            updated_at_ms=1700000000000,
        )
    )
    repo.save_ask_turn(
        conversation_id=conv_id,
        question="What are the action items?",
        answer_markdown="1. Deploy v2.",
        ask_id="ask-1",
        created_at_ms=1700000001000,
    )
    listed = repo.list_ask_turns(conv_id)
    assert len(listed) == 1
    assert listed[0].question == "What are the action items?"
    assert listed[0].answer_markdown == "1. Deploy v2."

    exported = repo.export_conversation(conv_id)
    assert len(exported["ask_turns"]) == 1

    repo.clear_ask_turns(conv_id)
    assert repo.list_ask_turns(conv_id) == []

    imported_cid = "conv-ask-imported"
    imported = repo.import_conversation(
        ConversationRecord(
            conversation_id=imported_cid,
            title="Imported Ask",
            created_at_ms=1700000000000,
            updated_at_ms=1700000000000,
            import_id="imp-ask-1",
        ),
        messages=(),
        ask_turns=[
            StoredAskTurn(
                ask_id="ask-imp-1",
                conversation_id=imported_cid,
                question=exported["ask_turns"][0]["question"],
                answer_markdown=exported["ask_turns"][0]["answer_markdown"],
                created_at_ms=exported["ask_turns"][0]["created_at_ms"],
            )
        ],
    )
    imported_turns = repo.list_ask_turns(imported.conversation_id)
    assert len(imported_turns) == 1
    assert imported_turns[0].answer_markdown == "1. Deploy v2."
    repo.close()


@pytest.mark.asyncio
async def test_dispatcher_pack_delete_and_ask_clear(monkeypatch, tmp_path):
    import json

    import engine.server.handlers_packs as handlers_packs

    monkeypatch.setattr(
        handlers_packs,
        "delete_local_model_files",
        lambda *_args, **_kwargs: 1,
    )
    sent = []

    class FakeWS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    repo = MagicMock()
    repo.clear_ask_turns.return_value = 2
    dispatcher = ControlCommandDispatcher(conversation_repo=repo)
    ws = FakeWS()

    await dispatcher.dispatch(
        "req-del",
        "pack_delete",
        {"capability": "stt", "pack_id": "legacy-faster-whisper-small"},
        ws,
        None,
    )
    assert sent[0]["op"] == "response"
    assert sent[0]["data"]["deleted"] is True
    assert sent[0]["data"]["removed_count"] == 1

    sent.clear()
    await dispatcher.dispatch(
        "req-clr",
        "assistant_ask_clear",
        {"conversation_id": "conv-1"},
        ws,
        None,
    )
    assert sent[0]["op"] == "response"
    assert "cleared" in sent[0]["data"]
    repo.clear_ask_turns.assert_called_once_with("conv-1")
