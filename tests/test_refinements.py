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
