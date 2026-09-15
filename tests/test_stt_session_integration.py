"""Realtime STT session orchestration with fake providers and fake audio inputs.

No audio hardware, models, network, or real secrets: every provider is ``FakeSTT`` and every
audio device is ``FakeInput``. Covers session config resolution (v2 and the legacy adapter),
the Local path on both audio sources, cloud failure with reconnect and bounded Local replay,
atomic Start, exactly-once Stop, and the engine wiring into capture and broadcast.
"""

import asyncio
import dataclasses
import json
import threading
import time
from typing import Any, Callable, List, Optional

import pytest

from engine.providers.contracts import StreamingSTTProvider
from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)
from engine.session.session_orchestrator import (
    ERROR_LOCAL_FALLBACK_UNAVAILABLE,
    ERROR_MISSING_CREDENTIAL,
    STTSessionError,
    STTSessionOrchestrator,
    STTStatusEvent,
    STTStatusKind,
    STTTranscriptEvent,
    resolve_session_config,
)
from engine.session.stt_failover import FailoverPolicy, FailoverState

BYTES_PER_SECOND = 16000 * 2
CHUNK_20MS = 640
# Placeholder credential value; nothing here talks to a real provider.
FAKE_KEY = "fake-key-for-tests"

CLOUD_START = {
    "services.stt": {
        "enabled": True,
        "providerId": "openai-realtime",
        "credentialRefs": {"api_key": {"configId": "openai-main", "slot": "api_key"}},
    }
}

_END = object()


def _error(code: ProviderErrorCode, retryable: bool = False) -> ProviderError:
    return ProviderError(code, ProviderCapability.STT, "fake failure", retryable=retryable)


class FakeSTT(StreamingSTTProvider):
    """Scriptable streaming STT adapter. Tests push results and failures by hand."""

    def __init__(
        self,
        provider_id: str,
        stream_ids=("mic",),
        *,
        start_error: Optional[BaseException] = None,
        start_hang: bool = False,
        health: ProviderHealthStatus = ProviderHealthStatus.HEALTHY,
        on_audio: Optional[Callable[["FakeSTT", Any], None]] = None,
    ) -> None:
        self._provider_id = provider_id
        self.stream_ids = tuple(stream_ids)
        self.start_error = start_error
        self.start_hang = start_hang
        self.health = health
        self.on_audio = on_audio
        self.start_calls = 0
        self.stop_calls = 0
        self.health_calls = 0
        self.started = False
        self.audio = {sid: bytearray() for sid in self.stream_ids}
        self.finalized: List[str] = []
        self.emitted: set = set()
        self._events: asyncio.Queue = asyncio.Queue()

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_hang:
            await asyncio.sleep(3600)
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False
        self._events.put_nowait(_END)

    async def health_check(self) -> ProviderHealthCheck:
        self.health_calls += 1
        return ProviderHealthCheck(status=self.health)

    async def send_audio(self, frame) -> None:
        if not self.started:
            raise ProviderRuntimeError(_error(ProviderErrorCode.INVALID_REQUEST))
        self.audio.setdefault(frame.stream_id, bytearray()).extend(frame.pcm_data)
        if self.on_audio is not None:
            self.on_audio(self, frame)

    async def receive_events(self):
        while True:
            item = await self._events.get()
            if item is _END:
                return
            if isinstance(item, ProviderError):
                raise ProviderRuntimeError(item)
            yield item

    async def finalize_stream(self, stream_id: str) -> None:
        self.finalized.append(stream_id)

    async def reset_stream(self, stream_id: str) -> None:
        return None

    def emit(
        self,
        stream_id: str,
        event_id: str,
        text: str,
        start_ms: float,
        end_ms: float,
        final: bool = True,
        revision: int = 1,
        language: Optional[str] = None,
    ) -> None:
        self._events.put_nowait(
            STTTranscriptResult(
                stream_id=stream_id,
                sequence=0,
                provider_event_id=event_id,
                text=text,
                is_final=final,
                revision=revision,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                monotonic_timestamp=0.0,
                detected_language=language,
            )
        )

    def fail(self, code=ProviderErrorCode.CONNECTION_CLOSED, retryable: bool = True) -> None:
        self._events.put_nowait(_error(code, retryable))


def _second(value: int) -> bytes:
    return bytes([value % 256]) * BYTES_PER_SECOND


async def wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.005)


def _finals(events, stream_id: Optional[str] = None) -> List[str]:
    return [
        e.text
        for e in events
        if isinstance(e, STTTranscriptEvent)
        and e.is_final
        and e.text
        and (stream_id is None or e.stream_id == stream_id)
    ]


def _statuses(events) -> List[STTStatusEvent]:
    return [e for e in events if isinstance(e, STTStatusEvent)]


def _secrets(resolved) -> dict:
    return {"api_key": FAKE_KEY}


def _cloud_session(events, clouds, local, stream_ids=("mic",), policy=None, **builder_kwargs):
    config = resolve_session_config(CLOUD_START, stream_ids, policy=policy)

    def builder(resolved, secrets, ids):
        assert secrets == {"api_key": FAKE_KEY}
        options = builder_kwargs.get("per_attempt", {}).get(len(clouds), {})
        provider = FakeSTT(resolved.provider_id, ids, **options)
        clouds.append(provider)
        return provider

    return STTSessionOrchestrator(
        config,
        events.append,
        cloud_builder=builder,
        local_builder=lambda cfg: local,
        secrets_resolver=_secrets,
        poll_interval_s=0.005,
    )


async def _speak_cloud_transcript(session, cloud, events) -> None:
    """Eight seconds of audio and the EF-015 overlap transcript on the cloud connection."""
    for second in range(8):
        assert session.submit_audio("mic", _second(second))
    await wait_until(lambda: len(cloud.audio["mic"]) == 8 * BYTES_PER_SECOND)
    cloud.emit("mic", "c1", "Good morning everyone.", 0, 2000)
    cloud.emit("mic", "c2", "We should schedule the meeting", 2500, 6000)
    cloud.emit("mic", "c3", "for tomorrow", 6200, 8000, final=False)
    await wait_until(
        lambda: (
            len(_finals(events)) == 2
            and any(isinstance(e, STTTranscriptEvent) and not e.is_final for e in events)
        )
    )


# ── Session config ───────────────────────────────────────────────────────────


def test_legacy_flat_start_config_resolves_once_to_an_immutable_local_session():
    config = resolve_session_config(
        {"asr.model_size": "tiny", "asr.language": "vi", "asr.device": "cpu"},
        ("mic", "system"),
        compute_type="int8",
    )

    assert not config.cloud_primary
    assert config.primary.provider_id == config.fallback.provider_id == "local"
    assert dict(config.primary.options) == {
        "language": "vi",
        "device": "cpu",
        "compute_type": "int8",
    }
    assert config.legacy_model_size == "tiny"
    assert config.stream_ids == ("mic", "system")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.primary = config.fallback
    with pytest.raises(TypeError):
        config.primary.options["device"] = "cuda"


def test_v2_stt_section_resolves_cloud_primary_with_local_fallback():
    settings = {
        "services.stt": {**CLOUD_START["services.stt"], "options": {"language": "de"}},
        "asr.model_size": "tiny",
    }

    config = resolve_session_config(settings, ("mic",))

    assert config.cloud_primary
    assert config.primary.provider_id == "openai-realtime"
    assert config.primary.endpoint.startswith("wss://")
    assert config.primary.model
    assert set(config.primary.credential_refs) == {"api_key"}
    assert config.fallback.provider_id == "local"
    assert config.fallback.options["language"] == "de"
    # The legacy Whisper size only applies to legacy flat configs.
    assert config.legacy_model_size is None


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        ({"services.stt": {"enabled": True, "providerId": "nope"}}, "unknown_provider"),
        (
            {"services.stt": {"enabled": True, "providerId": "openai-realtime"}},
            "missing_credential",
        ),
        ({"services.stt": {"enabled": False, "providerId": "local"}}, "stt_disabled"),
        (
            {"services.stt": {"enabled": True, "providerId": "local", "apiKey": FAKE_KEY}},
            "secret_value_rejected",
        ),
        ({"asr.device": "gpu"}, "invalid_option"),
    ],
)
def test_invalid_session_config_fails_with_a_safe_code(settings, code):
    with pytest.raises(STTSessionError) as info:
        resolve_session_config(settings, ("mic",))

    assert info.value.code == code
    assert FAKE_KEY not in str(info.value)


# ── Local path ───────────────────────────────────────────────────────────────


async def test_local_session_transcribes_both_audio_sources_and_stops_exactly_once():
    events: list = []
    local = FakeSTT("local", ("mic", "system"))
    config = resolve_session_config({"asr.language": "en"}, ("mic", "system"))
    session = STTSessionOrchestrator(
        config, events.append, local_builder=lambda cfg: local, poll_interval_s=0.005
    )

    await session.start()
    assert local.start_calls == 1
    for sid in ("mic", "system"):
        assert session.submit_audio(sid, _second(1), speech=True)
    await wait_until(
        lambda: all(len(local.audio[sid]) == BYTES_PER_SECOND for sid in ("mic", "system"))
    )

    local.emit("mic", "local:mic:0", "hello from the mic", 0, 800, final=False, language="en")
    local.emit("mic", "local:mic:0", "hello from the mic", 0, 900, revision=2, language="en")
    local.emit("system", "local:system:0", "hello from the speaker", 0, 900)
    await wait_until(lambda: len(_finals(events)) == 2)

    partial = next(e for e in events if isinstance(e, STTTranscriptEvent) and not e.is_final)
    assert (partial.stream_id, partial.revision, partial.provider_id, partial.language) == (
        "mic",
        1,
        "local",
        "en",
    )
    assert _finals(events, "mic") == ["hello from the mic"]
    assert _finals(events, "system") == ["hello from the speaker"]

    session.request_endpoint("mic")
    await wait_until(lambda: local.finalized == ["mic"])

    await asyncio.gather(session.stop(), session.stop())
    await session.stop()

    assert local.stop_calls == 1
    assert session.owned_tasks() == set()
    assert session.owned_providers() == set()
    assert session.state == "stopped"
    closing = events[-2:]
    assert {(e.stream_id, e.is_final, e.text) for e in closing} == {
        ("mic", True, ""),
        ("system", True, ""),
    }
    assert not session.submit_audio("mic", _second(2))


async def test_local_session_gates_audio_with_vad_and_bounded_preroll():
    local = FakeSTT("local", ("mic",))
    session = STTSessionOrchestrator(
        resolve_session_config({}, ("mic",)),
        lambda event: None,
        local_builder=lambda cfg: local,
        poll_interval_s=0.005,
    )
    await session.start()

    silence = [bytes([index]) * CHUNK_20MS for index in range(1, 6)]
    for chunk in silence:
        session.submit_audio("mic", chunk, speech=False)
    speech = bytes([9]) * CHUNK_20MS
    session.submit_audio("mic", speech, speech=True)

    await wait_until(lambda: len(local.audio["mic"]) == 4 * CHUNK_20MS)
    assert bytes(local.audio["mic"]) == silence[2] + silence[3] + silence[4] + speech

    await session.stop()


async def test_local_failure_is_reported_and_never_retried():
    events: list = []
    local = FakeSTT("local", ("mic",))
    session = STTSessionOrchestrator(
        resolve_session_config({}, ("mic",)),
        events.append,
        local_builder=lambda cfg: local,
        poll_interval_s=0.005,
    )
    await session.start()

    local.fail(ProviderErrorCode.INTERNAL_ERROR, retryable=False)
    await wait_until(lambda: _statuses(events))

    failed = _statuses(events)[0]
    assert (failed.status, failed.previous_provider_id, failed.reason, failed.state) == (
        STTStatusKind.FAILED,
        "local",
        "internal_error",
        "failed",
    )
    assert session.active_provider_id("mic") is None
    assert local.start_calls == 1

    await session.stop()
    assert local.stop_calls == 1


# ── Atomic Start ─────────────────────────────────────────────────────────────


async def test_cloud_start_without_credentials_fails_before_building_anything():
    built: list = []
    local = FakeSTT("local")
    session = STTSessionOrchestrator(
        resolve_session_config(CLOUD_START, ("mic",)),
        lambda event: None,
        cloud_builder=lambda *args: built.append(args),
        local_builder=lambda cfg: local,
    )

    with pytest.raises(STTSessionError) as info:
        await session.start()

    assert info.value.code == ERROR_MISSING_CREDENTIAL
    assert built == []
    assert (local.health_calls, local.start_calls) == (0, 0)
    assert session.state == "stopped"
    assert session.owned_tasks() == set()
    assert not session.submit_audio("mic", _second(0))
    await session.stop()


async def test_cloud_start_requires_a_healthy_local_fallback():
    built: list = []
    local = FakeSTT("local", health=ProviderHealthStatus.UNHEALTHY)
    session = STTSessionOrchestrator(
        resolve_session_config(CLOUD_START, ("mic",)),
        lambda event: None,
        cloud_builder=lambda *args: built.append(args),
        local_builder=lambda cfg: local,
        secrets_resolver=_secrets,
    )

    with pytest.raises(STTSessionError) as info:
        await session.start()

    assert info.value.code == ERROR_LOCAL_FALLBACK_UNAVAILABLE
    assert built == []
    assert (local.health_calls, local.start_calls) == (1, 0)


async def test_failed_cloud_start_releases_every_provider_once_and_clears_secrets():
    events: list = []
    clouds: list = []
    received_secrets: list = []
    auth_failure = ProviderRuntimeError(_error(ProviderErrorCode.AUTH_FAILED))

    def builder(resolved, secrets, ids):
        received_secrets.append(secrets)
        provider = FakeSTT(
            resolved.provider_id, ids, start_error=auth_failure if ids == ("system",) else None
        )
        clouds.append(provider)
        return provider

    session = STTSessionOrchestrator(
        resolve_session_config(CLOUD_START, ("mic", "system")),
        events.append,
        cloud_builder=builder,
        local_builder=lambda cfg: FakeSTT("local", ("mic", "system")),
        secrets_resolver=_secrets,
    )

    with pytest.raises(STTSessionError) as info:
        await session.start()

    assert info.value.code == "auth_failed"
    assert [c.stream_ids for c in clouds] == [("mic",), ("system",)]
    assert [c.stop_calls for c in clouds] == [1, 1]
    assert session.owned_providers() == set()
    assert session.owned_tasks() == set()
    assert events == []
    # The session's copy of the credentials does not outlive adapter construction.
    assert all(secrets == {} for secrets in received_secrets)


# ── Failover ─────────────────────────────────────────────────────────────────


async def test_cloud_failure_falls_back_to_local_with_bounded_replay_and_no_duplicate_final():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic",))
    network_down = ProviderRuntimeError(_error(ProviderErrorCode.NETWORK_UNAVAILABLE, True))
    session = _cloud_session(events, clouds, local, per_attempt={1: {"start_error": network_down}})

    await session.start()
    assert (local.health_calls, local.start_calls) == (1, 0)
    await _speak_cloud_transcript(session, clouds[0], events)

    clouds[0].fail(ProviderErrorCode.CONNECTION_CLOSED, retryable=True)
    await wait_until(lambda: session.active_provider_id("mic") == "local")

    # Replay is bounded: last final boundary (6000 ms) minus the 2000 ms overlap, to the end.
    await wait_until(lambda: len(local.audio["mic"]) == 4 * BYTES_PER_SECOND)
    assert bytes(local.audio["mic"][:BYTES_PER_SECOND]) == _second(4)
    assert bytes(local.audio["mic"][-BYTES_PER_SECOND:]) == _second(7)

    # Live audio continues on Local right after the replayed range.
    assert session.submit_audio("mic", _second(8))
    await wait_until(lambda: len(local.audio["mic"]) == 5 * BYTES_PER_SECOND)

    local.emit("mic", "l1", "schedule the meeting", 0, 2200)
    local.emit("mic", "l2", "the meeting for tomorrow at noon", 1600, 4500)
    local.emit("mic", "l3", "see you then", 5000, 6500)
    await wait_until(lambda: len(_finals(events)) == 4)

    committed = _finals(events)
    assert committed == [
        "Good morning everyone.",
        "We should schedule the meeting",
        "for tomorrow at noon",
        "see you then",
    ]
    assert " ".join(committed).lower().count("the meeting") == 1
    assert any(
        e.retracted and e.duplicate_of == "mic:1:c2"
        for e in events
        if isinstance(e, STTTranscriptEvent)
    )

    reconnecting, fallback = _statuses(events)
    assert (reconnecting.status, reconnecting.reason) == (
        STTStatusKind.RECONNECTING,
        "connection_closed",
    )
    assert fallback.status is STTStatusKind.FALLBACK
    assert (fallback.provider_id, fallback.previous_provider_id) == ("local", "openai-realtime")
    assert (fallback.reason, fallback.state) == ("network_unavailable", "local_active")
    assert (fallback.replay_start_ms, fallback.replay_end_ms, fallback.gap) == (
        4000.0,
        8000.0,
        None,
    )

    # Sticky Local: no further cloud attempts, and the Local endpoint still works.
    assert len(clouds) == 2
    assert session.failover_state("mic") is FailoverState.LOCAL_ACTIVE
    session.request_endpoint("mic")
    await wait_until(lambda: local.finalized == ["mic"])

    await session.stop()
    assert [c.stop_calls for c in clouds] == [1, 1]
    assert local.stop_calls == 1
    assert session.owned_providers() == set()
    assert session.owned_tasks() == set()


async def test_reconnect_within_deadline_resumes_cloud_with_replay():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic",))
    session = _cloud_session(events, clouds, local)

    await session.start()
    await _speak_cloud_transcript(session, clouds[0], events)
    clouds[0].fail(ProviderErrorCode.CONNECTION_CLOSED, retryable=True)
    await wait_until(lambda: len(_statuses(events)) == 2)

    assert [s.status for s in _statuses(events)] == [
        STTStatusKind.RECONNECTING,
        STTStatusKind.RECONNECTED,
    ]
    assert session.active_provider_id("mic") == "openai-realtime"
    await wait_until(lambda: len(clouds[1].audio["mic"]) == 4 * BYTES_PER_SECOND)
    assert bytes(clouds[1].audio["mic"][:BYTES_PER_SECOND]) == _second(4)

    clouds[1].emit("mic", "r1", "schedule the meeting", 0, 2200)
    clouds[1].emit("mic", "r2", "the meeting for tomorrow at noon", 1600, 4500)
    await wait_until(lambda: len(_finals(events)) == 3)
    assert _finals(events)[-1] == "for tomorrow at noon"
    assert local.start_calls == 0

    await session.stop()
    assert [c.stop_calls for c in clouds] == [1, 1]
    assert local.stop_calls == 0
    assert session.owned_tasks() == set()


async def test_reconnect_that_misses_the_deadline_falls_back_to_local():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic",))
    policy = FailoverPolicy(reconnect_timeout_s=0.05)
    session = _cloud_session(
        events, clouds, local, policy=policy, per_attempt={1: {"start_hang": True}}
    )

    await session.start()
    assert session.submit_audio("mic", _second(0))
    await wait_until(lambda: len(clouds[0].audio["mic"]) == BYTES_PER_SECOND)
    clouds[0].fail(ProviderErrorCode.TIMEOUT, retryable=True)
    await wait_until(lambda: session.active_provider_id("mic") == "local")

    fallback = _statuses(events)[-1]
    assert (fallback.status, fallback.reason) == (STTStatusKind.FALLBACK, "timeout")
    assert clouds[1].stop_calls == 1
    await wait_until(lambda: len(local.audio["mic"]) == BYTES_PER_SECOND)

    await session.stop()
    assert local.stop_calls == 1
    assert session.owned_tasks() == set()


async def test_non_retryable_cloud_failure_skips_reconnect():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic",))
    session = _cloud_session(events, clouds, local)

    await session.start()
    clouds[0].fail(ProviderErrorCode.AUTH_FAILED, retryable=False)
    await wait_until(lambda: session.active_provider_id("mic") == "local")

    assert len(clouds) == 1
    assert [s.status for s in _statuses(events)] == [STTStatusKind.FALLBACK]
    await session.stop()


async def test_streams_fail_over_independently_and_share_one_local_provider():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic", "system"))
    session = _cloud_session(
        events,
        clouds,
        local,
        stream_ids=("mic", "system"),
    )

    await session.start()
    mic_cloud = next(c for c in clouds if c.stream_ids == ("mic",))
    mic_cloud.fail(ProviderErrorCode.AUTH_FAILED, retryable=False)
    await wait_until(lambda: session.active_provider_id("mic") == "local")

    assert session.active_provider_id("system") == "openai-realtime"
    assert session.failover_state("system") is FailoverState.CLOUD_ACTIVE
    assert local.start_calls == 1

    await session.stop()
    assert all(c.stop_calls == 1 for c in clouds)
    assert local.stop_calls == 1


async def test_capture_never_blocks_and_dropped_audio_keeps_the_provider_timeline():
    events: list = []
    clouds: list = []
    local = FakeSTT("local", ("mic",))
    session = _cloud_session(events, clouds, local)
    session._config = dataclasses.replace(session.config, max_pending_frames=2)

    await session.start()
    accepted = [session.submit_audio("mic", bytes([index + 1]) * CHUNK_20MS) for index in range(10)]
    assert accepted == [True, True] + [False] * 8
    assert session.dropped_frames("mic") == 8

    await wait_until(lambda: len(clouds[0].audio["mic"]) == 2 * CHUNK_20MS)
    assert session.submit_audio("mic", bytes([42]) * CHUNK_20MS)
    await wait_until(lambda: len(clouds[0].audio["mic"]) == 11 * CHUNK_20MS)
    sent = bytes(clouds[0].audio["mic"])
    assert sent[2 * CHUNK_20MS : 10 * CHUNK_20MS] == bytes(8 * CHUNK_20MS)
    assert sent[-CHUNK_20MS:] == bytes([42]) * CHUNK_20MS

    await session.stop()


# ── Engine wiring ────────────────────────────────────────────────────────────


class FakeInput:
    """Stands in for MicrophoneInput / SystemAudioInput; opens nothing."""

    def __init__(self, created: list, config: dict, device_id=None) -> None:
        self.started = False
        self.stopped = False
        self.remaining = 25
        created.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.started = False

    def read_chunk(self) -> bytes:
        if not self.started or self.remaining <= 0:
            time.sleep(0.005)
            return b""
        self.remaining -= 1
        time.sleep(0.002)
        return b"\x01\x00" * (CHUNK_20MS // 2)

    def is_active(self) -> bool:
        return self.started

    def list_devices(self) -> list:
        return []


class FakeVAD:
    def __init__(self, config: dict) -> None:
        self.config = config

    def process(self, audio_chunk: bytes) -> bool:
        return True

    def reset(self) -> None:
        return None


@pytest.fixture
def fake_audio(monkeypatch):
    import engine.audio.microphone as microphone
    import engine.audio.system_audio as system_audio
    import engine.audio.vad as vad

    created: list = []
    monkeypatch.setattr(microphone, "MicrophoneInput", lambda *a, **k: FakeInput(created, *a, **k))
    monkeypatch.setattr(
        system_audio, "SystemAudioInput", lambda *a, **k: FakeInput(created, *a, **k)
    )
    monkeypatch.setattr(vad, "VAD", FakeVAD)
    return created


def _first_utterance(provider: FakeSTT, frame) -> None:
    sid = frame.stream_id
    if sid in provider.emitted or len(provider.audio[sid]) < 3 * CHUNK_20MS:
        return
    provider.emitted.add(sid)
    provider.emit(sid, f"local:{sid}:0", f"{sid} words", 0, 40, final=False)
    provider.emit(sid, f"local:{sid}:0", f"{sid} words", 0, 60, revision=2, language="en")


def _engine(monkeypatch):
    from engine.main import EchoFluxEngine

    engine = EchoFluxEngine()
    sent: list = []

    async def record(message):
        sent.append(message)

    monkeypatch.setattr(engine._ws_server, "broadcast", record)
    return engine, sent


def _alive_pipeline_threads() -> list:
    return [
        t
        for t in threading.enumerate()
        if t.is_alive() and t.name.startswith(("Capture-", "Process-"))
    ]


async def test_engine_start_runs_both_sources_through_the_session_and_stop_releases_all(
    fake_audio, monkeypatch
):
    engine, sent = _engine(monkeypatch)
    local = FakeSTT("local", ("mic", "system"), on_audio=_first_utterance)
    engine._stt_local_builder = lambda cfg: local

    await engine._on_client_start(
        {"audio.source": "both", "asr.model_size": "tiny", "asr.language": "en"}
    )

    session = engine._stt_session
    assert session is not None
    assert session.config.stream_ids == ("mic", "system")
    assert session.config.legacy_model_size == "tiny"
    assert local.start_calls == 1
    assert len(fake_audio) == 2 and all(i.started for i in fake_audio)

    def final_sources():
        return {m.get("source") for m in sent if m["type"] == "final" and m["text"]}

    await wait_until(lambda: final_sources() == {"mic", "system"})
    final = next(m for m in sent if m["type"] == "final" and m["text"])
    partial = next(m for m in sent if m["type"] == "partial")
    assert final["entry_id"].startswith("e-")
    assert final["translation"] is None and final["is_final"] is True
    assert (final["stt_provider"], final["revision"], final["language"]) == ("local", 2, "en")
    assert partial["is_final"] is False and "entry_id" not in partial
    entry_ids = [m["entry_id"] for m in sent if m["type"] == "final" and m["text"]]
    assert len(set(entry_ids)) == len(entry_ids)

    await engine._on_client_stop()

    assert engine._stt_session is None
    assert not engine._running
    assert local.stop_calls == 1
    assert session.owned_tasks() == set()
    assert session.owned_providers() == set()
    assert all(i.stopped for i in fake_audio)
    assert _alive_pipeline_threads() == []


async def test_engine_cloud_start_without_credentials_opens_no_audio_device(
    fake_audio, monkeypatch
):
    engine, _ = _engine(monkeypatch)
    replies: list = []

    class Client:
        async def send(self, raw):
            replies.append(json.loads(raw))

    monkeypatch.setattr(engine._ws_server, "current_requester", lambda: Client())
    local = FakeSTT("local", ("mic",))
    engine._stt_local_builder = lambda cfg: local

    await engine._on_client_start({**CLOUD_START, "audio.source": "microphone"})

    assert len(fake_audio) == 1 and not fake_audio[0].started
    assert engine._stt_session is None
    assert not engine._running
    assert engine._inputs == {}
    assert engine._session_config is None
    assert local.start_calls == 0
    assert replies == [
        {
            "type": "error",
            "message": "Speech recognition credentials are not available to the engine.",
        }
    ]


async def test_engine_reports_cloud_to_local_fallback_on_status(fake_audio, monkeypatch):
    engine, sent = _engine(monkeypatch)
    clouds: list = []
    local = FakeSTT("local", ("mic",))

    def builder(resolved, secrets, ids):
        provider = FakeSTT(resolved.provider_id, ids)
        clouds.append(provider)
        return provider

    engine._stt_local_builder = lambda cfg: local
    engine._stt_cloud_builder = builder
    engine._stt_secrets_resolver = _secrets

    await engine._on_client_start({**CLOUD_START, "audio.source": "microphone"})
    assert engine._stt_session is not None and fake_audio[0].started
    await wait_until(lambda: len(clouds[0].audio["mic"]) > 0)

    clouds[0].fail(ProviderErrorCode.AUTH_FAILED, retryable=False)
    await wait_until(lambda: any(m.get("status") == "stt_fallback" for m in sent))

    status = next(m for m in sent if m.get("status") == "stt_fallback")
    assert status["type"] == "status"
    assert (status["source"], status["stt_provider"], status["previous_stt_provider"]) == (
        "mic",
        "local",
        "openai-realtime",
    )
    assert status["reason"] == "auth_failed" and status["stt_state"] == "local_active"
    assert status["replay"]["start_ms"] == 0.0
    assert FAKE_KEY not in json.dumps(sent)

    await engine._on_client_stop()
    assert [c.stop_calls for c in clouds] == [1]
    assert local.stop_calls == 1
    assert _alive_pipeline_threads() == []


def test_engine_failed_stream_status_is_followed_by_a_display_error():
    from engine.main import EchoFluxEngine

    engine = EchoFluxEngine()
    engine._on_stt_event(
        STTStatusEvent(
            stream_id="system",
            status=STTStatusKind.FAILED,
            provider_id=None,
            previous_provider_id="local",
            reason="internal_error",
            state="failed",
        )
    )

    status = engine._result_queue.get_nowait()
    error = engine._result_queue.get_nowait()
    assert (status["type"], status["status"], status["source"]) == (
        "status",
        "stt_failed",
        "system",
    )
    assert error["type"] == "error" and error["message"]
