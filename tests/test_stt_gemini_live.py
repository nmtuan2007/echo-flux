"""Scripted fake-server tests for the Gemini Live STT adapter.

No network: every connection is an in-memory FakeConnection driven by the test, and
rotation deadlines run on an injected clock.
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pytest

from engine.providers.stt import gemini_protocol as proto
from engine.providers.stt.gemini_live import (
    GeminiLiveSTTConfig,
    GeminiLiveSTTProvider,
    LiveConnectError,
    LiveConnectionClosedError,
)
from engine.providers.types import (
    AudioFrame,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

SECRET = "AIzaTEST-secret-key-0123456789"
PCM_100MS = b"\x01\x00" * 1600  # 100 ms of 16 kHz PCM16 mono


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeConnection:
    """Scripted server side of one Live connection."""

    def __init__(self, auto_setup: bool = True) -> None:
        self.auto_setup = auto_setup
        self.sent: List[Dict[str, Any]] = []
        self.closed = False
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, message: str) -> None:
        if self.closed:
            raise LiveConnectionClosedError(1000, "closed")
        payload = json.loads(message)
        self.sent.append(payload)
        if "setup" in payload and self.auto_setup:
            self.push({"setupComplete": {}})

    async def recv(self) -> Any:
        item = await self._incoming.get()
        if isinstance(item, LiveConnectionClosedError):
            self.closed = True
            raise item
        return item

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._incoming.put_nowait(LiveConnectionClosedError(1000, ""))

    def push(self, message: Any) -> None:
        self._incoming.put_nowait(json.dumps(message) if isinstance(message, dict) else message)

    def push_close(self, code: int, reason: str = "") -> None:
        self._incoming.put_nowait(LiveConnectionClosedError(code, reason))

    def audio_payloads(self) -> List[bytes]:
        return [
            base64.b64decode(m["realtimeInput"]["audio"]["data"])
            for m in self.sent
            if "audio" in m.get("realtimeInput", {})
        ]

    def has_stream_end(self) -> bool:
        return any(m.get("realtimeInput", {}).get("audioStreamEnd") for m in self.sent)


class FakeConnector:
    def __init__(self, factory=None) -> None:
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeConnection] = []
        self._factory = factory or (lambda index: FakeConnection())

    async def __call__(self, url: str, headers: Dict[str, str]) -> FakeConnection:
        self.calls.append((url, dict(headers)))
        result = self._factory(len(self.calls) - 1)
        if isinstance(result, Exception):
            raise result
        self.connections.append(result)
        return result


def interim(text: str) -> Dict[str, Any]:
    return {"serverContent": {"interimInputTranscription": {"text": text}}}


def final(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"text": text}
    if language:
        body["languageCode"] = language
    return {"serverContent": {"inputTranscription": body}}


def frame(sequence: int, pcm: bytes = PCM_100MS, stream_id: str = "mic") -> AudioFrame:
    return AudioFrame(stream_id=stream_id, sequence=sequence, pcm_data=pcm, duration_ms=100.0)


async def settle(rounds: int = 25) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def next_event(provider: GeminiLiveSTTProvider, timeout: float = 1.0) -> STTTranscriptResult:
    stream = provider.receive_events()
    try:
        return await asyncio.wait_for(stream.__anext__(), timeout)
    finally:
        await stream.aclose()


def drain_events(provider: GeminiLiveSTTProvider) -> List[Any]:
    queue = provider._events
    assert queue is not None
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def make_provider(
    connector: FakeConnector, clock: Optional[FakeClock] = None, **overrides: Any
) -> GeminiLiveSTTProvider:
    options = {"api_key": SECRET, "chunk_ms": 0.0, **overrides}
    return GeminiLiveSTTProvider(
        GeminiLiveSTTConfig(**options), connector=connector, clock=clock or FakeClock()
    )


# ── Protocol ─────────────────────────────────────────────────────────────────


def test_setup_message_shape_and_auto_language():
    message = proto.build_setup_message("gemini-3.5-transcribe-live")
    assert list(message) == ["setup"]
    setup = message["setup"]
    assert setup["model"] == "models/gemini-3.5-transcribe-live"
    assert setup["generationConfig"] == {"responseModalities": ["TEXT"]}
    assert setup["inputAudioTranscription"] == {}
    assert "realtimeInputConfig" not in setup

    auto = proto.build_setup_message(None, language_hints=["auto", " "])
    assert "languageCodes" not in auto["setup"]["inputAudioTranscription"]
    assert auto["setup"]["model"] == f"models/{proto.DEFAULT_MODEL}"


def test_setup_message_carries_language_hints_and_vocabulary():
    message = proto.build_setup_message(
        "models/custom",
        language_hints=["en-US", "vi-VN", "en-US"],
        custom_vocabulary=["EchoFlux", "", "Tauri"],
        mode="smart",
        manual_activity=True,
    )
    setup = message["setup"]
    assert setup["model"] == "models/custom"
    assert setup["inputAudioTranscription"] == {
        "languageCodes": ["en-US", "vi-VN"],
        "customVocabulary": ["EchoFlux", "Tauri"],
        "mode": "SMART",
    }
    assert setup["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}

    with pytest.raises(ValueError):
        proto.build_setup_message(None, custom_vocabulary=[f"t{i}" for i in range(1001)])


def test_parse_server_message_events():
    events = proto.parse_server_message(
        json.dumps(
            {
                "serverContent": {
                    "inputTranscription": {"text": "hello", "languageCode": "en"},
                    "turnComplete": True,
                }
            }
        )
    )
    assert [e.kind for e in events] == [
        proto.ServerEventKind.FINAL,
        proto.ServerEventKind.TURN_COMPLETE,
    ]
    assert events[0].text == "hello" and events[0].language_code == "en"

    go_away = proto.parse_server_message(b'{"goAway": {"timeLeft": "12.5s"}}')
    assert go_away[0].kind is proto.ServerEventKind.GO_AWAY
    assert go_away[0].time_left_s == 12.5

    snake = proto.parse_server_message(
        '{"server_content": {"interim_input_transcription": {"text": "hi"}}}'
    )
    assert snake[0].kind is proto.ServerEventKind.INTERIM

    assert proto.parse_server_message('{"sessionResumptionUpdate": {"newHandle": "h"}}') == []


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff\xfe",
        "not json",
        "[1, 2]",
        '{"serverContent": "oops"}',
        '{"serverContent": {"inputTranscription": {"text": 5}}}',
        '{"goAway": 3}',
    ],
)
def test_parse_server_message_rejects_malformed(raw):
    with pytest.raises(proto.GeminiProtocolError):
        proto.parse_server_message(raw)


def test_close_and_status_classification():
    assert proto.classify_close(1007, "API key not valid. Please pass a valid API key.") == (
        ProviderErrorCode.AUTH_FAILED,
        False,
    )
    assert proto.classify_close(1011, "RESOURCE_EXHAUSTED: quota")[0] is (
        ProviderErrorCode.QUOTA_EXCEEDED
    )
    assert proto.classify_close(1008, "models/x is not found")[0] is (
        ProviderErrorCode.MODEL_UNAVAILABLE
    )
    assert proto.classify_close(1011, "Internal error") == (ProviderErrorCode.INTERNAL_ERROR, True)
    assert proto.classify_close(1007, "bad frame") == (ProviderErrorCode.INVALID_REQUEST, False)
    assert proto.classify_http_status(403) == (ProviderErrorCode.AUTH_FAILED, False)
    assert proto.classify_http_status(429) == (ProviderErrorCode.RATE_LIMITED, True)
    assert proto.classify_http_status(None) == (ProviderErrorCode.NETWORK_UNAVAILABLE, True)


def test_secret_bearing_urls_and_text_are_redacted():
    url = proto.build_connect_url(proto.EPHEMERAL_TOKEN_ENDPOINT, access_token=SECRET)
    assert SECRET in url
    redacted = proto.redact_url(url)
    assert SECRET not in redacted
    assert "access_token=***" in redacted

    keyed = proto.redact_url("wss://example.test/ws?alt=json&key=abc123secret")
    assert "abc123secret" not in keyed and "alt=json" in keyed

    text = proto.redact_text(f"bad key {SECRET} at wss://h/x?key=zzzz9999", [SECRET])
    assert SECRET not in text and "zzzz9999" not in text


def test_config_from_options_and_repr_hides_key():
    config = GeminiLiveSTTConfig.from_options(
        {
            "api_key": SECRET,
            "endpoint": "wss://proxy.test/ws",
            "model_override": "gemini-live-custom",
            "language_hints": "en-US, vi-VN",
            "custom_vocabulary": ["EchoFlux"],
            "unknown": 1,
        }
    )
    assert config.endpoint == "wss://proxy.test/ws"
    assert config.model == "gemini-live-custom"
    assert config.language_hints == ("en-US", "vi-VN")
    assert config.custom_vocabulary == ("EchoFlux",)
    assert SECRET not in repr(config)
    assert GeminiLiveSTTConfig.from_options({"api_key": SECRET}).language_hints == ()


# ── Setup ────────────────────────────────────────────────────────────────────


async def test_setup_is_first_message_and_audio_waits_for_setup_complete(caplog):
    caplog.set_level(logging.DEBUG)
    connector = FakeConnector(lambda i: FakeConnection(auto_setup=False))
    provider = make_provider(connector)
    await provider.start()

    sending = asyncio.ensure_future(provider.send_audio(frame(1)))
    await settle()
    connection = connector.connections[0]
    assert [list(m) for m in connection.sent] == [["setup"]]
    assert not sending.done()

    connection.push({"setupComplete": {}})
    await asyncio.wait_for(sending, 1.0)
    assert list(connection.sent[0]) == ["setup"]
    audio = connection.sent[1]["realtimeInput"]["audio"]
    assert audio["mimeType"] == "audio/pcm;rate=16000"
    assert base64.b64decode(audio["data"]) == PCM_100MS

    url, headers = connector.calls[0]
    assert url == proto.DEFAULT_ENDPOINT
    assert headers == {proto.API_KEY_HEADER: SECRET}
    assert SECRET not in caplog.text
    await provider.stop()


async def test_auto_language_setup_when_no_hints():
    connector = FakeConnector()
    provider = make_provider(connector, language_hints=())
    await provider.start()
    await provider.send_audio(frame(1))
    transcription = connector.connections[0].sent[0]["setup"]["inputAudioTranscription"]
    assert "languageCodes" not in transcription
    await provider.stop()


async def test_setup_rejection_raises_categorized_redacted_error():
    def factory(index: int) -> FakeConnection:
        connection = FakeConnection(auto_setup=False)
        connection.push_close(1007, f"API key not valid: {SECRET}")
        return connection

    connector = FakeConnector(factory)
    provider = make_provider(connector)
    await provider.start()

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(1))
    error = exc_info.value.error
    assert error.code is ProviderErrorCode.AUTH_FAILED
    assert error.retryable is False
    assert SECRET not in str(error) and SECRET not in json.dumps(error.details)
    assert connector.connections[0].closed
    assert connector.connections[0].audio_payloads() == []

    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.UNHEALTHY
    assert SECRET not in health.message
    await provider.stop()


async def test_handshake_rejection_and_setup_timeout():
    provider = make_provider(FakeConnector(lambda i: LiveConnectError(403, "forbidden")))
    await provider.start()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(1))
    assert exc_info.value.error.code is ProviderErrorCode.AUTH_FAILED
    await provider.stop()

    connector = FakeConnector(lambda i: FakeConnection(auto_setup=False))
    provider = make_provider(connector, setup_timeout_s=0.05)
    await provider.start()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(1))
    assert exc_info.value.error.code is ProviderErrorCode.TIMEOUT
    assert exc_info.value.error.retryable is True
    assert connector.connections[0].closed
    await provider.stop()


async def test_access_token_mode_keeps_token_out_of_logs(caplog):
    caplog.set_level(logging.DEBUG)
    connector = FakeConnector()
    provider = make_provider(
        connector, auth_mode="access_token", endpoint=proto.EPHEMERAL_TOKEN_ENDPOINT
    )
    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.HEALTHY
    url, headers = connector.calls[0]
    assert f"access_token={SECRET}" in url and headers == {}
    assert SECRET not in caplog.text
    assert "access_token=***" in caplog.text
    assert connector.connections[0].closed


async def test_start_requires_credential_and_valid_frames():
    provider = GeminiLiveSTTProvider(GeminiLiveSTTConfig(), connector=FakeConnector())
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code is ProviderErrorCode.AUTH_FAILED

    provider = make_provider(FakeConnector())
    await provider.start()
    stereo = AudioFrame(stream_id="mic", sequence=1, pcm_data=b"\x00" * 8, channels=2)
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(stereo)
    assert exc_info.value.error.code is ProviderErrorCode.INVALID_REQUEST
    with pytest.raises(ProviderRuntimeError):
        await provider.send_audio(frame(2, pcm=b"\x00\x00\x00"))
    await provider.stop()


# ── Transcription events ─────────────────────────────────────────────────────


async def test_pcm_is_coalesced_into_chunks_and_finalize_flushes():
    connector = FakeConnector()
    provider = make_provider(connector, chunk_ms=100.0)
    await provider.start()
    small = b"\x02\x00" * 512  # 32 ms
    for sequence in range(3):
        await provider.send_audio(frame(sequence, pcm=small))
    assert connector.calls == []  # 96 ms buffered, nothing sent yet

    await provider.send_audio(frame(3, pcm=small))
    connection = connector.connections[0]
    assert connection.audio_payloads() == [small * 4]

    await provider.send_audio(frame(4, pcm=small))
    await provider.finalize_stream("mic")
    assert connection.audio_payloads() == [small * 4, small]
    assert connection.sent[-1] == {"realtimeInput": {"audioStreamEnd": True}}
    await provider.stop()


async def test_interim_and_final_events_are_canonical_and_deduplicated():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(connector, clock)
    await provider.start()
    await provider.send_audio(frame(1))
    await provider.send_audio(frame(2))
    connection = connector.connections[0]

    connection.push(interim("hel"))
    connection.push(interim("hello"))
    connection.push(interim("hello"))  # duplicate hypothesis
    connection.push(final("hello world", "en"))
    connection.push(final("hello world", "en"))  # duplicate final
    connection.push({"serverContent": {"turnComplete": True}})
    await settle()

    events = drain_events(provider)
    assert [(e.text, e.is_final) for e in events] == [
        ("hel", False),
        ("hello", False),
        ("hello world", True),
    ]
    assert [e.revision for e in events] == [1, 2, 3]
    last = events[-1]
    assert last.is_endpoint and last.detected_language == "en"
    assert (last.start_time_ms, last.end_time_ms) == (0.0, 200.0)
    assert last.sequence == 2 and last.stream_id == "mic"
    assert len({e.provider_event_id for e in events}) == 1
    await provider.stop()


async def test_malformed_frames_are_skipped_without_ending_stream():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    await provider.send_audio(frame(1))
    connection = connector.connections[0]
    for raw in (b"\xff\xfe", "not json", "[1]", '{"serverContent": {"inputTranscription": 7}}'):
        connection.push(raw)
    connection.push(final("still here"))

    event = await next_event(provider)
    assert event.text == "still here" and event.is_final
    assert not connection.closed
    await provider.stop()


async def test_unexpected_close_surfaces_retryable_error_then_reconnects():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    await provider.send_audio(frame(1))
    connector.connections[0].push_close(1011, "Internal error encountered.")

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await next_event(provider)
    assert exc_info.value.error.code is ProviderErrorCode.INTERNAL_ERROR
    assert exc_info.value.error.retryable is True

    await provider.send_audio(frame(2))
    assert len(connector.connections) == 2
    assert list(connector.connections[1].sent[0]) == ["setup"]
    await provider.stop()


# ── Rotation ─────────────────────────────────────────────────────────────────


async def test_rotation_waits_for_boundary_and_does_not_duplicate_final():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(connector, clock, rotate_after_s=100.0, max_connection_s=200.0)
    await provider.start()

    await provider.send_audio(frame(1))
    first = connector.connections[0]
    first.push(interim("good morning"))
    await settle()

    clock.advance(150.0)  # past soft deadline, mid-utterance
    await provider.send_audio(frame(2))
    await settle()
    await provider.send_audio(frame(3))
    assert len(connector.connections) == 2
    second = connector.connections[1]
    assert [list(m) for m in second.sent] == [["setup"]]  # standby, no audio yet
    assert len(first.audio_payloads()) == 3

    first.push(final("good morning everyone"))
    await settle()
    await provider.send_audio(frame(4))  # boundary reached: swap
    await settle()
    assert len(first.audio_payloads()) == 3
    assert second.audio_payloads() == [PCM_100MS]
    assert first.has_stream_end() and first.closed

    first.push(final("good morning everyone"))  # replay on the retired connection
    second.push(final("good morning everyone"))  # provider repeating stale text
    second.push(interim("next"))
    second.push(final("next topic"))
    await settle()

    events = drain_events(provider)
    finals = [e for e in events if e.is_final]
    assert [e.text for e in finals] == ["good morning everyone", "next topic"]
    revisions = [e.revision for e in events]
    assert revisions == sorted(revisions) and len(set(revisions)) == len(revisions)
    assert finals[0].end_time_ms <= finals[1].start_time_ms
    assert finals[1].start_time_ms == 300.0 and finals[1].end_time_ms == 400.0
    assert finals[0].sequence < finals[1].sequence
    assert all(not isinstance(e, Exception) for e in events)
    await provider.stop()


async def test_hard_deadline_swaps_mid_utterance_and_old_final_arrives_once():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(
        connector, clock, rotate_after_s=100.0, max_connection_s=200.0, drain_timeout_s=1.0
    )
    await provider.start()
    await provider.send_audio(frame(1))
    first = connector.connections[0]
    first.push(interim("long sentence"))
    await settle()

    clock.advance(250.0)  # past the hard deadline with the utterance still open
    await provider.send_audio(frame(2))
    await settle()
    second = connector.connections[1]
    assert second.audio_payloads() == [PCM_100MS]
    assert first.has_stream_end() and not first.closed  # draining

    first.push(final("long sentence finished"))
    first.push(final("long sentence finished"))
    await settle()
    assert first.closed

    second.push(final("fresh start"))
    await settle()
    finals = [e for e in drain_events(provider) if e.is_final]
    assert [e.text for e in finals] == ["long sentence finished", "fresh start"]
    assert finals[0].end_time_ms == 100.0 and finals[1].start_time_ms == 100.0
    assert finals[0].provider_event_id != finals[1].provider_event_id
    await provider.stop()


async def test_drain_timeout_promotes_interim_exactly_once():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(
        connector, clock, rotate_after_s=10.0, max_connection_s=20.0, drain_timeout_s=0.05
    )
    await provider.start()
    await provider.send_audio(frame(1))
    first = connector.connections[0]
    first.push(interim("never finalized"))
    await settle()

    clock.advance(30.0)
    await provider.send_audio(frame(2))
    await asyncio.sleep(0.15)
    assert first.closed

    events = drain_events(provider)
    assert [(e.text, e.is_final) for e in events] == [
        ("never finalized", False),
        ("never finalized", True),
    ]
    await provider.stop()


async def test_go_away_rotates_without_surfacing_an_event():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(connector, clock)  # default deadlines are far away
    await provider.start()
    await provider.send_audio(frame(1))
    first = connector.connections[0]

    first.push({"goAway": {"timeLeft": "5s"}})
    await settle()
    assert len(connector.connections) == 2  # replacement opened proactively
    assert drain_events(provider) == []

    await provider.send_audio(frame(2))
    await settle()
    second = connector.connections[1]
    assert second.audio_payloads() == [PCM_100MS]
    assert first.closed
    assert drain_events(provider) == []
    await provider.stop()


async def test_close_after_go_away_is_silent_and_uses_standby():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(connector, clock)
    await provider.start()
    await provider.send_audio(frame(1))
    first = connector.connections[0]
    first.push(interim("keeps talking"))
    first.push({"goAway": {"timeLeft": "30s"}})
    await settle()
    first.push_close(1000, "session expired")
    await settle()
    events = drain_events(provider)
    assert [(e.text, e.is_final) for e in events] == [("keeps talking", False)]
    await provider.send_audio(frame(2))
    await settle()
    assert len(connector.connections) == 2
    assert connector.connections[1].audio_payloads() == [PCM_100MS]
    await provider.stop()


# ── Cleanup ──────────────────────────────────────────────────────────────────


async def test_stop_closes_connections_and_ends_event_stream():
    connector = FakeConnector()
    clock = FakeClock()
    provider = make_provider(connector, clock, rotate_after_s=1.0)
    await provider.start()
    await provider.send_audio(frame(1, stream_id="mic"))
    await provider.send_audio(frame(1, stream_id="system"))
    clock.advance(5.0)
    await provider.send_audio(frame(2, stream_id="mic"))  # spawns a replacement connect

    consumer = asyncio.ensure_future(_collect(provider))
    await settle()
    await provider.stop()
    events = await asyncio.wait_for(consumer, 1.0)

    assert events == []
    assert connector.connections and all(c.closed for c in connector.connections)
    assert provider._tasks == set()
    mic = connector.connections[0]
    assert mic.has_stream_end()


async def test_reset_stream_closes_connection_and_keeps_revisions_monotonic():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    await provider.send_audio(frame(1))
    connector.connections[0].push(final("before reset"))
    first_event = await next_event(provider)

    await provider.reset_stream("mic")
    assert connector.connections[0].closed

    await provider.send_audio(frame(2))
    connector.connections[1].push(final("after reset"))
    second_event = await next_event(provider)
    assert second_event.revision > first_event.revision
    assert second_event.start_time_ms >= first_event.end_time_ms
    await provider.stop()


async def _collect(provider: GeminiLiveSTTProvider) -> List[STTTranscriptResult]:
    return [event async for event in provider.receive_events()]
