"""Protocol and lifecycle tests for the Deepgram STT adapter and shared socket utility.

Nothing here reaches a real provider: most tests drive a scripted in-memory connection,
and the end-to-end cases use a local ``websockets`` server bound to 127.0.0.1.
"""

import asyncio
import json
import logging
import struct
from collections import deque
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import pytest

from engine.providers.stt import streaming_ws
from engine.providers.stt.deepgram import (
    DeepgramMalformed,
    DeepgramResults,
    DeepgramSettings,
    DeepgramSTTProvider,
    DeepgramTransportOptions,
    DeepgramUtteranceAssembler,
    build_auth_headers,
    build_connection_url,
    map_close,
    parse_message,
)
from engine.providers.stt.streaming_ws import (
    HandshakeRejectedError,
    SocketClosedError,
    StreamingSocket,
    redact_text,
    redact_url,
    validate_endpoint,
)
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

API_KEY = "dg-test-DO-NOT-LEAK-0123456789abcdef"
KEYTERM = "Nightjar"
ENDPOINT = "wss://listen.example.test/v1/listen"
TRANSCRIPT_SECRET = "confidential transcript words"


# ── Fakes and helpers ────────────────────────────────────────────────────────


class _Close:
    def __init__(self, code: int, reason: str = "") -> None:
        self.code = code
        self.reason = reason


class FakeConnection:
    """Scripted socket: records frames and replays pushed server frames."""

    def __init__(self, *, close_on_close_stream: bool = True) -> None:
        self.sent: List[str | bytes] = []
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.send_gate: Optional[asyncio.Event] = None
        self.on_text: Optional[Callable[[Dict[str, Any]], None]] = None
        self.close_on_close_stream = close_on_close_stream

    async def send(self, message: str | bytes) -> None:
        if self.closed:
            raise SocketClosedError(1006)
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent.append(message)
        if not isinstance(message, str):
            return
        payload = json.loads(message)
        if self.on_text is not None:
            self.on_text(payload)
        if payload["type"] == "CloseStream" and self.close_on_close_stream:
            # Deepgram flushes, sends Metadata, then closes the socket.
            self.push({"type": "Metadata", "request_id": "r1"})
            self.drop(1000)

    async def recv(self) -> str | bytes:
        item = await self.incoming.get()
        if isinstance(item, _Close):
            self.closed = True
            raise SocketClosedError(item.code, item.reason)
        return item

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(_Close(1000))

    def push(self, event: Dict[str, Any] | str | bytes) -> None:
        self.incoming.put_nowait(event if isinstance(event, (str, bytes)) else json.dumps(event))

    def drop(self, code: int, reason: str = "") -> None:
        self.incoming.put_nowait(_Close(code, reason))

    def text_types(self) -> List[str]:
        return [json.loads(m)["type"] for m in self.sent if isinstance(m, str)]

    def audio(self) -> List[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


class FakeConnector:
    def __init__(self, *plan: Any) -> None:
        self.plan: Deque[Any] = deque(plan)
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.opened: List[FakeConnection] = []

    async def __call__(self, url: str, headers: Dict[str, str], timeout: float) -> FakeConnection:
        self.calls.append((url, dict(headers)))
        item = self.plan.popleft() if self.plan else FakeConnection()
        if item == "hang":
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        self.opened.append(item)
        return item


class TickClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_settings(**overrides: Any) -> DeepgramSettings:
    values: Dict[str, Any] = {
        "endpoint": ENDPOINT,
        "model": "nova-3",
        "api_key": API_KEY,
        "sample_rate": 16000,
    }
    values.update(overrides)
    return DeepgramSettings(**values)


def make_transport(**overrides: Any) -> DeepgramTransportOptions:
    values: Dict[str, Any] = {
        "handshake_timeout_s": 1.0,
        "close_timeout_s": 0.5,
        "terminate_timeout_s": 1.0,
        "keepalive_interval_s": 3600.0,
        "keepalive_poll_s": 0.005,
    }
    values.update(overrides)
    return DeepgramTransportOptions(**values)


def make_provider(connector: FakeConnector, **kwargs: Any) -> DeepgramSTTProvider:
    settings = kwargs.pop("settings", make_settings())
    transport = kwargs.pop("transport", make_transport())
    return DeepgramSTTProvider(settings, transport=transport, connect_factory=connector, **kwargs)


def make_frame(stream_id: str = "default", seq: int = 0, samples: int = 320) -> AudioFrame:
    return AudioFrame(stream_id, seq, struct.pack("<h", seq) * samples, sample_rate=16000)


def results(
    text: str,
    start: float,
    duration: float,
    *,
    is_final: bool,
    speech_final: bool = False,
    from_finalize: bool = False,
) -> Dict[str, Any]:
    words = [
        {"word": w.lower(), "punctuated_word": w, "start": start, "end": start + duration}
        for w in text.split()
    ]
    return {
        "type": "Results",
        "channel_index": [0, 1],
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "speech_final": speech_final,
        "from_finalize": from_finalize,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.9, "words": words}]},
    }


def parsed(event: Dict[str, Any]) -> DeepgramResults:
    result = parse_message(json.dumps(event))
    assert isinstance(result, DeepgramResults)
    return result


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


async def next_event(events: Any, timeout: float = 2.0) -> STTTranscriptResult:
    return await asyncio.wait_for(events.__anext__(), timeout)


def live_adapter_tasks() -> List["asyncio.Task[Any]"]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("deepgram-stt") and not task.done()
    ]


# ── Protocol: settings, URL, auth ────────────────────────────────────────────


def test_connection_url_carries_resolved_settings_and_omits_unset_options():
    url = build_connection_url(make_settings())
    assert url == (
        f"{ENDPOINT}?model=nova-3&encoding=linear16&sample_rate=16000&channels=1"
        "&interim_results=true"
    )
    full = build_connection_url(
        make_settings(
            language="multi",
            endpointing_ms=300,
            utterance_end_ms=1000,
            smart_format=True,
            punctuate=False,
            keyterms=("AC-42", KEYTERM),
        )
    )
    assert full.endswith(
        "&language=multi&endpointing=300&utterance_end_ms=1000&smart_format=true"
        "&punctuate=false&keyterm=AC-42&keyterm=Nightjar"
    )


def test_auth_headers_use_token_for_keys_and_bearer_for_temporary_tokens():
    assert build_auth_headers(make_settings()) == {"Authorization": f"Token {API_KEY}"}
    assert build_auth_headers(make_settings(auth_scheme="bearer")) == {
        "Authorization": f"Bearer {API_KEY}"
    }


def test_settings_repr_hides_key_and_keyterms():
    text = repr(make_settings(keyterms=(KEYTERM,)))
    assert API_KEY not in text
    assert KEYTERM not in text


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"model": ""}, ProviderErrorCode.INVALID_REQUEST),
        ({"api_key": " "}, ProviderErrorCode.AUTH_FAILED),
        ({"sample_rate": 0}, ProviderErrorCode.INVALID_REQUEST),
        ({"utterance_end_ms": 500}, ProviderErrorCode.INVALID_REQUEST),
        ({"auth_scheme": "basic"}, ProviderErrorCode.INVALID_REQUEST),
        ({"endpoint": "ws://listen.example.test/v1/listen"}, ProviderErrorCode.INVALID_REQUEST),
        (
            {"endpoint": f"wss://u:{API_KEY}@listen.example.test/"},
            ProviderErrorCode.INVALID_REQUEST,
        ),
    ],
)
async def test_invalid_settings_fail_before_any_connection(overrides, code):
    connector = FakeConnector()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        make_provider(connector, settings=make_settings(**overrides))
    assert exc_info.value.error.code == code
    assert API_KEY not in str(exc_info.value)
    assert connector.calls == []


def test_shared_redaction_helpers():
    url = f"{ENDPOINT}?model=nova-3&token={API_KEY}"
    assert redact_url(url) == ENDPOINT
    text = redact_text(f"Authorization: Token {API_KEY} and key={API_KEY}x", (API_KEY,))
    assert API_KEY not in text
    assert redact_text("Bearer abcdefghijklmnop") == "Bearer ***"
    assert validate_endpoint("ws://127.0.0.1:9000/listen") == "ws://127.0.0.1:9000/listen"
    for bad in ("", "https://x.test/", "ws://remote.test/", "wss:///path"):
        with pytest.raises(ValueError):
            validate_endpoint(bad)


def test_shared_utility_is_provider_neutral():
    source = Path(streaming_ws.__file__).read_text(encoding="utf-8")
    for provider_term in (
        "Deepgram",
        "AssemblyAI",
        "KeepAlive",
        "Finalize",
        "CloseStream",
        "Terminate",
        "speech_final",
        "end_of_turn",
        "fallback(",
    ):
        assert provider_term not in source


# ── Protocol: parsing and error mapping ──────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        b"\xff\xfe",
        "[1]",
        json.dumps({"channel": {}}),
        json.dumps({"type": "Results", "is_final": True}),
        json.dumps({"type": "Results", "channel": {"alternatives": []}, "is_final": True}),
        json.dumps(
            {
                "type": "Results",
                "channel": {"alternatives": [{"transcript": "x"}]},
                "is_final": "yes",
                "start": 0,
                "duration": 1,
            }
        ),
        json.dumps({**results("x", 0, 1, is_final=True), "start": -1}),
    ],
)
def test_malformed_messages_never_raise(raw):
    assert isinstance(parse_message(raw), DeepgramMalformed)


def test_results_parse_words_confidence_and_flags():
    event = parsed(results("Hello there", 1.5, 0.5, is_final=True, speech_final=True))
    assert (event.is_final, event.speech_final, event.from_finalize) == (True, True, False)
    assert event.words[0].word == "Hello"
    assert event.words[0].start_time_ms == 1500.0
    assert event.confidence == 0.9


@pytest.mark.parametrize(
    ("code", "reason", "expected", "retryable", "identifier"),
    [
        (1011, "NET-0001: no audio received", ProviderErrorCode.TIMEOUT, True, "NET-0001"),
        (1011, "NET-0000", ProviderErrorCode.TIMEOUT, True, "NET-0000"),
        (1008, "DATA-0000: bad audio", ProviderErrorCode.INVALID_REQUEST, False, "DATA-0000"),
        (1011, "", ProviderErrorCode.INTERNAL_ERROR, True, None),
        (1006, "", ProviderErrorCode.CONNECTION_CLOSED, True, None),
    ],
)
def test_close_frames_map_to_redacted_canonical_errors(
    code, reason, expected, retryable, identifier
):
    error = map_close(code, f"{reason} key={API_KEY}")
    assert (error.code, error.retryable) == (expected, retryable)
    assert error.capability == ProviderCapability.STT
    assert error.details == (
        {"close_code": code, "provider_code": identifier} if identifier else {"close_code": code}
    )
    assert API_KEY not in f"{error} {error.details}"


# ── Protocol: finality and duplicates ────────────────────────────────────────


def test_is_final_segments_accumulate_until_speech_final():
    assembler = DeepgramUtteranceAssembler("mic", clock=TickClock())

    out: List[STTTranscriptResult] = []
    for event in (
        results("hello", 0.0, 0.5, is_final=False),
        results("hello there", 0.0, 1.0, is_final=True),
        results("how", 1.0, 0.3, is_final=False),
        results("how are you", 1.0, 1.0, is_final=True, speech_final=True),
        results("next", 2.0, 0.4, is_final=False),
    ):
        out.extend(assembler.on_results(parsed(event)))

    assert [(r.sequence, r.text, r.is_final, r.revision) for r in out] == [
        (0, "hello", False, 1),
        (0, "hello there", False, 2),
        (0, "hello there how", False, 3),
        (0, "hello there how are you", True, 4),
        (1, "next", False, 1),
    ]
    final = out[3]
    assert (final.start_time_ms, final.end_time_ms, final.is_endpoint) == (0.0, 2000.0, True)
    assert final.provider_event_id == "deepgram:mic:0"
    assert final.words is not None and len(final.words) == 5
    assert out[0].words is None


def test_duplicate_and_replayed_events_never_regress_or_refinalize():
    assembler = DeepgramUtteranceAssembler("mic", clock=TickClock())
    first = assembler.on_results(parsed(results("one", 0.0, 0.5, is_final=False)))
    assert len(first) == 1
    assert assembler.on_results(parsed(results("one", 0.0, 0.5, is_final=False))) == []

    final = assembler.on_results(
        parsed(results("one two", 0.0, 1.0, is_final=True, speech_final=True))
    )
    assert [r.is_final for r in final] == [True]

    # Replayed final, stale interim, and endpoint signals with nothing open are inert.
    assert (
        assembler.on_results(parsed(results("one two", 0.0, 1.0, is_final=True, speech_final=True)))
        == []
    )
    assert assembler.on_results(parsed(results("one", 0.0, 0.5, is_final=False))) == []
    assert assembler.on_utterance_end() is None
    assert assembler.flush() is None


def test_from_finalize_and_utterance_end_close_the_utterance_once():
    assembler = DeepgramUtteranceAssembler("mic", clock=TickClock())
    assembler.on_results(parsed(results("first part", 0.0, 1.0, is_final=True)))
    # Finalize with no further audio: zero-duration result at the same position.
    closed = assembler.on_results(parsed(results("", 1.0, 0.0, is_final=True, from_finalize=True)))
    assert [(r.text, r.is_final, r.revision) for r in closed] == [("first part", True, 2)]

    assembler.on_results(parsed(results("second", 1.0, 0.5, is_final=True)))
    ended = assembler.on_utterance_end()
    assert ended is not None and (ended.sequence, ended.text, ended.is_final) == (1, "second", True)
    assert assembler.on_utterance_end() is None


def test_retracted_partial_ends_with_empty_final_and_flush_promotes_interim():
    assembler = DeepgramUtteranceAssembler("mic", clock=TickClock())
    assembler.on_results(parsed(results("uh", 0.0, 0.3, is_final=False)))
    retracted = assembler.on_results(
        parsed(results("", 0.0, 0.5, is_final=True, speech_final=True))
    )
    assert [(r.text, r.is_final) for r in retracted] == [("", True)]

    assembler.on_results(parsed(results("trailing words", 0.5, 0.5, is_final=False)))
    flushed = assembler.flush()
    assert flushed is not None and (flushed.text, flushed.is_final) == ("trailing words", True)


def test_reset_discards_open_utterance_and_results_for_earlier_audio():
    assembler = DeepgramUtteranceAssembler("mic", clock=TickClock())
    assert assembler.on_results(parsed(results("stale", 0.0, 1.0, is_final=False)))
    assembler.reset(discard_before_ms=1000.0)

    assert (
        assembler.on_results(
            parsed(results("stale words", 0.0, 1.0, is_final=True, speech_final=True))
        )
        == []
    )
    fresh = assembler.on_results(parsed(results("fresh", 1.0, 0.5, is_final=False)))
    assert [(r.sequence, r.text, r.revision) for r in fresh] == [(1, "fresh", 1)]


# ── Shared socket utility ────────────────────────────────────────────────────


async def test_socket_keepalive_is_idle_only_text_and_uses_injected_clock():
    conn = FakeConnection()
    clock = ManualClock()
    socket = StreamingSocket(
        conn,
        label="unit-socket",
        on_message=lambda raw: None,
        on_closed=lambda exc: None,
        max_queued_audio=4,
        keepalive_message='{"ping":1}',
        keepalive_interval_s=4.0,
        keepalive_poll_s=0.002,
        clock=clock,
    )
    socket.start()
    try:
        await asyncio.sleep(0.03)
        assert conn.sent == []

        clock.now = 5.0
        await wait_until(lambda: conn.sent == ['{"ping":1}'])
        await asyncio.sleep(0.03)
        assert socket.keepalives_sent == 1

        socket.enqueue_audio(b"\x00\x00")
        await wait_until(lambda: len(conn.sent) == 2)
        clock.now = 8.0
        await asyncio.sleep(0.03)
        assert socket.keepalives_sent == 1

        clock.now = 9.5
        await wait_until(lambda: socket.keepalives_sent == 2 and len(conn.sent) == 3)
        assert isinstance(conn.sent[-1], str)
    finally:
        await socket.close()
    assert [t for t in asyncio.all_tasks() if t.get_name().startswith("unit-socket")] == []


async def test_socket_clear_audio_keeps_control_and_close_is_idempotent():
    conn = FakeConnection()
    conn.send_gate = asyncio.Event()
    closes: List[SocketClosedError] = []
    socket = StreamingSocket(
        conn,
        label="unit-socket",
        on_message=lambda raw: None,
        on_closed=closes.append,
        max_queued_audio=10,
    )
    socket.start()
    socket.enqueue_audio(b"\x01\x00")
    await asyncio.sleep(0.01)
    socket.enqueue_audio(b"\x02\x00")
    socket.enqueue_control("control")
    socket.enqueue_audio(b"\x03\x00")
    assert socket.clear_audio() == 2
    conn.send_gate.set()
    assert await socket.drain(1.0)
    assert conn.sent == [b"\x01\x00", "control"]

    await socket.close()
    await socket.close()
    assert conn.closed and closes == []
    assert not socket.enqueue_audio(b"\x00\x00")


# ── Adapter: streaming ───────────────────────────────────────────────────────


async def test_start_authenticates_and_streams_binary_audio():
    connector = FakeConnector()
    provider = make_provider(connector, settings=make_settings(language="en"))
    assert provider.capability == ProviderCapability.STT
    await provider.start()
    try:
        url, headers = connector.calls[0]
        assert url.startswith(f"{ENDPOINT}?model=nova-3&") and "language=en" in url
        assert headers == {"Authorization": f"Token {API_KEY}"}
        conn = connector.opened[0]
        frame = make_frame(seq=7)
        await provider.send_audio(frame)
        await wait_until(lambda: conn.audio() == [frame.pcm_data])
    finally:
        await provider.stop()


async def test_finalize_sends_text_control_and_maps_from_finalize_result():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic",))
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.finalize_stream("mic")
        await provider.send_audio(make_frame("mic"))
        conn.push(results("hel", 0.0, 0.02, is_final=False))
        partial = await next_event(events)
        await provider.finalize_stream("mic")
        await provider.finalize_stream("mic")
        await wait_until(lambda: conn.text_types() == ["Finalize"])

        conn.push(results("hello", 0.0, 0.02, is_final=True, from_finalize=True))
        final = await next_event(events)
        assert (partial.stream_id, partial.is_final, partial.revision) == ("mic", False, 1)
        assert (final.text, final.is_final, final.revision, final.is_endpoint) == (
            "hello",
            True,
            2,
            True,
        )
    finally:
        await provider.stop()


async def test_keepalive_sent_during_silence_per_injected_clock():
    clock = ManualClock()
    connector = FakeConnector()
    provider = make_provider(
        connector,
        transport=make_transport(keepalive_interval_s=4.0, keepalive_poll_s=0.002),
        clock=clock,
    )
    await provider.start()
    conn = connector.opened[0]
    try:
        await asyncio.sleep(0.02)
        assert conn.text_types() == []
        clock.now = 4.5
        await wait_until(lambda: conn.text_types() == ["KeepAlive"])
        await provider.send_audio(make_frame())
        await wait_until(lambda: len(conn.audio()) == 1)
        clock.now = 6.0
        await asyncio.sleep(0.02)
        assert conn.text_types() == ["KeepAlive"]
    finally:
        await provider.stop()


async def test_send_audio_never_blocks_and_reports_backpressure():
    connector = FakeConnector()
    provider = make_provider(connector, transport=make_transport(max_queued_frames=3))
    await provider.start()
    conn = connector.opened[0]
    conn.send_gate = asyncio.Event()
    try:
        for seq in range(10):
            await asyncio.wait_for(provider.send_audio(make_frame(seq=seq)), 0.05)
        await provider.finalize_stream("default")

        state = provider.send_queue_state("default")
        assert state.backpressured and state.dropped_audio == 6 and state.queued_audio == 3

        conn.send_gate.set()
        await wait_until(lambda: "Finalize" in conn.text_types())
        seqs = [struct.unpack("<h", pcm[:2])[0] for pcm in conn.audio()]
        assert seqs == [0, 7, 8, 9]
        assert isinstance(conn.sent[-1], str)
        assert not provider.send_queue_state("default").backpressured
    finally:
        await provider.stop()


async def test_reset_stream_drops_queued_audio_and_ignores_stale_results():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame())
        await wait_until(lambda: len(conn.audio()) == 1)
        conn.send_gate = asyncio.Event()
        await provider.send_audio(make_frame(seq=1))
        await provider.send_audio(make_frame(seq=2))
        await asyncio.sleep(0.01)

        await provider.reset_stream("default")
        conn.send_gate.set()
        await provider.finalize_stream("default")
        await asyncio.sleep(0.02)
        assert len(conn.audio()) == 2
        assert conn.text_types() == []

        # 40 ms were dequeued before the reset; results covering them are dropped.
        conn.push(results("stale words", 0.0, 0.04, is_final=True, speech_final=True))
        conn.push(results("fresh", 0.04, 0.02, is_final=False))
        fresh = await next_event(events)
        assert (fresh.text, fresh.is_final) == ("fresh", False)
    finally:
        await provider.stop()


async def test_multiple_streams_use_isolated_sockets():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic", "system"))
    await provider.start()
    mic, system = connector.opened
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame("system"))
        await wait_until(lambda: len(system.audio()) == 1)
        assert mic.audio() == []
        system.push(results("from system", 0.0, 0.02, is_final=False))
        result = await next_event(events)
        assert (result.stream_id, result.provider_event_id) == ("system", "deepgram:system:0")

        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(make_frame("speaker"))
        assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST
        with pytest.raises(ProviderRuntimeError):
            await provider.send_audio(AudioFrame("mic", 0, b"\x00\x00", sample_rate=48000))
    finally:
        await provider.stop()


# ── Adapter: errors ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (401, ProviderErrorCode.AUTH_FAILED, False),
        (402, ProviderErrorCode.QUOTA_EXCEEDED, False),
        (429, ProviderErrorCode.RATE_LIMITED, True),
        (400, ProviderErrorCode.INVALID_REQUEST, False),
        (503, ProviderErrorCode.INTERNAL_ERROR, True),
    ],
)
async def test_handshake_rejection_maps_status(status, expected, retryable):
    provider = make_provider(FakeConnector(HandshakeRejectedError(status)))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    error = exc_info.value.error
    assert (error.code, error.retryable, error.details) == (
        expected,
        retryable,
        {"http_status": status},
    )
    assert live_adapter_tasks() == []


async def test_connect_timeout_and_network_failure_are_redacted():
    provider = make_provider(
        FakeConnector("hang"), transport=make_transport(handshake_timeout_s=0.05)
    )
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.TIMEOUT

    provider = make_provider(FakeConnector(OSError(f"cannot reach {ENDPOINT}?token={API_KEY}")))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.NETWORK_UNAVAILABLE
    assert API_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


async def test_server_timeout_close_flushes_pending_text_then_surfaces_retryable_error():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame())
        conn.push(results("kept words", 0.0, 0.02, is_final=True))
        assert (await next_event(events)).is_final is False

        conn.drop(1011, f"NET-0001 no audio; Authorization: Token {API_KEY}")
        flushed = await next_event(events)
        assert (flushed.text, flushed.is_final) == ("kept words", True)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(events)
        error = exc_info.value.error
        assert (error.code, error.retryable) == (ProviderErrorCode.TIMEOUT, True)
        assert error.details == {"close_code": 1011, "provider_code": "NET-0001"}
        assert API_KEY not in f"{exc_info.value} {error.details}"

        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(make_frame())
        assert exc_info.value.error.code == ProviderErrorCode.TIMEOUT
    finally:
        await provider.stop()
    assert live_adapter_tasks() == []


async def test_in_band_error_and_malformed_frames():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        conn.push("{not json")
        conn.push({"type": "SpeechStarted", "channel": [0], "timestamp": 0.1})
        conn.push({"type": "Metadata", "request_id": "r1"})
        conn.push(results("still streaming", 0.0, 0.1, is_final=False))
        assert (await next_event(events)).text == "still streaming"

        conn.push({"type": "Error", "variant": "DATA-0000", "description": f"bad key {API_KEY}"})
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(events)
        assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST
        assert API_KEY not in str(exc_info.value.error)
    finally:
        await provider.stop()


# ── Adapter: termination and cleanup ─────────────────────────────────────────


async def test_stop_sends_close_stream_after_audio_and_waits_for_flush():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic", "system"))
    await provider.start()
    mic = connector.opened[0]

    def server(message: Dict[str, Any]) -> None:
        if message["type"] == "CloseStream":
            mic.push(results("last words", 0.0, 0.02, is_final=True, speech_final=True))

    mic.on_text = server
    received: List[STTTranscriptResult] = []

    async def consume() -> None:
        async for result in provider.receive_events():
            received.append(result)

    consumer = asyncio.ensure_future(consume())
    await provider.send_audio(make_frame("mic"))
    await provider.stop()
    await asyncio.wait_for(consumer, 1.0)

    assert isinstance(mic.sent[0], bytes)
    assert mic.text_types() == ["CloseStream"]
    assert [(r.text, r.is_final) for r in received] == [("last words", True)]
    assert all(conn.closed for conn in connector.opened)
    assert live_adapter_tasks() == []
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(make_frame("mic"))
    assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST


async def test_stop_closes_even_when_server_never_acknowledges(caplog):
    caplog.set_level(logging.WARNING, logger="echoflux")
    connector = FakeConnector(FakeConnection(close_on_close_stream=False))
    provider = make_provider(connector, transport=make_transport(terminate_timeout_s=0.05))
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    conn.push(results("pending", 0.0, 0.5, is_final=False))
    assert (await next_event(events)).is_final is False

    await provider.stop()
    flushed = await next_event(events)
    assert (flushed.text, flushed.is_final) == ("pending", True)
    with pytest.raises(StopAsyncIteration):
        await next_event(events)
    assert conn.closed and conn.text_types() == ["CloseStream"]
    assert "did not close after CloseStream" in caplog.text
    assert live_adapter_tasks() == []


async def test_cancelled_start_closes_opened_socket_and_leaks_no_task():
    first = FakeConnection()
    connector = FakeConnector(first, "hang")
    provider = make_provider(connector, stream_ids=("mic", "system"))

    start = asyncio.ensure_future(provider.start())
    await wait_until(lambda: len(connector.calls) == 2)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert first.closed
    assert live_adapter_tasks() == []


async def test_cancelled_consumer_does_not_leak_after_stop():
    connector = FakeConnector()
    provider = make_provider(connector, transport=make_transport(terminate_timeout_s=0.05))
    await provider.start()

    async def consume() -> None:
        async for _ in provider.receive_events():
            pass

    consumer = asyncio.ensure_future(consume())
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await provider.stop()
    assert connector.opened[0].closed
    assert live_adapter_tasks() == []


async def test_health_check_reports_status_without_leaking_secrets():
    conn = FakeConnection()
    provider = make_provider(FakeConnector(conn))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.HEALTHY
    assert conn.closed and conn.text_types() == ["CloseStream"]

    provider = make_provider(FakeConnector(HandshakeRejectedError(401)))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "auth_failed", "retryable": False}
    assert API_KEY not in health.message


async def test_logs_never_contain_secret_query_audio_or_transcript(caplog):
    caplog.set_level(logging.DEBUG, logger="echoflux")
    connector = FakeConnector()
    provider = make_provider(connector, settings=make_settings(keyterms=(KEYTERM,)))
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    frame = make_frame(seq=1234)
    try:
        await provider.send_audio(frame)
        conn.push(results(TRANSCRIPT_SECRET, 0.0, 0.5, is_final=False))
        await next_event(events)
        conn.push("{garbage")
        conn.drop(1008, f"DATA-0000 {API_KEY} {TRANSCRIPT_SECRET}")
        await next_event(events)
        with pytest.raises(ProviderRuntimeError):
            await next_event(events)
    finally:
        await provider.stop()

    text = caplog.text
    assert caplog.records
    for forbidden in (API_KEY, KEYTERM, TRANSCRIPT_SECRET, "keyterm=", "Token ", "sample_rate="):
        assert forbidden not in text


# ── Adapter against a local WebSocket server ─────────────────────────────────


@pytest.fixture
def no_proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


async def test_local_websocket_server_end_to_end(no_proxy_env):
    from websockets.asyncio.server import serve

    request: Dict[str, Any] = {}
    frames: List[str] = []
    server_closed = asyncio.Event()

    async def handler(ws: Any) -> None:
        request["authorization"] = ws.request.headers.get("Authorization")
        request["path"] = ws.request.path
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    frames.append("audio")
                    await ws.send(json.dumps(results("local", 0.0, 0.02, is_final=False)))
                    continue
                kind = json.loads(message)["type"]
                frames.append(kind)
                if kind == "Finalize":
                    await ws.send(
                        json.dumps(
                            results("local server", 0.0, 0.02, is_final=True, from_finalize=True)
                        )
                    )
                elif kind == "CloseStream":
                    await ws.send(json.dumps({"type": "Metadata", "request_id": "r1"}))
                    await ws.close()
        finally:
            server_closed.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        provider = DeepgramSTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v1/listen"),
            transport=make_transport(),
        )
        await provider.start()
        events = provider.receive_events()
        try:
            await provider.send_audio(make_frame())
            partial = await next_event(events)
            await provider.finalize_stream("default")
            final = await next_event(events)
        finally:
            await provider.stop()
        await asyncio.wait_for(server_closed.wait(), 2.0)

    assert request["authorization"] == f"Token {API_KEY}"
    assert request["path"].startswith("/v1/listen?model=nova-3&encoding=linear16")
    assert frames == ["audio", "Finalize", "CloseStream"]
    assert (partial.text, partial.is_final) == ("local", False)
    assert (final.text, final.is_final, final.revision) == ("local server", True, 2)
    assert live_adapter_tasks() == []


async def test_local_websocket_server_http_rejection_maps_to_auth_failed(no_proxy_env):
    from websockets.asyncio.server import serve

    def reject(connection: Any, request: Any) -> Any:
        return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    async def handler(ws: Any) -> None:  # pragma: no cover - never reached
        await ws.close()

    async with serve(handler, "127.0.0.1", 0, process_request=reject) as server:
        port = server.sockets[0].getsockname()[1]
        provider = DeepgramSTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v1/listen"),
            transport=make_transport(),
        )
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert exc_info.value.error.details == {"http_status": 401}
    assert live_adapter_tasks() == []
