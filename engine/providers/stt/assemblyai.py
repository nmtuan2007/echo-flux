"""AssemblyAI Universal Streaming (v3) STT adapter.

Protocol as documented on 2026-09-14 (``wss://streaming.assemblyai.com/v3/ws``, with
``streaming.us``/``streaming.eu`` residency hosts):

- Auth: ``Authorization: <api key>`` (no scheme), or ``token=<temporary token>`` in the
  query for temporary tokens.
- Query: ``sample_rate``, ``encoding=pcm_s16le``, ``speech_model``, plus optional
  ``format_turns``, ``language_detection``, ``language_codes``, ``keyterms_prompt``,
  ``end_of_turn_confidence_threshold``, ``min_turn_silence``, ``max_turn_silence``,
  ``inactivity_timeout``.
- Audio: binary PCM16 chunks of 50-1000 ms. Control: ``ForceEndpoint``, ``Terminate``.
- Server: ``Begin``, ``Turn`` (``turn_order``, ``transcript``, ``end_of_turn``,
  ``turn_is_formatted``, ``words``), ``Termination``; ``SpeechStarted``/``Heartbeat`` are
  ignored. Errors arrive as close codes (1008 auth/account, 3005 server, 3006 invalid
  message or inactivity, 3007 chunk duration/rate, 3008 session expired, 3009
  concurrency, 1011 internal).

Finality. A ``Turn`` with ``end_of_turn=false`` is a partial. ``end_of_turn=true`` ends the
turn; when ``format_turns`` was requested, the formatted copy (``turn_is_formatted``) is
the final and the unformatted end-of-turn stays a partial.

Billing. Sessions that are not terminated stay open (and billed) until they time out, so
``stop`` and failed starts always send ``Terminate`` and wait for ``Termination``.

The adapter is dormant: nothing registers it, it never chooses fallback, and it does not
reconnect. Nothing here logs the key, token, query string, audio, or transcript text.
"""

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.stt.streaming_ws import (
    ConnectFactory,
    HandshakeRejectedError,
    ResultChannel,
    SendQueueState,
    SocketClosedError,
    SocketConnectError,
    StreamingSocket,
    WireMessage,
    build_url,
    classify_handshake_status,
    make_stt_error,
    make_websocket_factory,
    open_connection,
    redact_url,
    validate_endpoint,
)
from engine.providers.types import (
    AudioFrame,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
    WordTiming,
)

logger = get_logger("providers.stt.assemblyai")

LABEL = "AssemblyAI"
AUTH_MODE_API_KEY = "api_key"
AUTH_MODE_TEMPORARY_TOKEN = "temporary_token"

TERMINATE_MESSAGE = json.dumps({"type": "Terminate"}, separators=(",", ":"))
FORCE_ENDPOINT_MESSAGE = json.dumps({"type": "ForceEndpoint"}, separators=(",", ":"))

MIN_CHUNK_MS = 50.0
MAX_CHUNK_MS = 1000.0
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 96000
MAX_KEYTERMS = 100
_EPSILON_MS = 1.0

CLOSE_SERVER_ERROR = 3005
CLOSE_INVALID_OR_INACTIVE = 3006
CLOSE_CHUNK_VIOLATION = 3007
CLOSE_SESSION_EXPIRED = 3008
CLOSE_TOO_MANY_SESSIONS = 3009


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssemblyAISettings:
    """Resolved settings. Endpoint, speech model, and sample rate have no adapter defaults.

    Optional features are sent only when set, so AssemblyAI's own defaults apply
    otherwise. ``api_key`` is a permanent key in ``api_key`` mode, or a temporary token.
    """

    endpoint: str
    speech_model: str
    api_key: str = field(repr=False)
    sample_rate: int
    auth_mode: str = AUTH_MODE_API_KEY
    format_turns: Optional[bool] = None
    language_detection: Optional[bool] = None
    language_codes: Tuple[str, ...] = ()
    keyterms_prompt: Tuple[str, ...] = field(default=(), repr=False)
    end_of_turn_confidence_threshold: Optional[float] = None
    min_turn_silence_ms: Optional[int] = None
    max_turn_silence_ms: Optional[int] = None
    inactivity_timeout_s: Optional[int] = None


@dataclass(frozen=True)
class AssemblyAITransportOptions:
    """Transport tuning only; provider behaviour lives in ``AssemblyAISettings``."""

    handshake_timeout_s: float = 10.0
    close_timeout_s: float = 2.0
    terminate_timeout_s: float = 5.0
    max_queued_frames: int = 100
    # Audio is re-chunked to this size; AssemblyAI accepts 50-1000 ms per message.
    chunk_ms: float = 100.0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _invalid(field_name: str) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        make_stt_error(LABEL, ProviderErrorCode.INVALID_REQUEST, details={"field": field_name})
    )


def _check_int_range(value: Optional[int], low: int, high: int, name: str) -> None:
    if value is not None and (not _is_int(value) or not low <= value <= high):
        raise _invalid(name)


def validate_settings(settings: AssemblyAISettings) -> None:
    """Reject unusable settings before any network activity."""
    if not isinstance(settings.speech_model, str) or not settings.speech_model.strip():
        raise _invalid("speech_model")
    if not isinstance(settings.api_key, str) or not settings.api_key.strip():
        raise ProviderRuntimeError(
            make_stt_error(LABEL, ProviderErrorCode.AUTH_FAILED, details={"field": "api_key"})
        )
    if settings.auth_mode not in (AUTH_MODE_API_KEY, AUTH_MODE_TEMPORARY_TOKEN):
        raise _invalid("auth_mode")
    _check_int_range(settings.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE, "sample_rate")
    if settings.sample_rate is None:
        raise _invalid("sample_rate")
    threshold = settings.end_of_turn_confidence_threshold
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise _invalid("end_of_turn_confidence_threshold")
    _check_int_range(settings.min_turn_silence_ms, 50, 10000, "min_turn_silence_ms")
    _check_int_range(settings.max_turn_silence_ms, 50, 60000, "max_turn_silence_ms")
    if (
        settings.min_turn_silence_ms is not None
        and settings.max_turn_silence_ms is not None
        and settings.max_turn_silence_ms < settings.min_turn_silence_ms
    ):
        raise _invalid("max_turn_silence_ms")
    _check_int_range(settings.inactivity_timeout_s, 5, 3600, "inactivity_timeout_s")
    if len(settings.keyterms_prompt) > MAX_KEYTERMS:
        raise _invalid("keyterms_prompt")
    for name, values in (
        ("keyterms_prompt", settings.keyterms_prompt),
        ("language_codes", settings.language_codes),
    ):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise _invalid(name)
    try:
        validate_endpoint(settings.endpoint)
    except ValueError:
        raise _invalid("endpoint") from None


def validate_transport(options: AssemblyAITransportOptions) -> None:
    if not MIN_CHUNK_MS <= options.chunk_ms <= MAX_CHUNK_MS:
        raise _invalid("chunk_ms")
    if options.max_queued_frames < 1:
        raise _invalid("max_queued_frames")


def _flag(value: bool) -> str:
    return "true" if value else "false"


def build_query(settings: AssemblyAISettings) -> List[Tuple[str, str]]:
    params: List[Tuple[str, str]] = [
        ("sample_rate", str(settings.sample_rate)),
        ("encoding", "pcm_s16le"),
        ("speech_model", settings.speech_model),
    ]
    if settings.format_turns is not None:
        params.append(("format_turns", _flag(settings.format_turns)))
    if settings.language_detection is not None:
        params.append(("language_detection", _flag(settings.language_detection)))
    if settings.language_codes:
        params.append(("language_codes", json.dumps(list(settings.language_codes))))
    if settings.keyterms_prompt:
        params.append(("keyterms_prompt", json.dumps(list(settings.keyterms_prompt))))
    if settings.end_of_turn_confidence_threshold is not None:
        params.append(
            ("end_of_turn_confidence_threshold", str(settings.end_of_turn_confidence_threshold))
        )
    if settings.min_turn_silence_ms is not None:
        params.append(("min_turn_silence", str(settings.min_turn_silence_ms)))
    if settings.max_turn_silence_ms is not None:
        params.append(("max_turn_silence", str(settings.max_turn_silence_ms)))
    if settings.inactivity_timeout_s is not None:
        params.append(("inactivity_timeout", str(settings.inactivity_timeout_s)))
    if settings.auth_mode == AUTH_MODE_TEMPORARY_TOKEN:
        params.append(("token", settings.api_key))
    return params


def build_connection_url(settings: AssemblyAISettings) -> str:
    return build_url(settings.endpoint, build_query(settings))


def build_auth_headers(settings: AssemblyAISettings) -> Dict[str, str]:
    if settings.auth_mode == AUTH_MODE_TEMPORARY_TOKEN:
        return {}
    return {"Authorization": settings.api_key}


# ── Server events ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssemblyAIBegin:
    session_id: Optional[str] = None
    expires_at: Optional[float] = None


@dataclass(frozen=True)
class AssemblyAITurn:
    turn_order: int
    transcript: str
    end_of_turn: bool
    turn_is_formatted: bool = False
    words: Tuple[WordTiming, ...] = ()
    language_code: Optional[str] = None
    end_of_turn_confidence: Optional[float] = None


@dataclass(frozen=True)
class AssemblyAITermination:
    audio_duration_s: Optional[float] = None
    session_duration_s: Optional[float] = None


@dataclass(frozen=True)
class AssemblyAIServerError:
    """``{"error": ...}`` payload. The provider text is dropped; a close frame follows."""


@dataclass(frozen=True)
class AssemblyAIIgnored:
    event_type: str


@dataclass(frozen=True)
class AssemblyAIMalformed:
    reason: str


AssemblyAIEvent = Union[  # noqa: UP007 -- typing house style (rule 04)
    AssemblyAIBegin,
    AssemblyAITurn,
    AssemblyAITermination,
    AssemblyAIServerError,
    AssemblyAIIgnored,
    AssemblyAIMalformed,
]


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_words(raw_words: Any) -> Tuple[WordTiming, ...]:
    if not isinstance(raw_words, list):
        return ()
    words: List[WordTiming] = []
    for raw in raw_words:
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            continue
        start, end = _optional_number(raw.get("start")), _optional_number(raw.get("end"))
        if start is None or end is None:
            continue
        words.append(
            WordTiming(
                word=raw["text"],
                start_time_ms=start,
                end_time_ms=max(start, end),
                confidence=_optional_number(raw.get("confidence")),
            )
        )
    return tuple(words)


def _parse_turn(payload: Dict[str, Any]) -> AssemblyAIEvent:
    order = payload.get("turn_order")
    if not _is_int(order) or order < 0:
        return AssemblyAIMalformed("Turn:turn_order")
    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        return AssemblyAIMalformed("Turn:transcript")
    end_of_turn = payload.get("end_of_turn")
    if not isinstance(end_of_turn, bool):
        return AssemblyAIMalformed("Turn:end_of_turn")
    language = payload.get("language_code")
    return AssemblyAITurn(
        turn_order=order,
        transcript=transcript,
        end_of_turn=end_of_turn,
        turn_is_formatted=payload.get("turn_is_formatted") is True,
        words=_parse_words(payload.get("words")),
        language_code=language if isinstance(language, str) and language else None,
        end_of_turn_confidence=_optional_number(payload.get("end_of_turn_confidence")),
    )


def parse_message(raw: WireMessage) -> AssemblyAIEvent:
    """Parse one server frame. Never raises; bad input becomes ``AssemblyAIMalformed``."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    except (ValueError, TypeError):
        return AssemblyAIMalformed("invalid_json")
    if not isinstance(payload, dict):
        return AssemblyAIMalformed("not_an_object")
    kind = payload.get("type")
    if kind is None and "error" in payload:
        return AssemblyAIServerError()
    if not isinstance(kind, str) or not kind:
        return AssemblyAIMalformed("missing_type")
    if kind == "Turn":
        return _parse_turn(payload)
    if kind == "Begin":
        session_id = payload.get("id")
        return AssemblyAIBegin(
            session_id if isinstance(session_id, str) and session_id else None,
            _optional_number(payload.get("expires_at")),
        )
    if kind == "Termination":
        return AssemblyAITermination(
            _optional_number(payload.get("audio_duration_seconds")),
            _optional_number(payload.get("session_duration_seconds")),
        )
    return AssemblyAIIgnored(kind)


# ── Error mapping ────────────────────────────────────────────────────────────

_ACCOUNT_HINTS = ("balance", "insufficient", "disabled", "payment", "billing", "credit")


def map_close(code: Optional[int], reason: str = "") -> ProviderError:
    """Map a close frame to a canonical error. The reason text never leaves this function."""
    text = (reason or "").lower()
    if code == 1008:
        if any(hint in text for hint in _ACCOUNT_HINTS):
            mapped = ProviderErrorCode.QUOTA_EXCEEDED
        elif "concurrent" in text or "too many" in text:
            mapped = ProviderErrorCode.RATE_LIMITED
        else:
            mapped = ProviderErrorCode.AUTH_FAILED
    elif code == CLOSE_TOO_MANY_SESSIONS:
        mapped = ProviderErrorCode.RATE_LIMITED
    elif code == CLOSE_SESSION_EXPIRED:
        mapped = ProviderErrorCode.CONNECTION_CLOSED
    elif code == CLOSE_CHUNK_VIOLATION:
        mapped = ProviderErrorCode.INVALID_REQUEST
    elif code == CLOSE_INVALID_OR_INACTIVE:
        mapped = (
            ProviderErrorCode.TIMEOUT if "inactiv" in text else ProviderErrorCode.INVALID_REQUEST
        )
    elif code in (CLOSE_SERVER_ERROR, 1011, 1012, 1013):
        mapped = ProviderErrorCode.INTERNAL_ERROR
    elif code in (1003, 1007, 1009):
        mapped = ProviderErrorCode.INVALID_REQUEST
    else:
        mapped = ProviderErrorCode.CONNECTION_CLOSED
    return make_stt_error(LABEL, mapped, details={"close_code": code})


# ── Turn assembly ────────────────────────────────────────────────────────────


@dataclass
class _TurnState:
    turn_order: int
    start_ms: Optional[float] = None
    end_ms: float = 0.0
    text: str = ""
    revision: int = 0
    language: Optional[str] = None


class AssemblyAITurnAssembler:
    """Maps ``Turn`` messages to canonical results for one stream.

    Invariants:

    - ``sequence`` is ``turn_order``; ``provider_event_id`` is stable per turn.
    - ``revision`` strictly increases per turn and the final is the last revision.
    - Exactly one canonical final per turn. Repeated partials, messages for turns already
      final, and turns abandoned by ``reset`` emit nothing.
    - If a later turn starts while an earlier one never received its final, the earlier
      one is finalized with the text already shown so nothing is lost or left open.
    """

    def __init__(
        self,
        stream_id: str,
        *,
        expects_formatting: bool,
        clock: Callable[[], float] = time.monotonic,
        max_finalized: int = 4096,
    ) -> None:
        self.stream_id = stream_id
        self._expects_formatting = expects_formatting
        self._clock = clock
        self._max_finalized = max_finalized
        self._session_key = stream_id
        self._open: OrderedDict[int, _TurnState] = OrderedDict()
        self._finalized: OrderedDict[int, None] = OrderedDict()
        self._discard_below_order = 0
        self._discard_before_ms = 0.0
        self._audio_ms = 0.0

    def set_session(self, session_id: Optional[str]) -> None:
        if session_id:
            self._session_key = session_id

    def set_audio_position(self, audio_ms: float) -> None:
        self._audio_ms = max(self._audio_ms, audio_ms)

    @property
    def has_open_turn(self) -> bool:
        return bool(self._open)

    def on_turn(self, event: AssemblyAITurn) -> List[STTTranscriptResult]:
        order = event.turn_order
        if order < self._discard_below_order or order in self._finalized:
            return []
        if (
            event.words
            and order not in self._open
            and event.words[-1].end_time_ms <= self._discard_before_ms + _EPSILON_MS
        ):
            return []
        results = self._promote_before(order)
        state = self._open.get(order)
        if state is None:
            state = _TurnState(order, start_ms=None, end_ms=self._audio_ms)
            self._open[order] = state
        if event.words:
            first, last = event.words[0].start_time_ms, event.words[-1].end_time_ms
            state.start_ms = first if state.start_ms is None else min(state.start_ms, first)
            state.end_ms = max(state.end_ms, last)
        if event.language_code:
            state.language = event.language_code

        is_final = event.end_of_turn and (event.turn_is_formatted or not self._expects_formatting)
        text = event.transcript
        if is_final:
            final_text = text if text.strip() else state.text
            if final_text or state.revision:
                results.append(self._emit(state, final_text, True, event.words))
            self._retire(order)
        elif text.strip() and text != state.text:
            results.append(self._emit(state, text, False, ()))
        return results

    def flush(self) -> List[STTTranscriptResult]:
        """Finalize every open turn with the text already shown."""
        results: List[STTTranscriptResult] = []
        for order in list(self._open):
            state = self._open[order]
            if state.revision:
                results.append(self._emit(state, state.text, True, ()))
            self._retire(order)
        return results

    def reset(self, discard_before_ms: float) -> None:
        """Abandon open turns and ignore turns that only cover earlier audio."""
        if self._open:
            self._discard_below_order = max(self._discard_below_order, max(self._open) + 1)
        for order in list(self._open):
            self._retire(order)
        self._discard_before_ms = max(self._discard_before_ms, discard_before_ms)

    def _promote_before(self, order: int) -> List[STTTranscriptResult]:
        results: List[STTTranscriptResult] = []
        for earlier in [o for o in self._open if o < order]:
            state = self._open[earlier]
            if state.revision:
                results.append(self._emit(state, state.text, True, ()))
            self._retire(earlier)
        return results

    def _retire(self, order: int) -> None:
        self._open.pop(order, None)
        self._finalized[order] = None
        while len(self._finalized) > self._max_finalized:
            self._finalized.popitem(last=False)

    def _emit(
        self,
        state: _TurnState,
        text: str,
        final: bool,
        words: Tuple[WordTiming, ...],
    ) -> STTTranscriptResult:
        state.revision += 1
        state.text = text
        if state.start_ms is None:
            state.start_ms = state.end_ms
        if not words and not final:
            # Without word timings, a partial covers audio submitted so far.
            state.end_ms = max(state.end_ms, self._audio_ms)
        return STTTranscriptResult(
            stream_id=self.stream_id,
            sequence=state.turn_order,
            provider_event_id=f"assemblyai:{self._session_key}:{state.turn_order}",
            text=text,
            is_final=final,
            revision=state.revision,
            start_time_ms=state.start_ms,
            end_time_ms=max(state.start_ms, state.end_ms),
            monotonic_timestamp=self._clock(),
            detected_language=state.language,
            words=words if final and words else None,
            is_endpoint=final,
        )


# ── Provider ─────────────────────────────────────────────────────────────────


class _AssemblyAIStream:
    def __init__(self, stream_id: str, assembler: AssemblyAITurnAssembler) -> None:
        self.stream_id = stream_id
        self.assembler = assembler
        self.socket: Optional[StreamingSocket] = None
        self.buffer = bytearray()
        self.enqueued_bytes = 0
        self.ready = asyncio.Event()
        self.ended = asyncio.Event()
        self.began = False
        self.handshaking = True
        self.terminating = False
        self.audio_since_endpoint = False


class AssemblyAISTTProvider(StreamingSTTProvider):
    """Streaming STT over AssemblyAI Universal Streaming v3, one session per stream id.

    - ``start`` opens each session and waits for ``Begin``; a failure terminates and
      closes every session already opened, then re-raises.
    - ``send_audio`` re-chunks PCM to ``chunk_ms`` and only enqueues; a full queue drops
      the oldest audio and reports ``send_queue_state(stream_id).backpressured``.
    - ``finalize_stream`` flushes buffered audio (padded to the 50 ms minimum) and sends
      ``ForceEndpoint``. ``reset_stream`` drops unsent audio, abandons the open turn, and
      forces the server to end it so later speech starts a new turn.
    - ``stop`` flushes audio, sends ``Terminate``, waits for ``Termination``, finalizes
      open turns, then closes sockets and cancels tasks.
    """

    def __init__(
        self,
        settings: AssemblyAISettings,
        *,
        provider_id: str = "assemblyai",
        stream_ids: Sequence[str] = ("default",),
        transport: Optional[AssemblyAITransportOptions] = None,
        connect_factory: Optional[ConnectFactory] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        validate_settings(settings)
        transport = transport or AssemblyAITransportOptions()
        validate_transport(transport)
        if not stream_ids or len(set(stream_ids)) != len(stream_ids):
            raise _invalid("stream_ids")
        self._settings = settings
        self._provider_id = provider_id
        self._stream_ids = tuple(stream_ids)
        self._transport = transport
        self._connect_factory = connect_factory or make_websocket_factory()
        self._clock = clock
        self._sleep = sleep
        self._url = build_connection_url(settings)
        self._bytes_per_ms = settings.sample_rate * 2 / 1000.0
        self._chunk_bytes = self._even(transport.chunk_ms * self._bytes_per_ms)
        self._min_chunk_bytes = self._even(MIN_CHUNK_MS * self._bytes_per_ms)
        self._channel = ResultChannel()
        self._streams: Dict[str, _AssemblyAIStream] = {}
        self._started = False
        self._stopping = False

    # ── BaseProvider ─────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        if self._started:
            return
        self._channel.reset()
        self._stopping = False
        self._streams = {stream_id: self._new_stream(stream_id) for stream_id in self._stream_ids}
        self._started = True
        try:
            for stream in self._streams.values():
                await self._open_stream(stream)
        except BaseException:
            self._stopping = True
            try:
                await self._shutdown(self._transport.close_timeout_s, flush_audio=False)
            finally:
                self._started = False
            raise
        logger.info(
            "AssemblyAI STT started: provider=%s streams=%d endpoint=%s model=%s",
            self._provider_id,
            len(self._streams),
            redact_url(self._url),
            self._settings.speech_model,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        try:
            await self._shutdown(self._transport.terminate_timeout_s, flush_audio=True)
        finally:
            self._started = False
            self._channel.end()
        logger.info("AssemblyAI STT stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        began = self._clock()
        probe = self._new_stream("health-check")
        try:
            await self._open_stream(probe)
        except ProviderRuntimeError as err:
            await self._release(probe, self._transport.close_timeout_s, flush_audio=False)
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        except BaseException:
            await self._release(probe, self._transport.close_timeout_s, flush_audio=False)
            raise
        await self._release(probe, self._transport.close_timeout_s, flush_audio=False)
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="AssemblyAI began a streaming session.",
            latency_ms=(self._clock() - began) * 1000.0,
            details={"model": self._settings.speech_model},
        )

    # ── StreamingSTTProvider ─────────────────────────────────────────────────

    async def send_audio(self, frame: AudioFrame) -> None:
        stream = self._require_stream(frame.stream_id)
        if (
            frame.channels != 1
            or frame.sample_rate != self._settings.sample_rate
            or len(frame.pcm_data) % 2
        ):
            raise _invalid("audio_format")
        if not frame.pcm_data:
            return
        stream.buffer.extend(frame.pcm_data)
        stream.audio_since_endpoint = True
        while len(stream.buffer) >= self._chunk_bytes:
            chunk = bytes(stream.buffer[: self._chunk_bytes])
            del stream.buffer[: self._chunk_bytes]
            self._enqueue_chunk(stream, chunk)

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        async for result in self._channel.iterate():
            yield result

    async def finalize_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        if not stream.audio_since_endpoint or stream.socket is None:
            return
        self._flush_buffer(stream)
        stream.audio_since_endpoint = False
        stream.socket.enqueue_control(FORCE_ENDPOINT_MESSAGE)

    async def reset_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        socket = stream.socket
        if socket is None:
            return
        stream.buffer.clear()
        socket.clear_audio()
        abandon = stream.audio_since_endpoint or stream.assembler.has_open_turn
        stream.assembler.reset(self._sent_audio_ms(socket))
        stream.audio_since_endpoint = False
        if abandon:
            socket.enqueue_control(FORCE_ENDPOINT_MESSAGE)

    def send_queue_state(self, stream_id: str) -> SendQueueState:
        """Non-blocking view of backpressure on one stream's outbound audio."""
        stream = self._streams.get(stream_id)
        if stream is None or stream.socket is None:
            raise _invalid("stream_id")
        return stream.socket.state()

    # ── Session lifecycle ────────────────────────────────────────────────────

    def _new_stream(self, stream_id: str) -> _AssemblyAIStream:
        assembler = AssemblyAITurnAssembler(
            stream_id,
            expects_formatting=bool(self._settings.format_turns),
            clock=self._clock,
        )
        return _AssemblyAIStream(stream_id, assembler)

    async def _open_stream(self, stream: _AssemblyAIStream) -> None:
        conn = await self._connect()
        stream.socket = self._make_socket(stream, conn)
        stream.socket.start()
        try:
            await asyncio.wait_for(stream.ready.wait(), self._transport.handshake_timeout_s)
        except asyncio.TimeoutError:
            raise ProviderRuntimeError(make_stt_error(LABEL, ProviderErrorCode.TIMEOUT)) from None
        if not stream.began:
            socket = stream.socket
            raise ProviderRuntimeError(map_close(socket.close_code, socket.close_reason))
        stream.handshaking = False

    async def _connect(self) -> Any:
        headers = build_auth_headers(self._settings)
        try:
            return await open_connection(
                self._connect_factory, self._url, headers, self._transport.handshake_timeout_s
            )
        except HandshakeRejectedError as exc:
            code, retryable = classify_handshake_status(exc.status)
            error = make_stt_error(
                LABEL, code, retryable=retryable, details={"http_status": exc.status}
            )
        except asyncio.TimeoutError:
            error = make_stt_error(LABEL, ProviderErrorCode.TIMEOUT)
        except SocketClosedError as exc:
            error = map_close(exc.code, exc.reason)
        except SocketConnectError as exc:
            logger.warning("AssemblyAI connect failed: error=%s", exc.cause)
            error = make_stt_error(LABEL, ProviderErrorCode.NETWORK_UNAVAILABLE)
        raise ProviderRuntimeError(error)

    def _make_socket(self, stream: _AssemblyAIStream, conn: Any) -> StreamingSocket:
        def on_message(raw: WireMessage) -> None:
            self._on_message(stream, raw)

        def on_closed(exc: SocketClosedError) -> None:
            self._on_closed(stream, exc)

        # No keepalive: v3 documents none; audio keeps the session active and
        # ``inactivity_timeout`` governs silence.
        return StreamingSocket(
            conn,
            label=f"assemblyai-stt-{stream.stream_id}",
            on_message=on_message,
            on_closed=on_closed,
            max_queued_audio=self._transport.max_queued_frames,
            close_timeout_s=self._transport.close_timeout_s,
            clock=self._clock,
            sleep=self._sleep,
        )

    async def _shutdown(self, timeout: float, flush_audio: bool) -> None:
        streams = list(self._streams.values())
        try:
            await asyncio.gather(*(self._terminate(s, timeout, flush_audio) for s in streams))
        finally:
            for stream in streams:
                if stream.socket is not None:
                    await stream.socket.close()
                for result in stream.assembler.flush():
                    self._channel.publish(result)

    async def _release(self, stream: _AssemblyAIStream, timeout: float, flush_audio: bool) -> None:
        try:
            await self._terminate(stream, timeout, flush_audio)
        finally:
            if stream.socket is not None:
                await stream.socket.close()

    async def _terminate(
        self, stream: _AssemblyAIStream, timeout: float, flush_audio: bool
    ) -> None:
        """Send ``Terminate`` and wait for ``Termination`` or the server's close."""
        socket = stream.socket
        stream.terminating = True
        if socket is None or socket.closed:
            return
        if flush_audio:
            self._flush_buffer(stream)
        socket.enqueue_control(TERMINATE_MESSAGE)
        try:
            await asyncio.wait_for(stream.ended.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "AssemblyAI session did not confirm termination: stream=%s", stream.stream_id
            )

    # ── Audio ────────────────────────────────────────────────────────────────

    @staticmethod
    def _even(byte_count: float) -> int:
        count = round(byte_count)
        return max(2, count - (count % 2))

    def _enqueue_chunk(self, stream: _AssemblyAIStream, chunk: bytes) -> None:
        socket = stream.socket
        if socket is None:
            return
        socket.enqueue_audio(chunk)
        stream.enqueued_bytes += len(chunk)
        stream.assembler.set_audio_position(stream.enqueued_bytes / self._bytes_per_ms)

    def _flush_buffer(self, stream: _AssemblyAIStream) -> None:
        if not stream.buffer:
            return
        chunk = bytes(stream.buffer)
        stream.buffer.clear()
        if len(chunk) < self._min_chunk_bytes:
            # Messages under 50 ms are rejected (close 3007); pad the tail with silence.
            chunk += b"\x00" * (self._min_chunk_bytes - len(chunk))
        self._enqueue_chunk(stream, chunk)

    def _sent_audio_ms(self, socket: StreamingSocket) -> float:
        return socket.state().audio_bytes_dequeued / self._bytes_per_ms

    # ── Server events ────────────────────────────────────────────────────────

    def _on_message(self, stream: _AssemblyAIStream, raw: WireMessage) -> None:
        event = parse_message(raw)
        if isinstance(event, AssemblyAITurn):
            for result in stream.assembler.on_turn(event):
                self._channel.publish(result)
        elif isinstance(event, AssemblyAIBegin):
            stream.began = True
            stream.assembler.set_session(event.session_id)
            stream.ready.set()
            logger.debug("AssemblyAI session began: stream=%s", stream.stream_id)
        elif isinstance(event, AssemblyAITermination):
            stream.ended.set()
            logger.debug(
                "AssemblyAI session terminated: stream=%s audio_s=%s session_s=%s",
                stream.stream_id,
                event.audio_duration_s,
                event.session_duration_s,
            )
        elif isinstance(event, AssemblyAIServerError):
            logger.warning("AssemblyAI reported an error message: stream=%s", stream.stream_id)
        elif isinstance(event, AssemblyAIMalformed):
            logger.debug(
                "AssemblyAI malformed event ignored: stream=%s reason=%s",
                stream.stream_id,
                event.reason,
            )

    def _on_closed(self, stream: _AssemblyAIStream, exc: SocketClosedError) -> None:
        stream.ended.set()
        stream.ready.set()
        if stream.handshaking:
            return
        for result in stream.assembler.flush():
            self._channel.publish(result)
        if self._stopping or stream.terminating:
            return
        error = map_close(exc.code, exc.reason)
        logger.warning(
            "AssemblyAI connection closed unexpectedly: stream=%s close_code=%s",
            stream.stream_id,
            exc.code,
        )
        self._fail(error)

    def _fail(self, error: ProviderError) -> None:
        if self._stopping:
            return
        if self._channel.fail(error):
            logger.error(
                "AssemblyAI STT failed: provider=%s code=%s retryable=%s",
                self._provider_id,
                error.code.value,
                error.retryable,
            )

    def _require_stream(self, stream_id: str) -> _AssemblyAIStream:
        if self._channel.failure is not None:
            raise ProviderRuntimeError(self._channel.failure)
        if not self._started or self._stopping:
            raise _invalid("not_started")
        stream = self._streams.get(stream_id)
        if stream is None:
            raise _invalid("stream_id")
        return stream
