"""Deepgram live streaming STT adapter.

Protocol as documented on 2026-09-14 (``wss://api.deepgram.com/v1/listen``):

- Auth: ``Authorization: Token <api key>``, or ``Bearer <temporary JWT>``.
- Query: ``model``, ``encoding=linear16``, ``sample_rate``, ``channels=1``,
  ``interim_results=true``, plus optional ``language``, ``endpointing``,
  ``utterance_end_ms``, ``smart_format``, ``punctuate``, and repeated ``keyterm``.
- Audio: binary PCM16 frames. Control: text frames ``KeepAlive``, ``Finalize``,
  ``CloseStream``. The server closes after ~10 s without audio or ``KeepAlive``
  (``NET-0001``).
- Server: ``Results`` (``is_final``, ``speech_final``, ``from_finalize``,
  ``channel.alternatives[0]``), ``UtteranceEnd``, ``SpeechStarted``, ``Metadata``.
  Errors arrive as close frames carrying ``DATA-xxxx``/``NET-xxxx`` identifiers.

Finality. ``is_final`` only means a segment's text will not be revised; a long utterance
spans several of them. The utterance ends on ``speech_final`` (endpointing),
``from_finalize`` (our ``Finalize``), or ``UtteranceEnd``. Canonical results therefore
stay ``is_final=False`` while ``is_final`` segments accumulate, and one canonical final
carrying the joined segments is emitted per utterance.

The adapter is dormant: nothing registers it, it never chooses fallback, and it does not
reconnect. Unexpected closes surface from ``receive_events`` as ``ProviderRuntimeError``.
Nothing here logs the key, the query string, audio, or transcript text.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
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
    sanitize_code,
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

logger = get_logger("providers.stt.deepgram")

LABEL = "Deepgram"
AUTH_SCHEME_TOKEN = "token"
AUTH_SCHEME_BEARER = "bearer"
_AUTH_PREFIX = {AUTH_SCHEME_TOKEN: "Token", AUTH_SCHEME_BEARER: "Bearer"}

KEEPALIVE_MESSAGE = json.dumps({"type": "KeepAlive"}, separators=(",", ":"))
FINALIZE_MESSAGE = json.dumps({"type": "Finalize"}, separators=(",", ":"))
CLOSE_STREAM_MESSAGE = json.dumps({"type": "CloseStream"}, separators=(",", ":"))

MIN_UTTERANCE_END_MS = 1000
_ERROR_ID_RE = re.compile(r"\b((?:DATA|NET)-\d{4})\b")
_EPSILON_MS = 1.0


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeepgramSettings:
    """Resolved settings. Endpoint, model, and sample rate have no adapter defaults.

    Optional features are sent only when set, so the provider's own defaults apply
    otherwise. ``interim_results`` is always on because partials depend on it.
    """

    endpoint: str
    model: str
    api_key: str = field(repr=False)
    sample_rate: int
    auth_scheme: str = AUTH_SCHEME_TOKEN
    language: Optional[str] = None
    endpointing_ms: Optional[int] = None
    utterance_end_ms: Optional[int] = None
    smart_format: Optional[bool] = None
    punctuate: Optional[bool] = None
    keyterms: Tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class DeepgramTransportOptions:
    """Transport tuning only; provider behaviour lives in ``DeepgramSettings``."""

    handshake_timeout_s: float = 10.0
    close_timeout_s: float = 2.0
    terminate_timeout_s: float = 5.0
    max_queued_frames: int = 250
    # Deepgram closes idle sockets after 10 s and recommends a KeepAlive every 3-5 s.
    keepalive_interval_s: float = 4.0
    keepalive_poll_s: float = 0.5


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _invalid(field_name: str) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        make_stt_error(LABEL, ProviderErrorCode.INVALID_REQUEST, details={"field": field_name})
    )


def validate_settings(settings: DeepgramSettings) -> None:
    """Reject unusable settings before any network activity."""
    if not isinstance(settings.model, str) or not settings.model.strip():
        raise _invalid("model")
    if not isinstance(settings.api_key, str) or not settings.api_key.strip():
        raise ProviderRuntimeError(
            make_stt_error(LABEL, ProviderErrorCode.AUTH_FAILED, details={"field": "api_key"})
        )
    if settings.auth_scheme not in _AUTH_PREFIX:
        raise _invalid("auth_scheme")
    if not _is_int(settings.sample_rate) or settings.sample_rate <= 0:
        raise _invalid("sample_rate")
    if settings.language is not None and (
        not isinstance(settings.language, str) or not settings.language.strip()
    ):
        raise _invalid("language")
    if settings.endpointing_ms is not None and (
        not _is_int(settings.endpointing_ms) or settings.endpointing_ms < 0
    ):
        raise _invalid("endpointing_ms")
    if settings.utterance_end_ms is not None and (
        not _is_int(settings.utterance_end_ms) or settings.utterance_end_ms < MIN_UTTERANCE_END_MS
    ):
        raise _invalid("utterance_end_ms")
    for term in settings.keyterms:
        if not isinstance(term, str) or not term.strip():
            raise _invalid("keyterms")
    try:
        validate_endpoint(settings.endpoint)
    except ValueError:
        raise _invalid("endpoint") from None


def _flag(value: bool) -> str:
    return "true" if value else "false"


def build_query(settings: DeepgramSettings) -> List[Tuple[str, str]]:
    params: List[Tuple[str, str]] = [
        ("model", settings.model),
        ("encoding", "linear16"),
        ("sample_rate", str(settings.sample_rate)),
        ("channels", "1"),
        ("interim_results", "true"),
    ]
    if settings.language:
        params.append(("language", settings.language))
    if settings.endpointing_ms is not None:
        params.append(("endpointing", str(settings.endpointing_ms)))
    if settings.utterance_end_ms is not None:
        params.append(("utterance_end_ms", str(settings.utterance_end_ms)))
    if settings.smart_format is not None:
        params.append(("smart_format", _flag(settings.smart_format)))
    if settings.punctuate is not None:
        params.append(("punctuate", _flag(settings.punctuate)))
    params.extend(("keyterm", term) for term in settings.keyterms)
    return params


def build_connection_url(settings: DeepgramSettings) -> str:
    return build_url(settings.endpoint, build_query(settings))


def build_auth_headers(settings: DeepgramSettings) -> Dict[str, str]:
    return {"Authorization": f"{_AUTH_PREFIX[settings.auth_scheme]} {settings.api_key}"}


# ── Server events ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeepgramResults:
    transcript: str
    is_final: bool
    speech_final: bool
    from_finalize: bool
    start_s: float
    duration_s: float
    confidence: Optional[float] = None
    words: Tuple[WordTiming, ...] = ()
    language: Optional[str] = None


@dataclass(frozen=True)
class DeepgramUtteranceEnd:
    last_word_end_s: Optional[float] = None


@dataclass(frozen=True)
class DeepgramMetadata:
    request_id: Optional[str] = None


@dataclass(frozen=True)
class DeepgramServerError:
    """In-band error. Only a sanitized identifier survives; free text is dropped."""

    code: Optional[str] = None


@dataclass(frozen=True)
class DeepgramIgnored:
    event_type: str


@dataclass(frozen=True)
class DeepgramMalformed:
    reason: str


DeepgramEvent = Union[  # noqa: UP007 -- typing house style (rule 04)
    DeepgramResults,
    DeepgramUtteranceEnd,
    DeepgramMetadata,
    DeepgramServerError,
    DeepgramIgnored,
    DeepgramMalformed,
]


class _MalformedError(Exception):
    pass


def _number(value: Any, name: str, required: bool = True) -> Optional[float]:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise _MalformedError(name)
    return float(value)


def _optional_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_words(raw_words: Any) -> Tuple[WordTiming, ...]:
    if not isinstance(raw_words, list):
        return ()
    words: List[WordTiming] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        text = raw.get("punctuated_word") or raw.get("word")
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(text, str) or isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        words.append(
            WordTiming(
                word=text,
                start_time_ms=float(start) * 1000.0,
                end_time_ms=float(end) * 1000.0,
                confidence=_optional_confidence(raw.get("confidence")),
            )
        )
    return tuple(words)


def _parse_results(payload: Dict[str, Any]) -> DeepgramResults:
    channel = payload.get("channel")
    if not isinstance(channel, dict):
        raise _MalformedError("channel")
    alternatives = channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        raise _MalformedError("alternatives")
    best = alternatives[0]
    if not isinstance(best, dict) or not isinstance(best.get("transcript"), str):
        raise _MalformedError("transcript")
    is_final = payload.get("is_final")
    if not isinstance(is_final, bool):
        raise _MalformedError("is_final")
    languages = best.get("languages")
    language = (
        languages[0]
        if isinstance(languages, list) and languages and isinstance(languages[0], str)
        else None
    )
    return DeepgramResults(
        transcript=best["transcript"],
        is_final=is_final,
        speech_final=payload.get("speech_final") is True,
        from_finalize=payload.get("from_finalize") is True,
        start_s=_number(payload.get("start"), "start") or 0.0,
        duration_s=_number(payload.get("duration"), "duration") or 0.0,
        confidence=_optional_confidence(best.get("confidence")),
        words=_parse_words(best.get("words")),
        language=language,
    )


def parse_message(raw: WireMessage) -> DeepgramEvent:
    """Parse one server frame. Never raises; bad input becomes ``DeepgramMalformed``."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    except (ValueError, TypeError):
        return DeepgramMalformed("invalid_json")
    if not isinstance(payload, dict):
        return DeepgramMalformed("not_an_object")
    kind = payload.get("type")
    if not isinstance(kind, str) or not kind:
        return DeepgramMalformed("missing_type")
    try:
        if kind == "Results":
            return _parse_results(payload)
        if kind == "UtteranceEnd":
            return DeepgramUtteranceEnd(
                _number(payload.get("last_word_end"), "last_word_end", required=False)
            )
        if kind == "Metadata":
            request_id = payload.get("request_id")
            return DeepgramMetadata(request_id if isinstance(request_id, str) else None)
        if kind == "Error":
            return DeepgramServerError(
                sanitize_code(payload.get("variant")) or sanitize_code(payload.get("err_code"))
            )
    except _MalformedError as exc:
        return DeepgramMalformed(f"{kind}:{exc}")
    return DeepgramIgnored(kind)


# ── Error mapping ────────────────────────────────────────────────────────────


def map_error_identifier(identifier: Optional[str]) -> Optional[ProviderErrorCode]:
    if not identifier:
        return None
    if identifier.startswith("DATA-"):
        return ProviderErrorCode.INVALID_REQUEST
    if identifier.startswith("NET-"):
        return ProviderErrorCode.TIMEOUT
    return None


def map_close(code: Optional[int], reason: str = "") -> ProviderError:
    """Map a close frame to a canonical error. Only the identifier leaves the reason."""
    match = _ERROR_ID_RE.search(reason or "")
    identifier = match.group(1) if match else None
    mapped = map_error_identifier(identifier)
    if mapped is None:
        if code in (1003, 1007, 1008, 1009):
            mapped = ProviderErrorCode.INVALID_REQUEST
        elif code in (1011, 1012, 1013):
            mapped = ProviderErrorCode.INTERNAL_ERROR
        else:
            mapped = ProviderErrorCode.CONNECTION_CLOSED
    return make_stt_error(LABEL, mapped, details={"close_code": code, "provider_code": identifier})


def map_server_error(event: DeepgramServerError) -> ProviderError:
    mapped = map_error_identifier(event.code) or ProviderErrorCode.INTERNAL_ERROR
    return make_stt_error(LABEL, mapped, details={"provider_code": event.code})


# ── Utterance assembly ───────────────────────────────────────────────────────


class DeepgramUtteranceAssembler:
    """Maps Deepgram segments to canonical utterance results for one stream.

    Invariants:

    - ``sequence`` is the utterance ordinal; ``provider_event_id`` is stable per utterance.
    - ``revision`` strictly increases within an utterance; its final is the last revision.
    - Canonical ``is_final`` is emitted once, on ``speech_final``, ``from_finalize``, or
      ``UtteranceEnd``, with every ``is_final`` segment of the utterance joined in order.
    - Replayed segments (audio already finalized), unchanged partials, and repeated
      endpoint signals emit nothing. Results for audio sent before ``reset`` are ignored.
    - Times are milliseconds of audio received by Deepgram on this stream.
    """

    def __init__(self, stream_id: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.stream_id = stream_id
        self._clock = clock
        self._ordinal = 0
        self._final_end_ms = 0.0
        self._discard_before_ms = 0.0
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._parts: List[str] = []
        self._words: List[WordTiming] = []
        self._interim = ""
        self._start_ms: Optional[float] = None
        self._end_ms = 0.0
        self._revision = 0
        self._last_text: Optional[str] = None
        self._confidence: Optional[float] = None
        self._language: Optional[str] = None

    @property
    def has_open_utterance(self) -> bool:
        return bool(self._parts or self._interim or self._revision)

    def on_results(self, event: DeepgramResults) -> List[STTTranscriptResult]:
        start_ms = event.start_s * 1000.0
        end_ms = start_ms + event.duration_s * 1000.0
        if event.duration_s > 0:
            before_reset = end_ms <= self._discard_before_ms + _EPSILON_MS
        else:
            before_reset = end_ms < self._discard_before_ms - _EPSILON_MS
        if before_reset:
            return []
        text = event.transcript.strip()
        if not event.is_final:
            if not text or end_ms <= self._final_end_ms + _EPSILON_MS:
                return []
            self._absorb(start_ms, end_ms, event)
            self._interim = text
            partial = self._emit_partial()
            return [partial] if partial is not None else []

        replay = event.duration_s > 0 and end_ms <= self._final_end_ms + _EPSILON_MS
        changed = False
        if not replay:
            self._final_end_ms = max(self._final_end_ms, end_ms)
            self._interim = ""
            if text:
                self._absorb(start_ms, end_ms, event)
                self._parts.append(text)
                self._words.extend(event.words)
                changed = True
        if event.speech_final or event.from_finalize:
            final = self._close(promote_interim=False)
            return [final] if final is not None else []
        if changed:
            partial = self._emit_partial()
            return [partial] if partial is not None else []
        return []

    def on_utterance_end(self) -> Optional[STTTranscriptResult]:
        if not self._parts:
            return None
        return self._close(promote_interim=False)

    def flush(self) -> Optional[STTTranscriptResult]:
        """Finalize whatever is shown when the session ends."""
        return self._close(promote_interim=True)

    def reset(self, discard_before_ms: float) -> None:
        """Abandon the open utterance and ignore results for earlier audio."""
        if self._revision:
            self._ordinal += 1
        self._reset_utterance()
        self._discard_before_ms = max(self._discard_before_ms, discard_before_ms)
        self._final_end_ms = max(self._final_end_ms, discard_before_ms)

    def _absorb(self, start_ms: float, end_ms: float, event: DeepgramResults) -> None:
        if self._start_ms is None:
            self._start_ms = start_ms
        self._end_ms = max(self._end_ms, end_ms)
        if event.confidence is not None:
            self._confidence = event.confidence
        if event.language:
            self._language = event.language

    def _emit_partial(self) -> Optional[STTTranscriptResult]:
        pieces = [*self._parts, self._interim] if self._interim else list(self._parts)
        text = " ".join(pieces)
        if not text or text == self._last_text:
            return None
        return self._emit(text, final=False)

    def _close(self, promote_interim: bool) -> Optional[STTTranscriptResult]:
        if self._parts:
            text = " ".join(self._parts)
        elif promote_interim and self._interim:
            text = self._interim
        elif self._revision:
            # Every partial was retracted; close the utterance explicitly.
            text = ""
        else:
            self._reset_utterance()
            return None
        result = self._emit(text, final=True)
        self._ordinal += 1
        self._reset_utterance()
        return result

    def _emit(self, text: str, final: bool) -> STTTranscriptResult:
        self._revision += 1
        self._last_text = text
        start_ms = self._start_ms if self._start_ms is not None else self._final_end_ms
        return STTTranscriptResult(
            stream_id=self.stream_id,
            sequence=self._ordinal,
            provider_event_id=f"deepgram:{self.stream_id}:{self._ordinal}",
            text=text,
            is_final=final,
            revision=self._revision,
            start_time_ms=start_ms,
            end_time_ms=max(start_ms, self._end_ms),
            monotonic_timestamp=self._clock(),
            detected_language=self._language,
            confidence=self._confidence,
            words=tuple(self._words) if final and self._words else None,
            is_endpoint=final,
        )


# ── Provider ─────────────────────────────────────────────────────────────────


class _DeepgramStream:
    def __init__(self, stream_id: str, assembler: DeepgramUtteranceAssembler) -> None:
        self.stream_id = stream_id
        self.assembler = assembler
        self.socket: Optional[StreamingSocket] = None
        self.unfinalized_audio = False
        self.closing = False


class DeepgramSTTProvider(StreamingSTTProvider):
    """Streaming STT over Deepgram's live WebSocket API, one socket per stream id.

    - ``send_audio`` only enqueues; a full queue drops the oldest audio and reports
      ``send_queue_state(stream_id).backpressured`` instead of blocking capture.
    - ``finalize_stream`` sends ``Finalize``; ``reset_stream`` drops queued audio and
      ignores results for audio already sent.
    - ``stop`` sends ``CloseStream`` after queued audio, waits for the server to flush
      and close, finalizes anything shown, then closes sockets and cancels tasks.
    """

    @classmethod
    def from_resolved(
        cls,
        resolved: Any,
        secrets: Mapping[str, str],
        *,
        stream_ids: Sequence[str] = ("default",),
        connect_factory: Optional[ConnectFactory] = None,
    ) -> "DeepgramSTTProvider":
        api_key = secrets.get("api_key", "")
        options = getattr(resolved, "options", {}) or {}
        language = options.get("language")
        if isinstance(language, str) and (not language.strip() or language.strip().lower() == "auto"):
            language = None
        endpoint = getattr(resolved, "endpoint", "") or "wss://api.deepgram.com/v1/listen"
        model = getattr(resolved, "model", "") or "nova-3"
        provider_id = getattr(resolved, "provider_id", "deepgram")
        settings = DeepgramSettings(
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            auth_scheme=options.get("auth_scheme", "token"),
            sample_rate=options.get("sample_rate", 16000),
            language=language,
            endpointing_ms=options.get("endpointing_ms"),
            utterance_end_ms=options.get("utterance_end_ms"),
            smart_format=options.get("smart_format"),
            punctuate=options.get("punctuate"),
            keyterms=tuple(options.get("keyterms", ())),
        )
        return cls(
            settings,
            provider_id=provider_id,
            stream_ids=stream_ids,
            connect_factory=connect_factory,
        )

    def __init__(
        self,
        settings: DeepgramSettings,
        *,
        provider_id: str = "deepgram",
        stream_ids: Sequence[str] = ("default",),
        transport: Optional[DeepgramTransportOptions] = None,
        connect_factory: Optional[ConnectFactory] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        validate_settings(settings)
        if not stream_ids or len(set(stream_ids)) != len(stream_ids):
            raise _invalid("stream_ids")
        self._settings = settings
        self._provider_id = provider_id
        self._stream_ids = tuple(stream_ids)
        self._transport = transport or DeepgramTransportOptions()
        self._connect_factory = connect_factory or make_websocket_factory()
        self._clock = clock
        self._sleep = sleep
        self._url = build_connection_url(settings)
        self._channel = ResultChannel()
        self._streams: Dict[str, _DeepgramStream] = {}
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
        self._streams = {
            stream_id: _DeepgramStream(
                stream_id, DeepgramUtteranceAssembler(stream_id, self._clock)
            )
            for stream_id in self._stream_ids
        }
        self._started = True
        try:
            for stream in self._streams.values():
                conn = await self._connect()
                stream.socket = self._make_socket(stream, conn)
                stream.socket.start()
        except BaseException:
            self._stopping = True
            try:
                await self._shutdown(self._transport.close_timeout_s)
            finally:
                self._started = False
            raise
        logger.info(
            "Deepgram STT started: provider=%s streams=%d endpoint=%s model=%s",
            self._provider_id,
            len(self._streams),
            redact_url(self._url),
            self._settings.model,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        try:
            await self._shutdown(self._transport.terminate_timeout_s)
        finally:
            self._started = False
            self._channel.end()
        logger.info("Deepgram STT stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        began = self._clock()
        try:
            conn = await self._connect()
        except ProviderRuntimeError as err:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        try:
            await asyncio.wait_for(conn.send(CLOSE_STREAM_MESSAGE), self._transport.close_timeout_s)
        except (SocketClosedError, asyncio.TimeoutError):
            pass
        finally:
            try:
                await asyncio.wait_for(conn.close(), self._transport.close_timeout_s)
            except Exception as exc:
                logger.debug("Deepgram health check close failed: error=%s", type(exc).__name__)
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Deepgram accepted the live transcription connection.",
            latency_ms=(self._clock() - began) * 1000.0,
            details={"model": self._settings.model},
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
        if not frame.pcm_data or stream.socket is None:
            return
        stream.unfinalized_audio = True
        stream.socket.enqueue_audio(frame.pcm_data)

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        async for result in self._channel.iterate():
            yield result

    async def finalize_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        if not stream.unfinalized_audio or stream.socket is None:
            return
        stream.unfinalized_audio = False
        stream.socket.enqueue_control(FINALIZE_MESSAGE)

    async def reset_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        if stream.socket is None:
            return
        stream.socket.clear_audio()
        stream.unfinalized_audio = False
        stream.assembler.reset(self._sent_audio_ms(stream.socket))

    def send_queue_state(self, stream_id: str) -> SendQueueState:
        """Non-blocking view of backpressure on one stream's outbound audio."""
        stream = self._streams.get(stream_id)
        if stream is None or stream.socket is None:
            raise _invalid("stream_id")
        return stream.socket.state()

    # ── Internals ────────────────────────────────────────────────────────────

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
            logger.warning("Deepgram connect failed: error=%s", exc.cause)
            error = make_stt_error(LABEL, ProviderErrorCode.NETWORK_UNAVAILABLE)
        raise ProviderRuntimeError(error)

    def _make_socket(self, stream: _DeepgramStream, conn: Any) -> StreamingSocket:
        def on_message(raw: WireMessage) -> None:
            self._on_message(stream, raw)

        def on_closed(exc: SocketClosedError) -> None:
            self._on_closed(stream, exc)

        return StreamingSocket(
            conn,
            label=f"deepgram-stt-{stream.stream_id}",
            on_message=on_message,
            on_closed=on_closed,
            max_queued_audio=self._transport.max_queued_frames,
            keepalive_message=KEEPALIVE_MESSAGE,
            keepalive_interval_s=self._transport.keepalive_interval_s,
            keepalive_poll_s=self._transport.keepalive_poll_s,
            close_timeout_s=self._transport.close_timeout_s,
            clock=self._clock,
            sleep=self._sleep,
        )

    async def _shutdown(self, terminate_timeout_s: float) -> None:
        streams = list(self._streams.values())
        try:
            await asyncio.gather(*(self._close_stream(s, terminate_timeout_s) for s in streams))
        finally:
            for stream in streams:
                if stream.socket is not None:
                    await stream.socket.close()
                self._channel.publish(stream.assembler.flush())

    async def _close_stream(self, stream: _DeepgramStream, timeout: float) -> None:
        socket = stream.socket
        if socket is None or socket.closed:
            return
        stream.closing = True
        # Queued audio goes out first; the server flushes results, then closes.
        socket.enqueue_control(CLOSE_STREAM_MESSAGE)
        if not await socket.wait_closed(timeout):
            logger.warning("Deepgram did not close after CloseStream: stream=%s", stream.stream_id)

    def _on_message(self, stream: _DeepgramStream, raw: WireMessage) -> None:
        event = parse_message(raw)
        if isinstance(event, DeepgramResults):
            for result in stream.assembler.on_results(event):
                self._channel.publish(result)
        elif isinstance(event, DeepgramUtteranceEnd):
            self._channel.publish(stream.assembler.on_utterance_end())
        elif isinstance(event, DeepgramServerError):
            self._fail(map_server_error(event))
        elif isinstance(event, DeepgramMalformed):
            logger.debug(
                "Deepgram malformed event ignored: stream=%s reason=%s",
                stream.stream_id,
                event.reason,
            )
        elif isinstance(event, DeepgramMetadata):
            logger.debug("Deepgram metadata received: stream=%s", stream.stream_id)

    def _on_closed(self, stream: _DeepgramStream, exc: SocketClosedError) -> None:
        self._channel.publish(stream.assembler.flush())
        if self._stopping or stream.closing:
            return
        error = map_close(exc.code, exc.reason)
        logger.warning(
            "Deepgram connection closed unexpectedly: stream=%s close_code=%s provider_code=%s",
            stream.stream_id,
            exc.code,
            (error.details or {}).get("provider_code"),
        )
        self._fail(error)

    def _sent_audio_ms(self, socket: StreamingSocket) -> float:
        sent_bytes = socket.state().audio_bytes_dequeued
        return sent_bytes / (2 * self._settings.sample_rate) * 1000.0

    def _fail(self, error: ProviderError) -> None:
        if self._stopping:
            return
        if self._channel.fail(error):
            logger.error(
                "Deepgram STT failed: provider=%s code=%s retryable=%s",
                self._provider_id,
                error.code.value,
                error.retryable,
            )

    def _require_stream(self, stream_id: str) -> _DeepgramStream:
        if self._channel.failure is not None:
            raise ProviderRuntimeError(self._channel.failure)
        if not self._started or self._stopping:
            raise _invalid("not_started")
        stream = self._streams.get(stream_id)
        if stream is None:
            raise _invalid("stream_id")
        return stream
