"""OpenAI Realtime transcription protocol: request builders, event parsing, and mapping.

Everything here is pure and socket-free so the lifecycle in ``openai_realtime`` can be
driven by scripted events in tests. Protocol shape follows the GA Realtime transcription
session (``session.type = "transcription"``) as documented on 2026-09-14:

- client: ``session.update``, ``input_audio_buffer.append|commit|clear``
- server: ``session.created|updated``, ``input_audio_buffer.committed|cleared``,
  ``input_audio_buffer.speech_started|speech_stopped``,
  ``conversation.item.input_audio_transcription.delta|completed|failed``, ``error``

Provider error messages are never copied into canonical errors: they can echo request
fragments, so only a sanitized provider code survives.
"""

import base64
import json
import re
import sys
import time
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderRuntimeError,
    STTTranscriptResult,
)

# The Realtime API only accepts 24 kHz mono PCM16 for ``audio/pcm``.
PCM_SAMPLE_RATE = 24000
DELAY_VALUES = ("minimal", "low", "medium", "high", "xhigh")
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SESSION_EXPIRED_CODE = "session_expired"
COMMIT_EMPTY_CODE = "input_audio_buffer_commit_empty"
BENIGN_ERROR_CODES = frozenset({COMMIT_EMPTY_CODE})

_PROVIDER_CODE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenAIRealtimeSettings:
    """Resolved provider settings. The adapter owns no provider defaults.

    ``endpoint``, ``model``, and ``api_key`` come from the resolved service config and
    credential reference; hints are optional and only sent when present.
    """

    endpoint: str
    model: str
    api_key: str = field(repr=False)
    languages: Tuple[str, ...] = ()
    prompt: Optional[str] = field(default=None, repr=False)
    keywords: Tuple[str, ...] = field(default=(), repr=False)
    delay: Optional[str] = None
    noise_reduction: Optional[str] = None


def validate_settings(settings: OpenAIRealtimeSettings) -> None:
    """Reject unusable settings before any network activity."""
    if not settings.model or not settings.model.strip():
        raise ProviderRuntimeError(make_error(ProviderErrorCode.INVALID_REQUEST, detail="model"))
    if not settings.api_key or not settings.api_key.strip():
        raise ProviderRuntimeError(make_error(ProviderErrorCode.AUTH_FAILED, detail="api_key"))
    if settings.delay is not None and settings.delay not in DELAY_VALUES:
        raise ProviderRuntimeError(make_error(ProviderErrorCode.INVALID_REQUEST, detail="delay"))
    for language in settings.languages:
        if not isinstance(language, str) or not language.strip():
            raise ProviderRuntimeError(
                make_error(ProviderErrorCode.INVALID_REQUEST, detail="languages")
            )
    build_connection_url(settings.endpoint)


def build_connection_url(endpoint: str) -> str:
    """Return the socket URL with ``intent=transcription`` ensured.

    Plain ``ws://`` is allowed only for loopback hosts, and credentials embedded in the
    URL are rejected so they can never reach a log line or a proxy.
    """
    invalid = ProviderRuntimeError(make_error(ProviderErrorCode.INVALID_REQUEST, detail="endpoint"))
    if not endpoint or not isinstance(endpoint, str):
        raise invalid
    try:
        parts = urlsplit(endpoint.strip())
        hostname = parts.hostname
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError:
        raise invalid from None
    if not hostname or has_userinfo:
        raise invalid
    scheme = parts.scheme.lower()
    if scheme not in ("wss", "ws") or (scheme == "ws" and hostname not in LOOPBACK_HOSTS):
        raise invalid
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key == "intent" for key, _ in query):
        query.append(("intent", "transcription"))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), ""))


def redact_endpoint(url: str) -> str:
    """Scheme, host, port, and path only: safe for logs."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<invalid endpoint>"
    return f"{parts.scheme}://{host}{port}{parts.path}"


def uses_language_list(model: str) -> bool:
    """Live transcription models take ``languages``; older ones take ``language``."""
    return "live" in re.split(r"[-_.]", model.lower())


# ── Client requests ──────────────────────────────────────────────────────────


def build_session_update(settings: OpenAIRealtimeSettings) -> Dict[str, Any]:
    transcription: Dict[str, Any] = {"model": settings.model}
    if settings.languages:
        if uses_language_list(settings.model):
            transcription["languages"] = list(settings.languages)
        else:
            transcription["language"] = settings.languages[0]
    if settings.prompt:
        transcription["prompt"] = settings.prompt
    if settings.keywords:
        transcription["keywords"] = list(settings.keywords)
    if settings.delay:
        transcription["delay"] = settings.delay

    audio_input: Dict[str, Any] = {
        "format": {"type": "audio/pcm", "rate": PCM_SAMPLE_RATE},
        "transcription": transcription,
        # Boundaries come from EchoFlux via finalize_stream(), not server VAD.
        "turn_detection": None,
    }
    if settings.noise_reduction:
        audio_input["noise_reduction"] = {"type": settings.noise_reduction}
    return {
        "type": "session.update",
        "session": {"type": "transcription", "audio": {"input": audio_input}},
    }


def build_append(pcm16: bytes) -> Dict[str, Any]:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode("ascii")}


def build_commit() -> Dict[str, Any]:
    return {"type": "input_audio_buffer.commit"}


def build_clear() -> Dict[str, Any]:
    return {"type": "input_audio_buffer.clear"}


def encode(message: Dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"))


class PCM16Resampler:
    """Stateful linear resampler for mono little-endian PCM16.

    State carries across frames so chunk boundaries do not introduce clicks; feeding a
    signal in pieces yields the same samples as feeding it whole.
    """

    def __init__(self, src_rate: int, dst_rate: int = PCM_SAMPLE_RATE) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._step = src_rate / dst_rate
        self._pos = 0.0
        self._prev: Optional[int] = None

    def process(self, pcm16: bytes) -> bytes:
        if self.src_rate == self.dst_rate:
            return pcm16
        samples = array("h")
        samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
        if sys.byteorder == "big":
            samples.byteswap()
        if not samples:
            return b""
        buf = samples.tolist() if self._prev is None else [self._prev, *samples]
        last = len(buf) - 1
        out = array("h")
        pos = self._pos
        while pos < last:
            index = int(pos)
            frac = pos - index
            left = buf[index]
            out.append(round(left + (buf[index + 1] - left) * frac))
            pos += self._step
        # Re-base so index 0 of the next buffer is this buffer's final sample.
        self._pos = pos - last
        self._prev = buf[last]
        if sys.byteorder == "big":
            out.byteswap()
        return out.tobytes()


# ── Server events ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionReady:
    kind: str
    session_id: Optional[str] = None
    expires_at: Optional[float] = None
    event_id: Optional[str] = None


@dataclass(frozen=True)
class SpeechStarted:
    item_id: str
    audio_start_ms: float
    event_id: Optional[str] = None


@dataclass(frozen=True)
class SpeechStopped:
    item_id: str
    audio_end_ms: float
    event_id: Optional[str] = None


@dataclass(frozen=True)
class BufferCommitted:
    item_id: str
    previous_item_id: Optional[str] = None
    event_id: Optional[str] = None


@dataclass(frozen=True)
class BufferCleared:
    event_id: Optional[str] = None


@dataclass(frozen=True)
class TranscriptDelta:
    item_id: str
    delta: str
    content_index: int = 0
    event_id: Optional[str] = None


@dataclass(frozen=True)
class TranscriptCompleted:
    item_id: str
    transcript: str
    content_index: int = 0
    event_id: Optional[str] = None


@dataclass(frozen=True)
class TranscriptFailed:
    item_id: str
    error_code: Optional[str] = None
    event_id: Optional[str] = None


@dataclass(frozen=True)
class ServerError:
    """Server ``error`` event. The provider's free-text message is deliberately dropped."""

    error_type: Optional[str] = None
    code: Optional[str] = None
    param: Optional[str] = None
    event_id: Optional[str] = None


@dataclass(frozen=True)
class IgnoredEvent:
    event_type: str
    event_id: Optional[str] = None


@dataclass(frozen=True)
class MalformedEvent:
    reason: str


ServerEvent = (
    SessionReady
    | SpeechStarted
    | SpeechStopped
    | BufferCommitted
    | BufferCleared
    | TranscriptDelta
    | TranscriptCompleted
    | TranscriptFailed
    | ServerError
    | IgnoredEvent
    | MalformedEvent
)


class _MalformedError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _opt_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _req_str(payload: Dict[str, Any], key: str, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise _MalformedError(key)
    return value


def _req_ms(payload: Dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise _MalformedError(key)
    return float(value)


def _content_index(payload: Dict[str, Any]) -> int:
    value = payload.get("content_index", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _MalformedError("content_index")
    return value


def _parse_session(payload: Dict[str, Any], event_id: Optional[str]) -> ServerEvent:
    session = payload.get("session")
    if not isinstance(session, dict):
        raise _MalformedError("session")
    expires = session.get("expires_at")
    expires_at = (
        float(expires)
        if isinstance(expires, (int, float)) and not isinstance(expires, bool) and expires > 0
        else None
    )
    kind = "created" if payload["type"] == "session.created" else "updated"
    return SessionReady(kind, _opt_str(session.get("id")), expires_at, event_id)


def _parse_error(payload: Dict[str, Any], event_id: Optional[str]) -> ServerEvent:
    error = payload.get("error")
    if not isinstance(error, dict):
        raise _MalformedError("error")
    return ServerError(
        error_type=_opt_str(error.get("type")),
        code=_opt_str(error.get("code")),
        param=_opt_str(error.get("param")),
        event_id=event_id,
    )


def _parse_failed(payload: Dict[str, Any], event_id: Optional[str]) -> ServerEvent:
    error = payload.get("error")
    code = _opt_str(error.get("code")) if isinstance(error, dict) else None
    return TranscriptFailed(_req_str(payload, "item_id"), code, event_id)


_Parser = Callable[[Dict[str, Any], Optional[str]], ServerEvent]

_PARSERS: Dict[str, _Parser] = {
    "session.created": _parse_session,
    "session.updated": _parse_session,
    "error": _parse_error,
    "input_audio_buffer.speech_started": lambda p, e: SpeechStarted(
        _req_str(p, "item_id"), _req_ms(p, "audio_start_ms"), e
    ),
    "input_audio_buffer.speech_stopped": lambda p, e: SpeechStopped(
        _req_str(p, "item_id"), _req_ms(p, "audio_end_ms"), e
    ),
    "input_audio_buffer.committed": lambda p, e: BufferCommitted(
        _req_str(p, "item_id"), _opt_str(p.get("previous_item_id")), e
    ),
    "input_audio_buffer.cleared": lambda p, e: BufferCleared(e),
    "conversation.item.input_audio_transcription.delta": lambda p, e: TranscriptDelta(
        _req_str(p, "item_id"), _req_str(p, "delta", allow_empty=True), _content_index(p), e
    ),
    "conversation.item.input_audio_transcription.completed": lambda p, e: TranscriptCompleted(
        _req_str(p, "item_id"),
        _req_str(p, "transcript", allow_empty=True),
        _content_index(p),
        e,
    ),
    "conversation.item.input_audio_transcription.failed": _parse_failed,
}


def parse_server_event(raw: str | bytes) -> ServerEvent:
    """Parse one server frame. Never raises; bad input becomes ``MalformedEvent``."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return MalformedEvent("invalid_json")
    if not isinstance(payload, dict):
        return MalformedEvent("not_an_object")
    event_type = payload.get("type")
    if not isinstance(event_type, str) or not event_type:
        return MalformedEvent("missing_type")
    event_id = _opt_str(payload.get("event_id"))
    parser = _PARSERS.get(event_type)
    if parser is None:
        return IgnoredEvent(event_type, event_id)
    try:
        return parser(payload, event_id)
    except _MalformedError as exc:
        return MalformedEvent(f"{event_type}:{exc.reason}")


# ── Error mapping ────────────────────────────────────────────────────────────

_SAFE_MESSAGES: Dict[ProviderErrorCode, str] = {
    ProviderErrorCode.AUTH_FAILED: "OpenAI Realtime rejected the credentials.",
    ProviderErrorCode.RATE_LIMITED: "OpenAI Realtime rate limit reached.",
    ProviderErrorCode.QUOTA_EXCEEDED: "OpenAI Realtime quota exhausted.",
    ProviderErrorCode.NETWORK_UNAVAILABLE: "OpenAI Realtime endpoint is unreachable.",
    ProviderErrorCode.TIMEOUT: "OpenAI Realtime did not respond in time.",
    ProviderErrorCode.CONNECTION_CLOSED: "OpenAI Realtime connection closed.",
    ProviderErrorCode.INVALID_REQUEST: "OpenAI Realtime request or configuration is invalid.",
    ProviderErrorCode.INVALID_RESPONSE: "OpenAI Realtime returned an unexpected response.",
    ProviderErrorCode.MODEL_UNAVAILABLE: "OpenAI Realtime transcription model is unavailable.",
    ProviderErrorCode.UNSUPPORTED_LANGUAGE: "OpenAI Realtime does not support the language hint.",
    ProviderErrorCode.INTERNAL_ERROR: "OpenAI Realtime reported an internal error.",
    ProviderErrorCode.CANCELLED: "OpenAI Realtime session was cancelled.",
}

_RETRYABLE = frozenset(
    {
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.NETWORK_UNAVAILABLE,
        ProviderErrorCode.TIMEOUT,
        ProviderErrorCode.CONNECTION_CLOSED,
        ProviderErrorCode.INTERNAL_ERROR,
    }
)

_CODE_MAP: Dict[str, ProviderErrorCode] = {
    "invalid_api_key": ProviderErrorCode.AUTH_FAILED,
    "unauthorized": ProviderErrorCode.AUTH_FAILED,
    "authentication_error": ProviderErrorCode.AUTH_FAILED,
    "permission_denied": ProviderErrorCode.AUTH_FAILED,
    "rate_limit_exceeded": ProviderErrorCode.RATE_LIMITED,
    "insufficient_quota": ProviderErrorCode.QUOTA_EXCEEDED,
    "model_not_found": ProviderErrorCode.MODEL_UNAVAILABLE,
    "unsupported_language": ProviderErrorCode.UNSUPPORTED_LANGUAGE,
    SESSION_EXPIRED_CODE: ProviderErrorCode.CONNECTION_CLOSED,
    "server_error": ProviderErrorCode.INTERNAL_ERROR,
}

_TYPE_MAP: Dict[str, ProviderErrorCode] = {
    "authentication_error": ProviderErrorCode.AUTH_FAILED,
    "permission_error": ProviderErrorCode.AUTH_FAILED,
    "rate_limit_error": ProviderErrorCode.RATE_LIMITED,
    "invalid_request_error": ProviderErrorCode.INVALID_REQUEST,
    "server_error": ProviderErrorCode.INTERNAL_ERROR,
}


def sanitize_provider_code(code: Optional[str]) -> Optional[str]:
    if code and _PROVIDER_CODE_RE.match(code):
        return code
    return None


def make_error(
    code: ProviderErrorCode,
    *,
    provider_code: Optional[str] = None,
    detail: Optional[str] = None,
    http_status: Optional[int] = None,
) -> ProviderError:
    details: Dict[str, Any] = {}
    safe_code = sanitize_provider_code(provider_code)
    if safe_code:
        details["provider_code"] = safe_code
    if detail:
        details["field"] = detail
    if http_status is not None:
        details["http_status"] = http_status
    return ProviderError(
        code=code,
        capability=ProviderCapability.STT,
        message=_SAFE_MESSAGES.get(code, "OpenAI Realtime error."),
        retryable=code in _RETRYABLE,
        details=details or None,
    )


def map_server_error(event: ServerError) -> ProviderError:
    code = _CODE_MAP.get(event.code or "")
    if code is None:
        code = _TYPE_MAP.get(event.error_type or "", ProviderErrorCode.INTERNAL_ERROR)
    return make_error(code, provider_code=event.code or event.error_type)


def map_http_status(status: int) -> ProviderError:
    if status in (401, 403):
        code = ProviderErrorCode.AUTH_FAILED
    elif status == 429:
        code = ProviderErrorCode.RATE_LIMITED
    elif status in (400, 404, 422):
        code = ProviderErrorCode.INVALID_REQUEST
    elif 500 <= status < 600:
        code = ProviderErrorCode.NETWORK_UNAVAILABLE
    else:
        code = ProviderErrorCode.INVALID_RESPONSE
    return make_error(code, http_status=status)


def is_benign_server_error(event: ServerError) -> bool:
    return (event.code or "") in BENIGN_ERROR_CODES


# ── Transcript assembly ──────────────────────────────────────────────────────


@dataclass
class _ItemState:
    item_id: str
    ordinal: int
    start_ms: float
    end_ms: float
    text: str = ""
    revision: int = 0
    is_final: bool = False
    discarded: bool = False
    end_fixed: bool = False
    vad_bound: bool = False


class TranscriptAssembler:
    """Maps provider item events to canonical, monotonically revised transcript results.

    One instance per EchoFlux stream. Invariants:

    - ``sequence`` is the item's ordinal in first-observed order and never changes.
    - ``provider_event_id`` is the provider ``item_id``, shared by every revision.
    - ``revision`` strictly increases per item; the final result is the last revision.
    - Events after an item's final, repeated ``event_id``s, and events for items
      discarded by ``reset`` produce nothing, so replays cannot regress a transcript.
    - Intervals are milliseconds of audio submitted on this stream. ``end_time_ms`` never
      moves backwards and never precedes ``start_time_ms``.
    """

    def __init__(
        self,
        stream_id: str,
        clock: Callable[[], float] = time.monotonic,
        max_items: int = 512,
        max_event_ids: int = 4096,
    ) -> None:
        self.stream_id = stream_id
        self._clock = clock
        self._max_items = max_items
        self._max_event_ids = max_event_ids
        self._items: OrderedDict[str, _ItemState] = OrderedDict()
        self._retired: OrderedDict[str, None] = OrderedDict()
        self._seen_event_ids: OrderedDict[str, None] = OrderedDict()
        self._pending_commits: Deque[Tuple[float, float]] = deque()
        self._next_ordinal = 0
        self._cursor_ms = 0.0
        self._boundary_ms = 0.0
        self._awaiting_clear = 0

    @property
    def cursor_ms(self) -> float:
        return self._cursor_ms

    def has_uncommitted_audio(self) -> bool:
        return self._cursor_ms > self._boundary_ms

    def advance_audio(self, duration_ms: float) -> None:
        if duration_ms > 0:
            self._cursor_ms += duration_ms

    def mark_commit(self) -> None:
        """Record the audio interval covered by a commit that is about to be sent."""
        self._pending_commits.append((self._boundary_ms, self._cursor_ms))
        self._boundary_ms = self._cursor_ms

    def drop_oldest_pending_commit(self) -> None:
        if self._pending_commits:
            self._pending_commits.popleft()

    def on_speech_started(self, item_id: str, start_ms: float) -> None:
        item = self._item(item_id)
        if item is None or item.is_final:
            return
        item.vad_bound = True
        item.start_ms = start_ms
        item.end_ms = max(item.end_ms, start_ms)

    def on_speech_stopped(self, item_id: str, end_ms: float) -> None:
        item = self._item(item_id)
        if item is None or item.is_final:
            return
        item.vad_bound = True
        item.end_ms = max(item.start_ms, end_ms)
        item.end_fixed = True

    def on_committed(self, item_id: str) -> None:
        if not self._pending_commits:
            return
        start_ms, end_ms = self._pending_commits.popleft()
        item = self._item(item_id)
        if item is None or item.is_final or item.vad_bound:
            return
        item.start_ms = start_ms
        item.end_ms = max(start_ms, end_ms)
        item.end_fixed = True

    def on_delta(self, event: TranscriptDelta) -> Optional[STTTranscriptResult]:
        if not self._accept_event_id(event.event_id) or not event.delta:
            return None
        item = self._item(event.item_id)
        if item is None or item.is_final or item.discarded:
            return None
        item.text += event.delta
        return self._emit(item, final=False)

    def on_completed(self, event: TranscriptCompleted) -> Optional[STTTranscriptResult]:
        if not self._accept_event_id(event.event_id):
            return None
        item = self._item(event.item_id)
        if item is None or item.is_final or item.discarded:
            return None
        item.text = event.transcript
        return self._emit(item, final=True)

    def on_failed(self, event: TranscriptFailed) -> Optional[STTTranscriptResult]:
        if not self._accept_event_id(event.event_id):
            return None
        item = self._item(event.item_id)
        if item is None or item.is_final or item.discarded:
            return None
        if item.text:
            # Keep what was already shown instead of retracting it.
            return self._emit(item, final=True)
        item.discarded = True
        return None

    def flush_pending(self) -> List[STTTranscriptResult]:
        """Finalize in-flight items whose provider session is gone."""
        results: List[STTTranscriptResult] = []
        for item in list(self._items.values()):
            if item.is_final or item.discarded:
                continue
            if item.text:
                results.append(self._emit(item, final=True))
            else:
                item.discarded = True
        self._pending_commits.clear()
        self._boundary_ms = self._cursor_ms
        # A clear sent to the dead session will never be acknowledged.
        self._awaiting_clear = 0
        return results

    def reset(self) -> None:
        """Discard in-flight items; ignore unseen items until the provider acks the clear."""
        for item in self._items.values():
            if not item.is_final:
                item.discarded = True
        self._pending_commits.clear()
        self._boundary_ms = self._cursor_ms
        self._awaiting_clear += 1

    def on_cleared(self) -> None:
        self._awaiting_clear = max(0, self._awaiting_clear - 1)

    def _accept_event_id(self, event_id: Optional[str]) -> bool:
        if event_id is None:
            return True
        if event_id in self._seen_event_ids:
            return False
        self._seen_event_ids[event_id] = None
        while len(self._seen_event_ids) > self._max_event_ids:
            self._seen_event_ids.popitem(last=False)
        return True

    def _item(self, item_id: str) -> Optional[_ItemState]:
        item = self._items.get(item_id)
        if item is not None:
            return item
        if item_id in self._retired:
            return None
        if self._awaiting_clear:
            # Audio committed before a reset still gets transcribed server-side.
            self._retire(item_id)
            return None
        item = _ItemState(
            item_id=item_id,
            ordinal=self._next_ordinal,
            start_ms=self._boundary_ms,
            end_ms=self._boundary_ms,
        )
        self._next_ordinal += 1
        self._items[item_id] = item
        self._prune()
        return item

    def _prune(self) -> None:
        if len(self._items) <= self._max_items:
            return
        for item_id, item in list(self._items.items()):
            if len(self._items) <= self._max_items:
                break
            if item.is_final or item.discarded:
                del self._items[item_id]
                self._retire(item_id)

    def _retire(self, item_id: str) -> None:
        self._retired[item_id] = None
        while len(self._retired) > self._max_event_ids:
            self._retired.popitem(last=False)

    def _emit(self, item: _ItemState, final: bool) -> STTTranscriptResult:
        if not item.end_fixed:
            item.end_ms = max(item.end_ms, self._cursor_ms)
        item.end_ms = max(item.end_ms, item.start_ms)
        item.revision += 1
        if final:
            item.is_final = True
            item.end_fixed = True
        return STTTranscriptResult(
            stream_id=self.stream_id,
            sequence=item.ordinal,
            provider_event_id=item.item_id,
            text=item.text,
            is_final=final,
            revision=item.revision,
            start_time_ms=item.start_ms,
            end_time_ms=item.end_ms,
            monotonic_timestamp=self._clock(),
            is_endpoint=final,
        )
