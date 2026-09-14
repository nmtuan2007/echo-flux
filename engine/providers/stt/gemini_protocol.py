"""Gemini Live wire protocol: setup/audio serialization and server event parsing.

Pure functions with no sockets and no retained secrets, so the adapter's protocol
assumptions stay testable in isolation. Checked on 2026-09-14 against:

- https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
- https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket
- https://ai.google.dev/gemini-api/docs/live-api/session-management
- https://ai.google.dev/api/live
"""

import base64
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine.providers.types import ProviderErrorCode

DEFAULT_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
# Ephemeral tokens are only accepted by the constrained method, as a query parameter.
EPHEMERAL_TOKEN_ENDPOINT = DEFAULT_ENDPOINT + "Constrained"
DEFAULT_MODEL = "gemini-3.5-transcribe-live"
API_KEY_HEADER = "x-goog-api-key"
DEFAULT_SAMPLE_RATE = 16000
PCM16_SAMPLE_WIDTH_BYTES = 2
MAX_CUSTOM_VOCABULARY = 1000
TRANSCRIPTION_MODES = frozenset({"SMART", "VERBATIM"})
REDACTED = "***"

RawFrame = Union[str, bytes, bytearray]  # noqa: UP007 -- typing house style (rule 04)

SECRET_QUERY_PARAMS = frozenset({"key", "api_key", "access_token", "token"})
_AUTO_LANGUAGE_VALUES = frozenset({"", "auto", "detect", "multi"})
_SECRET_QUERY_RE = re.compile(r"(?i)\b(key|api_key|access_token|token)=([^&\s\"']+)")
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


class GeminiProtocolError(ValueError):
    """A server frame that cannot be interpreted as a Live API message."""


class ServerEventKind(str, Enum):
    """Server message parts the STT adapter acts on."""

    SETUP_COMPLETE = "setup_complete"
    INTERIM = "interim"
    FINAL = "final"
    TURN_COMPLETE = "turn_complete"
    GO_AWAY = "go_away"
    ERROR = "error"


@dataclass(frozen=True)
class ServerEvent:
    """One interpreted part of a server frame. A frame may carry several."""

    kind: ServerEventKind
    text: str = ""
    language_code: Optional[str] = None
    time_left_s: Optional[float] = None
    error_code: Optional[int] = None
    error_status: str = ""
    error_message: str = ""


# ── Client messages ──────────────────────────────────────────────────────────


def normalize_model_name(model: Optional[str]) -> str:
    """Return the resource-style model name the setup message requires."""
    name = (model or "").strip() or DEFAULT_MODEL
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return f"models/{name}"


def normalize_language_hints(hints: Optional[Sequence[str]]) -> List[str]:
    """Drop auto-detect sentinels and duplicates. An empty result means auto language."""
    result: List[str] = []
    for hint in hints or ():
        code = str(hint).strip()
        if code.lower() in _AUTO_LANGUAGE_VALUES or code in result:
            continue
        result.append(code)
    return result


def normalize_vocabulary(terms: Optional[Sequence[str]]) -> List[str]:
    result: List[str] = []
    for term in terms or ():
        value = str(term).strip()
        if value and value not in result:
            result.append(value)
    if len(result) > MAX_CUSTOM_VOCABULARY:
        raise ValueError(
            f"custom vocabulary has {len(result)} terms; Gemini Live accepts at most "
            f"{MAX_CUSTOM_VOCABULARY}"
        )
    return result


def build_setup_message(
    model: Optional[str],
    language_hints: Optional[Sequence[str]] = None,
    custom_vocabulary: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    manual_activity: bool = False,
) -> Dict[str, Any]:
    """Build the setup message, which must be the first frame on every connection.

    With no language hints ``languageCodes`` is omitted so the service auto-detects.
    """
    transcription: Dict[str, Any] = {}
    languages = normalize_language_hints(language_hints)
    if languages:
        transcription["languageCodes"] = languages
    vocabulary = normalize_vocabulary(custom_vocabulary)
    if vocabulary:
        transcription["customVocabulary"] = vocabulary
    if mode:
        normalized_mode = mode.strip().upper()
        if normalized_mode not in TRANSCRIPTION_MODES:
            raise ValueError(f"unsupported transcription mode: {mode!r}")
        transcription["mode"] = normalized_mode

    setup: Dict[str, Any] = {
        "model": normalize_model_name(model),
        "generationConfig": {"responseModalities": ["TEXT"]},
        "inputAudioTranscription": transcription,
    }
    if manual_activity:
        setup["realtimeInputConfig"] = {"automaticActivityDetection": {"disabled": True}}
    return {"setup": setup}


def build_audio_message(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, Any]:
    """Wrap little-endian PCM16 mono bytes as a realtime audio input."""
    return {
        "realtimeInput": {
            "audio": {
                "data": base64.b64encode(pcm).decode("ascii"),
                "mimeType": f"audio/pcm;rate={int(sample_rate)}",
            }
        }
    }


def build_audio_stream_end() -> Dict[str, Any]:
    """Flush cached audio when automatic activity detection is on."""
    return {"realtimeInput": {"audioStreamEnd": True}}


def build_activity_start() -> Dict[str, Any]:
    return {"realtimeInput": {"activityStart": {}}}


def build_activity_end() -> Dict[str, Any]:
    return {"realtimeInput": {"activityEnd": {}}}


def encode_message(message: Mapping[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


# ── Server messages ──────────────────────────────────────────────────────────


def _field(container: Mapping[str, Any], camel: str, snake: str) -> Any:
    # The wire format is camelCase, but proto-JSON readers accept both spellings.
    if camel in container:
        return container[camel]
    return container.get(snake)


def _transcription(content: Mapping[str, Any], camel: str, snake: str) -> Optional[Tuple]:
    value = _field(content, camel, snake)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise GeminiProtocolError(f"{camel} is not an object")
    text = value.get("text", "")
    if not isinstance(text, str):
        raise GeminiProtocolError(f"{camel}.text is not a string")
    language = _field(value, "languageCode", "language_code")
    return text, language if isinstance(language, str) and language else None


def parse_duration_seconds(value: Any) -> Optional[float]:
    """Parse a protobuf Duration ("10s", "1.5s" or {"seconds", "nanos"})."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _DURATION_RE.match(value)
        return float(match.group(1)) if match else None
    if isinstance(value, dict):
        try:
            return float(value.get("seconds", 0)) + float(value.get("nanos", 0)) / 1e9
        except (TypeError, ValueError):
            return None
    return None


def parse_server_message(raw: RawFrame) -> List[ServerEvent]:
    """Interpret one server frame. Parts the STT adapter does not use are ignored.

    Raises GeminiProtocolError for frames that are not a JSON object or whose known
    fields have the wrong shape.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raise GeminiProtocolError("frame is not UTF-8") from None
    try:
        payload = json.loads(raw)
    except ValueError:
        raise GeminiProtocolError("frame is not valid JSON") from None
    if not isinstance(payload, dict):
        raise GeminiProtocolError("frame is not a JSON object")

    events: List[ServerEvent] = []
    if _field(payload, "setupComplete", "setup_complete") is not None:
        events.append(ServerEvent(ServerEventKind.SETUP_COMPLETE))

    error = payload.get("error")
    if error is not None:
        if not isinstance(error, dict):
            raise GeminiProtocolError("error is not an object")
        code = error.get("code")
        events.append(
            ServerEvent(
                ServerEventKind.ERROR,
                error_code=code if isinstance(code, int) else None,
                error_status=str(error.get("status") or ""),
                error_message=str(error.get("message") or ""),
            )
        )

    content = _field(payload, "serverContent", "server_content")
    if content is not None:
        if not isinstance(content, dict):
            raise GeminiProtocolError("serverContent is not an object")
        interim = _transcription(
            content, "interimInputTranscription", "interim_input_transcription"
        )
        final = _transcription(content, "inputTranscription", "input_transcription")
        if interim is not None:
            events.append(ServerEvent(ServerEventKind.INTERIM, interim[0], interim[1]))
        if final is not None:
            events.append(ServerEvent(ServerEventKind.FINAL, final[0], final[1]))
        if _field(content, "turnComplete", "turn_complete") is True:
            events.append(ServerEvent(ServerEventKind.TURN_COMPLETE))

    go_away = _field(payload, "goAway", "go_away")
    if go_away is not None:
        if not isinstance(go_away, dict):
            raise GeminiProtocolError("goAway is not an object")
        time_left = parse_duration_seconds(_field(go_away, "timeLeft", "time_left"))
        events.append(ServerEvent(ServerEventKind.GO_AWAY, time_left_s=time_left))
    return events


# ── Error classification ─────────────────────────────────────────────────────


def classify_close(code: Optional[int], reason: str = "") -> Tuple[ProviderErrorCode, bool]:
    """Map a WebSocket close (or in-band error status) to a code and retryability.

    Gemini reports rejections as close frames whose reason carries the status text,
    so the reason is inspected before the numeric code.
    """
    text = (reason or "").lower()
    if any(
        token in text
        for token in ("api key", "api_key", "unauthenticated", "permission", "credential")
    ):
        return ProviderErrorCode.AUTH_FAILED, False
    if "quota" in text or "resource_exhausted" in text or "billing" in text:
        return ProviderErrorCode.QUOTA_EXCEEDED, False
    if "rate limit" in text or "too many requests" in text:
        return ProviderErrorCode.RATE_LIMITED, True
    if "model" in text and any(t in text for t in ("not found", "not supported", "unsupported")):
        return ProviderErrorCode.MODEL_UNAVAILABLE, False
    if "language" in text and any(t in text for t in ("not supported", "unsupported")):
        return ProviderErrorCode.UNSUPPORTED_LANGUAGE, False
    if "deadline" in text or "timeout" in text:
        return ProviderErrorCode.TIMEOUT, True
    if code in (1007, 1008, 1003, 1009):
        return ProviderErrorCode.INVALID_REQUEST, False
    if code in (1011, 1012, 1013):
        return ProviderErrorCode.INTERNAL_ERROR, True
    return ProviderErrorCode.CONNECTION_CLOSED, True


def classify_http_status(status: Optional[int]) -> Tuple[ProviderErrorCode, bool]:
    """Map a rejected WebSocket handshake to a code and retryability."""
    if status is None:
        return ProviderErrorCode.NETWORK_UNAVAILABLE, True
    if status in (401, 403):
        return ProviderErrorCode.AUTH_FAILED, False
    if status == 404:
        return ProviderErrorCode.MODEL_UNAVAILABLE, False
    if status == 429:
        return ProviderErrorCode.RATE_LIMITED, True
    if status >= 500:
        return ProviderErrorCode.INTERNAL_ERROR, True
    return ProviderErrorCode.INVALID_REQUEST, False


# ── Secret handling ──────────────────────────────────────────────────────────


def build_connect_url(endpoint: str, access_token: Optional[str] = None) -> str:
    """Return the socket URL. Only ephemeral tokens travel in the query string."""
    if not access_token:
        return endpoint
    parts = urlsplit(endpoint)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    query.append(("access_token", access_token))
    return urlunsplit(parts._replace(query=urlencode(query)))


def redact_url(url: str) -> str:
    """Mask secret-bearing query parameters so the URL is safe to log."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [
        (k, REDACTED if k.lower() in SECRET_QUERY_PARAMS else v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query, safe="*")))


def redact_text(text: str, secrets: Sequence[str] = ()) -> str:
    """Mask known secret values and ``key=``-style query fragments in free text."""
    result = text or ""
    for secret in secrets:
        if secret and len(secret) >= 4:
            result = result.replace(secret, REDACTED)
    return _SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", result)
