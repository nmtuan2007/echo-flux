"""Transport-layer protocol for the authenticated engine socket.

Frames defined here wrap the legacy application messages; they are not application
message types. Transport frames are discriminated by an ``op`` key so they can never be
confused with the ``type``-keyed messages documented in ``.agent/rules/03-architecture.md``.

Handshake (first frame on every connection when bootstrap tokens are present)::

    client -> {"op": "auth", "role": "control" | "event", "token": "<hex>", "protocol": 1}
    server -> {"op": "auth_ok", "role": "...", "protocol": 1}
    server -> {"op": "error", "code": "auth_failed" | ..., "message": "..."} then close

Control role::

    client -> {"op": "request", "id": "<request id>", "command": {<legacy command>}}
    server -> {"op": "response", "id": "<request id>", "message": {<legacy reply>}}   (0..n)
    server -> {"op": "complete", "id": "<request id>"}

Event role::

    server -> {"op": "event", "event": {<sanitized legacy event>}}

Error text is fixed per code. Nothing a client sends is ever echoed back, and token
values are scrubbed from every outbound frame.
"""

import hmac
import ipaddress
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple

PROTOCOL_VERSION = 1

ENV_CONTROL_TOKEN = "ECHOFLUX_CONTROL_TOKEN"
ENV_EVENT_TOKEN = "ECHOFLUX_EVENT_TOKEN"

AUTH_TIMEOUT_SECONDS = 5.0
MIN_TOKEN_LENGTH = 32
MAX_TOKEN_LENGTH = 512

# 4xxx close codes are reserved for applications by RFC 6455.
CLOSE_BAD_HANDSHAKE = 4400
CLOSE_AUTH_FAILED = 4401
CLOSE_AUTH_TIMEOUT = 4408

OP_AUTH = "auth"
OP_AUTH_OK = "auth_ok"
OP_REQUEST = "request"
OP_RESPONSE = "response"
OP_COMPLETE = "complete"
OP_EVENT = "event"
OP_ERROR = "error"

REDACTED = "[redacted]"

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TOKEN_RE = re.compile(r"^[\x21-\x7e]+$")

# Matched against a key lowercased with "_" and "-" removed.
_SECRET_KEY_FRAGMENTS = (
    "apikey",
    "secret",
    "password",
    "authorization",
    "credential",
    "bearer",
)


class Role(str, Enum):
    CONTROL = "control"
    EVENT = "event"
    LEGACY = "legacy"


class SecurityMode(str, Enum):
    AUTHENTICATED = "authenticated"
    # Implicit, deprecated path used when bootstrap tokens are absent. Removed by EF-031.
    LEGACY_TRANSITIONAL = "legacy_transitional"
    INSECURE_DEV = "insecure_dev"


class BootstrapError(ValueError):
    """Bootstrap tokens are malformed. Messages name variables, never values."""


class ProtocolError(Exception):
    """A client frame was rejected. ``message`` is safe to send back to the client."""

    def __init__(
        self,
        code: str,
        message: str,
        close_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(code)
        self.code = code
        self.message = message
        self.close_code = close_code
        self.request_id = request_id


@dataclass(frozen=True)
class BootstrapTokens:
    control: str = field(repr=False)
    event: str = field(repr=False)

    def __post_init__(self):
        for name, value in ((ENV_CONTROL_TOKEN, self.control), (ENV_EVENT_TOKEN, self.event)):
            if not (MIN_TOKEN_LENGTH <= len(value) <= MAX_TOKEN_LENGTH):
                raise BootstrapError(
                    f"{name} must be {MIN_TOKEN_LENGTH}-{MAX_TOKEN_LENGTH} characters"
                )
            if not _TOKEN_RE.match(value):
                raise BootstrapError(f"{name} must be printable ASCII without whitespace")
        # Equal tokens would let an event client authenticate as control.
        if hmac.compare_digest(self.control.encode(), self.event.encode()):
            raise BootstrapError(f"{ENV_CONTROL_TOKEN} and {ENV_EVENT_TOKEN} must differ")

    def __repr__(self) -> str:
        return "BootstrapTokens(<redacted>)"

    @classmethod
    def from_environ(
        cls, environ: MutableMapping[str, str], consume: bool = True
    ) -> Optional["BootstrapTokens"]:
        """Read the tokens Tauri passes to the engine process.

        Returns None when neither is set, which selects the legacy path. Setting only one
        is a misconfiguration and fails closed. With ``consume`` the variables are removed
        so child processes and later environment dumps never see them.
        """
        control = environ.get(ENV_CONTROL_TOKEN) or ""
        event = environ.get(ENV_EVENT_TOKEN) or ""
        if consume:
            environ.pop(ENV_CONTROL_TOKEN, None)
            environ.pop(ENV_EVENT_TOKEN, None)

        if not control and not event:
            return None
        if not control or not event:
            raise BootstrapError(f"{ENV_CONTROL_TOKEN} and {ENV_EVENT_TOKEN} must be set together")
        return cls(control=control, event=event)

    def verify(self, role: Role, presented: str) -> bool:
        if role is Role.CONTROL:
            expected = self.control
        elif role is Role.EVENT:
            expected = self.event
        else:
            return False
        return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))

    def redact(self, text: str) -> str:
        return text.replace(self.control, REDACTED).replace(self.event, REDACTED)


@dataclass(frozen=True)
class AuthRequest:
    role: Role
    token: str = field(repr=False)


@dataclass(frozen=True)
class ControlRequest:
    request_id: str
    command: Dict[str, Any] = field(repr=False)


def _decode_object(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def parse_auth_frame(raw: Any) -> AuthRequest:
    frame = _decode_object(raw)
    if frame is None or frame.get("op") != OP_AUTH:
        raise ProtocolError(
            "auth_required", "Authenticate before sending commands", CLOSE_AUTH_FAILED
        )

    protocol = frame.get("protocol", PROTOCOL_VERSION)
    if protocol != PROTOCOL_VERSION:
        raise ProtocolError(
            "unsupported_protocol", "Unsupported protocol version", CLOSE_BAD_HANDSHAKE
        )

    # Unknown roles and bad tokens share one code so the reply is not an oracle.
    role_value = frame.get("role")
    token = frame.get("token")
    if role_value not in (Role.CONTROL.value, Role.EVENT.value) or not isinstance(token, str):
        raise ProtocolError("auth_failed", "Authentication failed", CLOSE_AUTH_FAILED)
    return AuthRequest(role=Role(role_value), token=token)


def parse_control_frame(raw: Any) -> ControlRequest:
    frame = _decode_object(raw)
    if frame is None or frame.get("op") != OP_REQUEST:
        raise ProtocolError("bad_request", "Expected a request envelope")

    request_id = frame.get("id")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.match(request_id):
        raise ProtocolError("bad_request_id", "Request id must match [A-Za-z0-9._:-]{1,128}")

    command = frame.get("command")
    if not isinstance(command, dict):
        raise ProtocolError(
            "bad_request", "Request command must be an object", request_id=request_id
        )
    return ControlRequest(request_id=request_id, command=command)


def auth_ok_frame(role: Role) -> Dict[str, Any]:
    return {"op": OP_AUTH_OK, "role": role.value, "protocol": PROTOCOL_VERSION}


def response_frame(request_id: str, message: Any) -> Dict[str, Any]:
    return {"op": OP_RESPONSE, "id": request_id, "message": message}


def complete_frame(request_id: str) -> Dict[str, Any]:
    return {"op": OP_COMPLETE, "id": request_id}


def event_frame(event: Any) -> Dict[str, Any]:
    return {"op": OP_EVENT, "event": event}


def error_frame(code: str, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    frame: Dict[str, Any] = {"op": OP_ERROR, "code": code, "message": message}
    if request_id is not None:
        frame["id"] = request_id
    return frame


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("_", "").replace("-", "")
    if normalized.endswith("token") or normalized.endswith("config"):
        return True
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def sanitize_event(value: Any) -> Any:
    """Copy of an outbound event with secret-bearing keys redacted, recursively."""
    if isinstance(value, dict):
        return {
            key: REDACTED if isinstance(key, str) and _is_secret_key(key) else sanitize_event(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_event(item) for item in value]
    return value


def encode_frame(frame: Dict[str, Any], tokens: Optional[BootstrapTokens]) -> str:
    payload = json.dumps(frame)
    return tokens.redact(payload) if tokens is not None else payload


# ---------------------------------------------------------------------------------------------
# Assistant command payloads
#
# Repository-backed Assist and Summary travel inside the two documented commands,
# ``request_suggestion`` and ``request_summary``; they add no message types. A command that
# carries ``conversation_id`` selects the repository-backed path and may name ids only: the
# engine assembles every byte of context itself. A command without it is the transitional
# legacy shape (client-assembled context plus ``llm_config``) until EF-031 removes it.
#
# Replies keep the legacy terminal types. ``request_suggestion`` always ends with exactly one
# ``suggestion_result``; ``request_summary`` always ends with exactly one ``llm_done``, after
# zero or more ``llm_chunk``. Structured data is added as extra keys only.
# ---------------------------------------------------------------------------------------------

ASSIST_COMMAND_KEYS = frozenset({"type", "conversation_id", "message_id", "entry_id"})
SUMMARY_COMMAND_KEYS = frozenset({"type", "conversation_id"})

ERROR_INVALID_REQUEST = "invalid_request"
ERROR_FORBIDDEN = "forbidden"
ERROR_ASSISTANT_UNAVAILABLE = "assistant_unavailable"
ERROR_CANCELLED = "cancelled"
ERROR_INTERNAL = "internal_error"

ASSISTANT_ERROR_MESSAGES = {
    ERROR_INVALID_REQUEST: "Assistant request is invalid.",
    ERROR_FORBIDDEN: "Conversation requests require an authenticated control connection.",
    ERROR_ASSISTANT_UNAVAILABLE: "The Assistant is not configured for stored conversations.",
    ERROR_CANCELLED: "The Assistant request was cancelled.",
    ERROR_INTERNAL: "The Assistant request failed unexpectedly.",
}

LEGACY_SUMMARY_ERROR_PREFIX = "**Error:** "

# Same shape as conversation record ids.
_ASSISTANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class AssistCommand:
    conversation_id: str
    message_id: str


@dataclass(frozen=True)
class SummaryCommand:
    conversation_id: str


def is_conversation_command(command: Dict[str, Any]) -> bool:
    """True when the command selects the repository-backed path."""
    return isinstance(command, dict) and "conversation_id" in command


def _assistant_id(value: Any) -> str:
    if not isinstance(value, str) or not _ASSISTANT_ID_RE.match(value):
        raise ProtocolError(ERROR_INVALID_REQUEST, ASSISTANT_ERROR_MESSAGES[ERROR_INVALID_REQUEST])
    return value


def _reject_extra_keys(command: Dict[str, Any], allowed: frozenset) -> None:
    # Client-assembled context (context, target_text, entries, llm_config, ...) is refused
    # outright rather than ignored, so a client can never believe it steered the prompt.
    if set(command) - allowed:
        raise ProtocolError(ERROR_INVALID_REQUEST, ASSISTANT_ERROR_MESSAGES[ERROR_INVALID_REQUEST])


def parse_assist_command(command: Dict[str, Any]) -> AssistCommand:
    """``{"type": "request_suggestion", "conversation_id", "message_id"}``.

    ``entry_id`` is accepted as an alias of ``message_id``; when both are present they must
    agree. Raises ``ProtocolError`` with a fixed message.
    """
    _reject_extra_keys(command, ASSIST_COMMAND_KEYS)
    conversation_id = _assistant_id(command.get("conversation_id"))
    message_id = command.get("message_id", command.get("entry_id"))
    message_id = _assistant_id(message_id)
    if "entry_id" in command and command.get("entry_id") != message_id:
        raise ProtocolError(ERROR_INVALID_REQUEST, ASSISTANT_ERROR_MESSAGES[ERROR_INVALID_REQUEST])
    return AssistCommand(conversation_id=conversation_id, message_id=message_id)


def parse_summary_command(command: Dict[str, Any]) -> SummaryCommand:
    """``{"type": "request_summary", "conversation_id"}``. Raises ``ProtocolError``."""
    _reject_extra_keys(command, SUMMARY_COMMAND_KEYS)
    return SummaryCommand(conversation_id=_assistant_id(command.get("conversation_id")))


def safe_command_id(command: Dict[str, Any], key: str) -> str:
    """The id to echo in an error reply: the client's value only if it is a valid id."""
    value = command.get(key) if isinstance(command, dict) else None
    return value if isinstance(value, str) and _ASSISTANT_ID_RE.match(value) else ""


def suggestion_result_message(
    conversation_id: str,
    message_id: str,
    assist_id: str,
    suggested_replies: Sequence[Dict[str, Any]],
    clarifying_questions: Sequence[Dict[str, Any]],
    provider: Optional[Dict[str, Any]] = None,
    context_window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Successful repository-backed Assist.

    ``entry_id`` and ``options`` keep the legacy shape the current store renders; the
    structured groups and metadata are additive.
    """
    return {
        "type": "suggestion_result",
        "entry_id": message_id,
        "options": [
            {"strategy": reply.get("strategy"), "text": reply.get("text")}
            for reply in suggested_replies
        ],
        "conversation_id": conversation_id,
        "message_id": message_id,
        "assist_id": assist_id,
        "suggested_replies": list(suggested_replies),
        "clarifying_questions": list(clarifying_questions),
        "provider": provider,
        "context_window": context_window or {},
    }


def suggestion_error_message(
    message_id: str,
    code: str,
    message: str,
    conversation_id: Optional[str] = None,
    retryable: bool = False,
    provider_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Failed Assist. ``error`` stays a display string, as in the legacy shape."""
    reply: Dict[str, Any] = {
        "type": "suggestion_result",
        "entry_id": message_id,
        "error": message,
        "error_code": code,
        "retryable": retryable,
    }
    if conversation_id is not None:
        reply["conversation_id"] = conversation_id
        reply["message_id"] = message_id
    if provider_code is not None:
        reply["provider_code"] = provider_code
    return reply


def summary_chunk_message(
    text: str, conversation_id: Optional[str] = None, is_error: bool = False
) -> Dict[str, Any]:
    reply: Dict[str, Any] = {"type": "llm_chunk", "text": text}
    if conversation_id is not None:
        reply["conversation_id"] = conversation_id
    if is_error:
        reply["is_error"] = True
    return reply


def summary_done_message(
    conversation_id: Optional[str] = None,
    summary_id: Optional[str] = None,
    warning: Optional[Dict[str, Any]] = None,
    provider: Optional[Dict[str, Any]] = None,
    included_message_count: Optional[int] = None,
    omitted_message_count: Optional[int] = None,
    error_code: Optional[str] = None,
    error: Optional[str] = None,
    retryable: bool = False,
    provider_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Terminal Summary reply. The bare legacy form is ``{"type": "llm_done"}``."""
    reply: Dict[str, Any] = {"type": "llm_done"}
    if conversation_id is None:
        return reply
    reply.update(
        {
            "conversation_id": conversation_id,
            "summary_id": summary_id,
            "warning": warning,
            "provider": provider,
            "included_message_count": included_message_count,
            "omitted_message_count": omitted_message_count,
        }
    )
    if error_code is not None:
        reply.update({"error_code": error_code, "error": error, "retryable": retryable})
        if provider_code is not None:
            reply["provider_code"] = provider_code
    return reply


def summary_error_replies(
    code: str,
    message: str,
    conversation_id: str,
    after_chunks: bool = False,
    retryable: bool = False,
    provider_code: Optional[str] = None,
    warning: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Legacy-compatible failure: an error ``llm_chunk`` then the terminal ``llm_done``."""
    prefix = "\n\n" if after_chunks else ""
    chunk = summary_chunk_message(
        prefix + LEGACY_SUMMARY_ERROR_PREFIX + message, conversation_id, is_error=True
    )
    done = summary_done_message(
        conversation_id=conversation_id,
        warning=warning,
        error_code=code,
        error=message,
        retryable=retryable,
        provider_code=provider_code,
    )
    return chunk, done


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
