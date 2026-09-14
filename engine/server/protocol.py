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
from typing import Any, Dict, MutableMapping, Optional

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


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
