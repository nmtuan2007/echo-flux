"""Bounded async client for OpenAI-compatible Chat Completions text APIs.

Owns transport, authentication, and request/response shape only. It never builds prompts,
chooses fallback, or knows about sessions; adapters pass fully built messages and
already-resolved endpoint, model, and credential values.

Modes are typed endpoint strategies rather than conditionals:

==========================  =====================================================  ==========================  =========================
Strategy                    URL                                                    Auth header                 Output token field
==========================  =====================================================  ==========================  =========================
OpenAIStrategy              ``{base_url}/chat/completions``                        ``Authorization: Bearer``   ``max_completion_tokens``
AzureOpenAIStrategy (v1)    ``{resource}/openai/v1/chat/completions``              ``api-key``                 ``max_completion_tokens``
AzureOpenAIStrategy (date)  ``{resource}/openai/deployments/{d}/chat/completions``  ``api-key``                 ``max_completion_tokens``
                            ``?api-version={v}``
AssemblyAIGatewayStrategy   ``{base_url}/chat/completions``                        ``Authorization: <key>``    ``max_tokens``
LocalOpenAICompatible...    ``{base_url}/chat/completions``                        optional Bearer             ``max_tokens``
CustomOpenAICompatible...   ``{base_url}/chat/completions``                        optional Bearer + secret    configurable
                                                                                   headers
==========================  =====================================================  ==========================  =========================

Bounds and safety
- Every request has connect/read/write/pool timeouts and a total deadline.
- Response bodies and SSE streams are read incrementally and cut off at a byte limit.
- Chat completion POSTs are not idempotent (they bill and may run), so they are retried only
  when the request never left this process (connect failure or pool timeout). The GET probe
  is idempotent and is also retried on 408/429/5xx with capped backoff.
- Streaming responses are closed in ``finally``; cancelling or abandoning the iterator
  releases the connection immediately.
- Errors carry a stable code, HTTP status, and a sanitized provider code. Response bodies,
  URLs, headers, and exception text never reach errors or logs, and every configured secret
  value is scrubbed from anything that could.
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from engine.core.config import endpoint_has_embedded_credentials, is_secret_key_name
from engine.core.logging import get_logger
from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderRuntimeError,
)

logger = get_logger("providers.text.openai_compatible")

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
AZURE_V1_API_VERSIONS = frozenset({"v1", "preview"})
MAX_TOKENS = "max_tokens"
MAX_COMPLETION_TOKENS = "max_completion_tokens"
_REDACTED = "***"
_MIN_SCRUB_LENGTH = 4

_PROVIDER_CODE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~\-]{1,128}$")
_DEPLOYMENT_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
_API_VERSION_RE = re.compile(r"^[A-Za-z0-9.\-]{1,32}$")
_ROLES = frozenset({"system", "developer", "user", "assistant"})

# Header names that always carry credentials; they may only arrive as secret values.
_SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "cookie",
        "set-cookie",
        "ocp-apim-subscription-key",
    }
)
_RESERVED_HEADER_NAMES = frozenset({"content-type", "accept", "host", "content-length"})


# ── Messages and results ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextMessage:
    """One chat message. Content is plain text; adapters decide what goes in it."""

    role: str
    content: str


@dataclass(frozen=True)
class TextGenerationOptions:
    """Resolved generation options. ``None`` omits the field so the server default applies."""

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


@dataclass(frozen=True)
class TextCompletion:
    """Canonical non-streaming (or fully assembled streaming) completion."""

    text: str
    response_id: Optional[str] = None
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass(frozen=True)
class TextStreamChunk:
    """Streamed delta. The last chunk has ``is_done`` and carries the assembled completion."""

    delta_text: str
    is_done: bool = False
    completion: Optional[TextCompletion] = None


@dataclass(frozen=True)
class TextTransportOptions:
    """Transport bounds. Provider settings (endpoint, model, keys) are never defaulted here."""

    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    write_timeout_s: float = 10.0
    pool_timeout_s: float = 10.0
    total_timeout_s: float = 120.0
    stream_idle_timeout_s: float = 30.0
    max_response_bytes: int = 2 * 1024 * 1024
    max_stream_bytes: int = 8 * 1024 * 1024
    max_sse_line_bytes: int = 256 * 1024
    max_error_body_bytes: int = 64 * 1024
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    max_retry_after_s: float = 10.0


# ── Errors and redaction ─────────────────────────────────────────────────────


_RETRYABLE = frozenset(
    {
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.NETWORK_UNAVAILABLE,
        ProviderErrorCode.TIMEOUT,
        ProviderErrorCode.CONNECTION_CLOSED,
        ProviderErrorCode.INTERNAL_ERROR,
    }
)

_SAFE_MESSAGES: Dict[ProviderErrorCode, str] = {
    ProviderErrorCode.AUTH_FAILED: "The text provider rejected the credentials.",
    ProviderErrorCode.RATE_LIMITED: "The text provider is rate limiting requests.",
    ProviderErrorCode.QUOTA_EXCEEDED: "The text provider quota is exhausted.",
    ProviderErrorCode.NETWORK_UNAVAILABLE: "The text provider could not be reached.",
    ProviderErrorCode.TIMEOUT: "The text provider did not respond in time.",
    ProviderErrorCode.CONNECTION_CLOSED: "The text provider closed the response early.",
    ProviderErrorCode.INVALID_REQUEST: "The text provider rejected the request.",
    ProviderErrorCode.INVALID_RESPONSE: "The text provider returned an unusable response.",
    ProviderErrorCode.MODEL_UNAVAILABLE: "The requested model or deployment is unavailable.",
    ProviderErrorCode.CONTEXT_OVERFLOW: "The request exceeds the model context window.",
    ProviderErrorCode.INTERNAL_ERROR: "The text provider reported an internal error.",
    ProviderErrorCode.CANCELLED: "The text request was cancelled.",
}

_PROVIDER_CODE_MAP: Dict[str, ProviderErrorCode] = {
    "invalid_api_key": ProviderErrorCode.AUTH_FAILED,
    "unauthorized": ProviderErrorCode.AUTH_FAILED,
    "authentication_error": ProviderErrorCode.AUTH_FAILED,
    "permission_error": ProviderErrorCode.AUTH_FAILED,
    "rate_limit_exceeded": ProviderErrorCode.RATE_LIMITED,
    "rate_limit_error": ProviderErrorCode.RATE_LIMITED,
    "insufficient_quota": ProviderErrorCode.QUOTA_EXCEEDED,
    "context_length_exceeded": ProviderErrorCode.CONTEXT_OVERFLOW,
    "string_above_max_length": ProviderErrorCode.CONTEXT_OVERFLOW,
    "model_not_found": ProviderErrorCode.MODEL_UNAVAILABLE,
    "deploymentnotfound": ProviderErrorCode.MODEL_UNAVAILABLE,
    "invalid_request_error": ProviderErrorCode.INVALID_REQUEST,
    "server_error": ProviderErrorCode.INTERNAL_ERROR,
}


def sanitize_provider_code(code: Any, secrets: Sequence[str] = ()) -> Optional[str]:
    """A short identifier-shaped provider code, or ``None``. Never echoes a secret."""
    if not isinstance(code, str) or not _PROVIDER_CODE_RE.match(code):
        return None
    if any(secret and len(secret) >= _MIN_SCRUB_LENGTH and secret in code for secret in secrets):
        return None
    return code


def scrub_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace every occurrence of each configured secret value."""
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        if len(secret) >= _MIN_SCRUB_LENGTH:
            text = text.replace(secret, _REDACTED)
    return text


def redact_url(url: str) -> str:
    """Scheme, host, port, and path only: userinfo and query strings are dropped."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<invalid endpoint>"
    return f"{parts.scheme}://{host}{port}{parts.path}"


def make_text_error(
    code: ProviderErrorCode,
    capability: ProviderCapability,
    *,
    http_status: Optional[int] = None,
    provider_code: Optional[str] = None,
    field_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> ProviderError:
    details: Dict[str, Any] = {}
    if mode:
        details["mode"] = mode
    if http_status is not None:
        details["http_status"] = http_status
    if provider_code:
        details["provider_code"] = provider_code
    if field_name:
        details["field"] = field_name
    return ProviderError(
        code=code,
        capability=capability,
        message=_SAFE_MESSAGES.get(code, "Text provider error."),
        retryable=code in _RETRYABLE,
        details=details or None,
    )


def classify_http_status(status: int) -> ProviderErrorCode:
    if status in (401, 403):
        return ProviderErrorCode.AUTH_FAILED
    if status == 404:
        return ProviderErrorCode.MODEL_UNAVAILABLE
    if status == 408:
        return ProviderErrorCode.TIMEOUT
    if status == 413:
        return ProviderErrorCode.CONTEXT_OVERFLOW
    if status == 429:
        return ProviderErrorCode.RATE_LIMITED
    if 400 <= status < 500:
        return ProviderErrorCode.INVALID_REQUEST
    if status in (502, 503, 504):
        return ProviderErrorCode.NETWORK_UNAVAILABLE
    if status >= 500:
        return ProviderErrorCode.INTERNAL_ERROR
    return ProviderErrorCode.INVALID_RESPONSE


class TextConfigError(ValueError):
    """Invalid strategy or request input. Names the field only, never its value."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"invalid text provider config: field={field_name}")
        self.field_name = field_name


def config_error_to_provider_error(
    exc: TextConfigError, capability: ProviderCapability, mode: Optional[str] = None
) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        make_text_error(
            ProviderErrorCode.INVALID_REQUEST, capability, field_name=exc.field_name, mode=mode
        )
    )


def _invalid_config(field_name: str) -> TextConfigError:
    # Strategies are capability-agnostic; the client or adapter attaches the capability.
    return TextConfigError(field_name)


# ── Endpoint strategies ──────────────────────────────────────────────────────


def _validate_base_url(url: Any, field_name: str, *, allow_insecure: bool) -> str:
    """Normalized URL without trailing slash; embedded credentials are rejected outright."""
    if not isinstance(url, str) or not url.strip():
        raise _invalid_config(field_name)
    candidate = url.strip()
    try:
        parts = urlsplit(candidate)
        hostname = parts.hostname
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError:
        raise _invalid_config(field_name) from None
    if not hostname or has_userinfo or parts.fragment:
        raise _invalid_config(field_name)
    if parts.query and endpoint_has_embedded_credentials(candidate):
        raise _invalid_config(field_name)
    scheme = parts.scheme.lower()
    if scheme == "http":
        if not allow_insecure:
            raise _invalid_config(field_name)
    elif scheme != "https":
        raise _invalid_config(field_name)
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), parts.query, ""))


def _join(base_url: str, suffix: str, query: Optional[Dict[str, str]] = None) -> str:
    parts = urlsplit(base_url)
    pairs = [p for p in parts.query.split("&") if p]
    if query:
        pairs.append(urlencode(query))
    return urlunsplit((parts.scheme, parts.netloc, parts.path + suffix, "&".join(pairs), ""))


def _require_secret(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or _has_control_chars(value):
        raise _invalid_config(field_name)
    return value


def _optional_secret(value: Any, field_name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    return _require_secret(value, field_name)


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _is_loopback(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    try:
        return (urlsplit(url.strip()).hostname or "") in LOOPBACK_HOSTS
    except ValueError:
        return False


class TextEndpointStrategy(ABC):
    """URL, authentication, and request-shape rules for one provider mode."""

    mode: str = "abstract"
    max_tokens_field: str = MAX_TOKENS
    supports_stream_usage: bool = False

    @abstractmethod
    def chat_url(self) -> str:
        raise NotImplementedError

    def probe_url(self) -> Optional[str]:
        """Idempotent GET used by health checks, or ``None`` when no documented probe exists."""
        return None

    @abstractmethod
    def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def extra_headers(self) -> Dict[str, str]:
        return {}

    @abstractmethod
    def secret_values(self) -> Tuple[str, ...]:
        """Every credential value this strategy may send, for scrubbing."""
        raise NotImplementedError

    def body_model(self, model: str) -> Optional[str]:
        """Model field for the request body; ``None`` omits it."""
        if not isinstance(model, str) or not model.strip():
            raise _invalid_config("model")
        return model.strip()

    def headers(self) -> Dict[str, str]:
        merged = dict(self.extra_headers())
        merged.update(self.auth_headers())
        return merged

    def __repr__(self) -> str:
        # Endpoints may identify a private resource; secrets must never appear.
        return f"{type(self).__name__}(mode={self.mode!r})"


class _BaseURLStrategy(TextEndpointStrategy):
    def __init__(self, base_url: str, *, allow_insecure: bool) -> None:
        self._base_url = _validate_base_url(base_url, "endpoint", allow_insecure=allow_insecure)

    def chat_url(self) -> str:
        return _join(self._base_url, "/chat/completions")

    def probe_url(self) -> Optional[str]:
        return _join(self._base_url, "/models")


class OpenAIStrategy(_BaseURLStrategy):
    """OpenAI Chat Completions: ``Authorization: Bearer`` and ``max_completion_tokens``."""

    mode = "openai"
    max_tokens_field = MAX_COMPLETION_TOKENS
    supports_stream_usage = True

    def __init__(self, base_url: str, api_key: str) -> None:
        super().__init__(base_url, allow_insecure=_is_loopback(base_url))
        self._api_key = _require_secret(api_key, "api_key")

    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def secret_values(self) -> Tuple[str, ...]:
        return (self._api_key,)


class AzureOpenAIStrategy(TextEndpointStrategy):
    """Azure OpenAI with explicit resource endpoint, deployment, and API version.

    ``api_version`` ``v1``/``preview`` targets the v1 surface, where the deployment name is
    sent as the body ``model``. A dated version (for example ``2024-10-21``) targets the
    deployment-scoped path, where the deployment is in the URL and no model is sent.
    """

    mode = "azure-openai"
    max_tokens_field = MAX_COMPLETION_TOKENS
    supports_stream_usage = True

    def __init__(
        self, resource_endpoint: str, deployment: str, api_version: str, api_key: str
    ) -> None:
        base = _validate_base_url(resource_endpoint, "endpoint", allow_insecure=False)
        if urlsplit(base).query:
            raise _invalid_config("endpoint")
        # The resource root is required; a pasted ".../openai/v1" would double the path.
        if urlsplit(base).path.rstrip("/").lower().endswith(("/openai", "/openai/v1")):
            raise _invalid_config("endpoint")
        if not isinstance(deployment, str) or not _DEPLOYMENT_RE.match(deployment):
            raise _invalid_config("deployment")
        if not isinstance(api_version, str) or not _API_VERSION_RE.match(api_version):
            raise _invalid_config("api_version")
        self._resource_endpoint = base
        self._deployment = deployment
        self._api_version = api_version
        self._api_key = _require_secret(api_key, "api_key")

    @property
    def uses_v1_surface(self) -> bool:
        return self._api_version in AZURE_V1_API_VERSIONS

    def chat_url(self) -> str:
        if self.uses_v1_surface:
            query = {"api-version": self._api_version} if self._api_version != "v1" else None
            return _join(self._resource_endpoint, "/openai/v1/chat/completions", query)
        return _join(
            self._resource_endpoint,
            f"/openai/deployments/{quote(self._deployment, safe='')}/chat/completions",
            {"api-version": self._api_version},
        )

    def probe_url(self) -> Optional[str]:
        if self.uses_v1_surface:
            return _join(self._resource_endpoint, "/openai/v1/models")
        return None

    def auth_headers(self) -> Dict[str, str]:
        return {"api-key": self._api_key}

    def secret_values(self) -> Tuple[str, ...]:
        return (self._api_key,)

    def body_model(self, model: str) -> Optional[str]:
        # The deployment, not a model name, selects the Azure model.
        return self._deployment if self.uses_v1_surface else None


class AssemblyAIGatewayStrategy(_BaseURLStrategy):
    """AssemblyAI LLM Gateway: raw key in ``Authorization`` (no Bearer) and ``max_tokens``."""

    mode = "assemblyai-llm-gateway"

    def __init__(self, base_url: str, api_key: str) -> None:
        super().__init__(base_url, allow_insecure=_is_loopback(base_url))
        self._api_key = _require_secret(api_key, "api_key")

    def probe_url(self) -> Optional[str]:
        return None

    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": self._api_key}

    def secret_values(self) -> Tuple[str, ...]:
        return (self._api_key,)


class LocalOpenAICompatibleStrategy(_BaseURLStrategy):
    """Local server (Ollama, LM Studio, vLLM, ...). The API key is optional."""

    mode = "local-openai-compatible"

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        super().__init__(base_url, allow_insecure=True)
        self._api_key = _optional_secret(api_key, "api_key")
        # A key over plain HTTP is only acceptable when it never leaves the machine.
        if (
            self._api_key
            and self._base_url.startswith("http:")
            and not _is_loopback(self._base_url)
        ):
            raise _invalid_config("endpoint")

    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def secret_values(self) -> Tuple[str, ...]:
        return (self._api_key,) if self._api_key else ()


class CustomOpenAICompatibleStrategy(_BaseURLStrategy):
    """User-configured OpenAI-compatible endpoint.

    ``headers`` are non-secret and may not use credential header names; credentials go in
    ``api_key`` or ``secret_headers`` so they are always known and scrubbed.
    """

    mode = "custom-openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        secret_headers: Optional[Dict[str, str]] = None,
        max_tokens_field: str = MAX_TOKENS,
        supports_stream_usage: bool = False,
    ) -> None:
        super().__init__(base_url, allow_insecure=True)
        self._api_key = _optional_secret(api_key, "api_key")
        if max_tokens_field not in (MAX_TOKENS, MAX_COMPLETION_TOKENS):
            raise _invalid_config("max_tokens_field")
        self.max_tokens_field = max_tokens_field
        self.supports_stream_usage = bool(supports_stream_usage)
        reserved = set(_RESERVED_HEADER_NAMES)
        if self._api_key:
            reserved.add("authorization")
        self._secret_headers = self._validate_headers(secret_headers, reserved, secret=True)
        reserved.update(name.lower() for name in self._secret_headers)
        self._headers = self._validate_headers(headers, reserved, secret=False)
        if (
            self.secret_values()
            and self._base_url.startswith("http:")
            and not _is_loopback(self._base_url)
        ):
            raise _invalid_config("endpoint")

    @staticmethod
    def _validate_headers(
        headers: Optional[Dict[str, str]], reserved: Set[str], *, secret: bool
    ) -> Dict[str, str]:
        field_name = "secret_headers" if secret else "headers"
        if headers is None:
            return {}
        if not isinstance(headers, dict):
            raise _invalid_config(field_name)
        cleaned: Dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not _HEADER_NAME_RE.match(name):
                raise _invalid_config(field_name)
            lowered = name.lower()
            if lowered in reserved:
                raise _invalid_config(field_name)
            if not isinstance(value, str) or _has_control_chars(value):
                raise _invalid_config(field_name)
            if not secret and (lowered in _SENSITIVE_HEADER_NAMES or is_secret_key_name(name)):
                raise _invalid_config(field_name)
            if secret and not value.strip():
                raise _invalid_config(field_name)
            cleaned[name] = value
        return cleaned

    def auth_headers(self) -> Dict[str, str]:
        merged = dict(self._secret_headers)
        if self._api_key:
            merged["Authorization"] = f"Bearer {self._api_key}"
        return merged

    def extra_headers(self) -> Dict[str, str]:
        return dict(self._headers)

    def secret_values(self) -> Tuple[str, ...]:
        values = list(self._secret_headers.values())
        if self._api_key:
            values.append(self._api_key)
        return tuple(values)


# ── Request shape and response parsing ───────────────────────────────────────


def build_chat_body(
    strategy: TextEndpointStrategy,
    model: str,
    messages: Sequence[TextMessage],
    generation: TextGenerationOptions,
    *,
    stream: bool,
) -> Dict[str, Any]:
    if not messages:
        raise _invalid_config("messages")
    wire_messages: List[Dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, TextMessage)
            or message.role not in _ROLES
            or not isinstance(message.content, str)
        ):
            raise _invalid_config("messages")
        wire_messages.append({"role": message.role, "content": message.content})
    body: Dict[str, Any] = {}
    body_model = strategy.body_model(model)
    if body_model is not None:
        body["model"] = body_model
    body["messages"] = wire_messages
    temperature = generation.temperature
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise _invalid_config("temperature")
        if not 0.0 <= float(temperature) <= 2.0:
            raise _invalid_config("temperature")
        body["temperature"] = float(temperature)
    max_tokens = generation.max_output_tokens
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise _invalid_config("max_output_tokens")
        body[strategy.max_tokens_field] = max_tokens
    if stream:
        body["stream"] = True
        if strategy.supports_stream_usage:
            body["stream_options"] = {"include_usage": True}
    return body


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_usage(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    """OpenAI/Azure report prompt/completion tokens; AssemblyAI reports input/output tokens."""
    if not isinstance(usage, dict):
        return None, None
    input_tokens = _int_or_none(usage.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _int_or_none(usage.get("input_tokens"))
    output_tokens = _int_or_none(usage.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _int_or_none(usage.get("output_tokens"))
    return input_tokens, output_tokens


def _response_id(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "request_id"):
        value = payload.get(key)
        if isinstance(value, str) and _PROVIDER_CODE_RE.match(value):
            return value
    return None


def _content_text(content: Any) -> Optional[str]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return None


class _MalformedResponseError(Exception):
    pass


def parse_completion(payload: Any) -> TextCompletion:
    if not isinstance(payload, dict):
        raise _MalformedResponseError("payload")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _MalformedResponseError("choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise _MalformedResponseError("message")
    text = _content_text(message.get("content"))
    if text is None:
        raise _MalformedResponseError("content")
    finish = choices[0].get("finish_reason")
    input_tokens, output_tokens = parse_usage(payload.get("usage"))
    return TextCompletion(
        text=text,
        response_id=_response_id(payload),
        finish_reason=finish if isinstance(finish, str) else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _error_provider_code(payload: Any) -> Optional[Any]:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("code") or error.get("type")
    return None


# ── Client ───────────────────────────────────────────────────────────────────


class OpenAICompatibleTextClient:
    """Shared async request/stream client. One instance per adapter; close with ``aclose``."""

    def __init__(
        self,
        strategy: TextEndpointStrategy,
        *,
        capability: ProviderCapability,
        options: Optional[TextTransportOptions] = None,
        http_transport: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if not isinstance(strategy, TextEndpointStrategy):
            raise _invalid_config("strategy")
        self._strategy = strategy
        self._capability = capability
        self._options = options or TextTransportOptions()
        self._http_transport = http_transport
        self._clock = clock
        self._sleep = sleep
        self._client: Any = None
        self._secrets = strategy.secret_values()

    def __repr__(self) -> str:
        return f"OpenAICompatibleTextClient(mode={self._strategy.mode!r})"

    @property
    def mode(self) -> str:
        return self._strategy.mode

    @property
    def redacted_endpoint(self) -> str:
        return redact_url(self._strategy.chat_url())

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Create the HTTP client. Imports httpx on first use; call off the event loop if cold."""
        if self._client is not None:
            return
        import httpx

        opts = self._options
        self._client = httpx.AsyncClient(
            transport=self._http_transport,
            timeout=httpx.Timeout(
                connect=opts.connect_timeout_s,
                read=opts.read_timeout_s,
                write=opts.write_timeout_s,
                pool=opts.pool_timeout_s,
            ),
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ── Public operations ────────────────────────────────────────────────────

    async def complete(
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
    ) -> TextCompletion:
        body = self._build_body(model, messages, generation, stream=False)
        began = self._clock()
        try:
            payload = await asyncio.wait_for(
                self._post_json(body), timeout=self._options.total_timeout_s
            )
        except asyncio.TimeoutError:
            raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
        try:
            completion = parse_completion(payload)
        except _MalformedResponseError as exc:
            logger.warning(
                "Text completion malformed: mode=%s part=%s", self._strategy.mode, exc.args[0]
            )
            raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE) from None
        return TextCompletion(
            text=completion.text,
            response_id=completion.response_id,
            finish_reason=completion.finish_reason,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=(self._clock() - began) * 1000.0,
        )

    async def stream(
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
    ) -> AsyncIterator[TextStreamChunk]:
        """Yield deltas then one ``is_done`` chunk. Closing the iterator closes the response."""
        body = self._build_body(model, messages, generation, stream=True)
        began = self._clock()
        deadline = began + self._options.total_timeout_s
        response = await self._send_with_retry(
            "POST", self._strategy.chat_url(), body, stream=True, idempotent=False
        )
        try:
            await self._raise_for_status(response)
            parts: List[str] = []
            state: Dict[str, Any] = {"id": None, "finish": None, "usage": (None, None)}
            done = False
            async for data in self._iter_sse_data(response, deadline):
                if data == "[DONE]":
                    done = True
                    break
                delta = self._parse_stream_event(data, state)
                if delta:
                    parts.append(delta)
                    yield TextStreamChunk(delta_text=delta)
            if not done and state["finish"] is None:
                raise self._raise_code(ProviderErrorCode.CONNECTION_CLOSED)
            input_tokens, output_tokens = state["usage"]
            yield TextStreamChunk(
                delta_text="",
                is_done=True,
                completion=TextCompletion(
                    text="".join(parts),
                    response_id=state["id"],
                    finish_reason=state["finish"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=(self._clock() - began) * 1000.0,
                ),
            )
        finally:
            await response.aclose()

    async def probe(self) -> Optional[float]:
        """Idempotent connectivity/auth check. Returns latency in ms, or ``None`` if unsupported."""
        url = self._strategy.probe_url()
        if url is None:
            return None
        began = self._clock()

        async def _run() -> None:
            response = await self._send_with_retry("GET", url, None, stream=True, idempotent=True)
            try:
                await self._raise_for_status(response)
            finally:
                await response.aclose()

        try:
            await asyncio.wait_for(_run(), timeout=self._options.total_timeout_s)
        except asyncio.TimeoutError:
            raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
        return (self._clock() - began) * 1000.0

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_body(
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions],
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        try:
            return build_chat_body(
                self._strategy,
                model,
                messages,
                generation or TextGenerationOptions(),
                stream=stream,
            )
        except TextConfigError as exc:
            raise config_error_to_provider_error(
                exc, self._capability, self._strategy.mode
            ) from None

    def _raise_code(
        self,
        code: ProviderErrorCode,
        *,
        http_status: Optional[int] = None,
        provider_code: Optional[str] = None,
    ) -> ProviderRuntimeError:
        return ProviderRuntimeError(
            make_text_error(
                code,
                self._capability,
                http_status=http_status,
                provider_code=provider_code,
                mode=self._strategy.mode,
            )
        )

    def _require_client(self) -> Any:
        if self._client is None:
            self.open()
        return self._client

    async def _post_json(self, body: Dict[str, Any]) -> Any:
        response = await self._send_with_retry(
            "POST", self._strategy.chat_url(), body, stream=True, idempotent=False
        )
        try:
            await self._raise_for_status(response)
            raw = await self._read_bounded(response, self._options.max_response_bytes)
        finally:
            await response.aclose()
        try:
            return json.loads(raw)
        except ValueError:
            raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE) from None

    async def _send_with_retry(
        self,
        method: str,
        url: str,
        body: Optional[Dict[str, Any]],
        *,
        stream: bool,
        idempotent: bool,
    ) -> Any:
        import httpx

        client = self._require_client()
        headers = self._strategy.headers()
        headers["Accept"] = (
            "text/event-stream" if body and body.get("stream") else "application/json"
        )
        attempts = 1 + max(0, self._options.max_retries)
        for attempt in range(1, attempts + 1):
            request = client.build_request(method, url, json=body, headers=headers)
            try:
                response = await client.send(request, stream=stream)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                # The request never reached the server, so even a POST is safe to resend.
                error_code = (
                    ProviderErrorCode.NETWORK_UNAVAILABLE
                    if isinstance(exc, httpx.ConnectError)
                    else ProviderErrorCode.TIMEOUT
                )
                self._log_transport_failure(exc, attempt)
                if attempt < attempts:
                    await self._sleep(self._options.retry_backoff_s * attempt)
                    continue
                raise self._raise_code(error_code) from None
            except httpx.TimeoutException as exc:
                self._log_transport_failure(exc, attempt)
                if idempotent and attempt < attempts:
                    await self._sleep(self._options.retry_backoff_s * attempt)
                    continue
                raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
            except httpx.HTTPError as exc:
                self._log_transport_failure(exc, attempt)
                if idempotent and attempt < attempts:
                    await self._sleep(self._options.retry_backoff_s * attempt)
                    continue
                raise self._raise_code(ProviderErrorCode.NETWORK_UNAVAILABLE) from None
            if (
                idempotent
                and attempt < attempts
                and (response.status_code in (408, 429) or response.status_code >= 500)
            ):
                delay = self._retry_delay(response, attempt)
                await response.aclose()
                logger.info(
                    "Text probe retrying: mode=%s status=%d attempt=%d",
                    self._strategy.mode,
                    response.status_code,
                    attempt,
                )
                await self._sleep(delay)
                continue
            return response
        raise self._raise_code(ProviderErrorCode.NETWORK_UNAVAILABLE)  # pragma: no cover

    def _retry_delay(self, response: Any, attempt: int) -> float:
        delay = self._options.retry_backoff_s * attempt
        header = response.headers.get("retry-after")
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = 0.0
            delay = max(delay, retry_after)
        return min(delay, self._options.max_retry_after_s)

    def _log_transport_failure(self, exc: BaseException, attempt: int) -> None:
        # httpx exception text includes the URL; only the type is safe to log.
        logger.warning(
            "Text request transport failure: mode=%s endpoint=%s error=%s attempt=%d",
            self._strategy.mode,
            self.redacted_endpoint,
            type(exc).__name__,
            attempt,
        )

    async def _raise_for_status(self, response: Any) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        provider_code: Optional[str] = None
        try:
            raw = await self._read_bounded(
                response, self._options.max_error_body_bytes, truncate=True
            )
            provider_code = sanitize_provider_code(
                _error_provider_code(json.loads(raw)), self._secrets
            )
        except ProviderRuntimeError:
            provider_code = None
        except ValueError:
            provider_code = None
        code = classify_http_status(status)
        if provider_code:
            mapped = _PROVIDER_CODE_MAP.get(provider_code.lower())
            # A specific provider code refines a generic status, never a server failure.
            if mapped is not None and status < 500:
                code = mapped
        logger.warning(
            "Text request failed: mode=%s status=%d code=%s provider_code=%s",
            self._strategy.mode,
            status,
            code.value,
            provider_code,
        )
        raise self._raise_code(code, http_status=status, provider_code=provider_code)

    async def _read_bounded(self, response: Any, limit: int, *, truncate: bool = False) -> bytes:
        import httpx

        buffer = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                if len(buffer) > limit:
                    if truncate:
                        return bytes(buffer[:limit])
                    raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE)
        except httpx.TimeoutException:
            raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
        except httpx.HTTPError:
            raise self._raise_code(ProviderErrorCode.CONNECTION_CLOSED) from None
        return bytes(buffer)

    async def _iter_sse_data(self, response: Any, deadline: float) -> AsyncIterator[str]:
        """Yield the joined ``data:`` payload of each SSE event, bounded in size and time."""
        import httpx

        opts = self._options
        total = 0
        pending = b""
        data_lines: List[str] = []
        iterator = response.aiter_bytes().__aiter__()
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise self._raise_code(ProviderErrorCode.TIMEOUT)
            try:
                chunk = await asyncio.wait_for(
                    iterator.__anext__(), timeout=min(remaining, opts.stream_idle_timeout_s)
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
            except httpx.TimeoutException:
                raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
            except httpx.HTTPError:
                raise self._raise_code(ProviderErrorCode.CONNECTION_CLOSED) from None
            total += len(chunk)
            if total > opts.max_stream_bytes:
                raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE)
            pending += chunk
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    break
                line, pending = pending[:index], pending[index + 1 :]
                text = line.rstrip(b"\r").decode("utf-8", errors="replace")
                if not text:
                    if data_lines:
                        yield "\n".join(data_lines)
                        data_lines = []
                    continue
                if text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    data_lines.append(text[5:].lstrip(" "))
            if len(pending) > opts.max_sse_line_bytes:
                raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE)
        tail = pending.rstrip(b"\r").decode("utf-8", errors="replace")
        if tail.startswith("data:"):
            data_lines.append(tail[5:].lstrip(" "))
        if data_lines:
            yield "\n".join(data_lines)

    def _parse_stream_event(self, data: str, state: Dict[str, Any]) -> str:
        try:
            payload = json.loads(data)
        except ValueError:
            logger.debug("Text stream event not JSON: mode=%s", self._strategy.mode)
            return ""
        if not isinstance(payload, dict):
            return ""
        if "error" in payload:
            provider_code = sanitize_provider_code(_error_provider_code(payload), self._secrets)
            mapped = _PROVIDER_CODE_MAP.get((provider_code or "").lower())
            raise self._raise_code(
                mapped or ProviderErrorCode.INTERNAL_ERROR, provider_code=provider_code
            )
        if state["id"] is None:
            state["id"] = _response_id(payload)
        usage = parse_usage(payload.get("usage"))
        if usage != (None, None):
            state["usage"] = usage
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ""
        finish = choices[0].get("finish_reason")
        if isinstance(finish, str):
            state["finish"] = finish
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return ""
        return _content_text(delta.get("content")) or ""


__all__ = [
    "AZURE_V1_API_VERSIONS",
    "AssemblyAIGatewayStrategy",
    "AzureOpenAIStrategy",
    "CustomOpenAICompatibleStrategy",
    "LocalOpenAICompatibleStrategy",
    "OpenAICompatibleTextClient",
    "OpenAIStrategy",
    "TextCompletion",
    "TextConfigError",
    "TextEndpointStrategy",
    "TextGenerationOptions",
    "TextMessage",
    "TextStreamChunk",
    "TextTransportOptions",
    "build_chat_body",
    "classify_http_status",
    "config_error_to_provider_error",
    "make_text_error",
    "parse_completion",
    "parse_usage",
    "redact_url",
    "sanitize_provider_code",
    "scrub_secrets",
]
