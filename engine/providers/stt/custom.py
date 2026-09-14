"""Custom STT: a self-hosted realtime server behind one of two bounded protocol profiles.

A Custom STT server must speak an existing protocol exactly (D-011). ``protocol_profile``
picks which one, and this module only delegates:

- ``openai-realtime`` -> ``OpenAIRealtimeSTTProvider`` with ``openai_protocol`` wire formats.
- ``gemini-live`` -> ``GeminiLiveSTTProvider`` with ``gemini_protocol`` wire formats.

There is no generic event mapper and no parsing here; any other profile is rejected. The
delegate owns the socket lifecycle, transcript revisions, and error mapping.

Headers. Public headers are plain config values and may not use credential-like names.
Secret headers take their names from config and their values from credential references
(the ``secret_header_N`` vault slots), so a header value never lands in config. Both kinds
are merged into the delegate's WebSocket handshake through its injectable connector, next to
the profile's own auth header, which neither kind may override.

Construction. ``from_resolved`` builds from a registry-resolved service plus the secrets for
its credential references. The EF-007 family builders may also pass the delegate's own
settings object directly; that path cannot carry custom headers (see ``__init__``).

Adapter modules are imported only when a profile is built, so importing this module never
loads ``websockets`` or the other profile's adapter.
"""

import re
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
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
)

from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.manifests import (
    CUSTOM_STT_PROFILE_FIELD,
    CUSTOM_STT_PROFILES,
    CUSTOM_STT_PUBLIC_HEADER_SLOTS,
    CUSTOM_STT_SECRET_HEADER_SLOTS,
)
from engine.providers.stt.streaming_ws import validate_endpoint
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderRuntimeError,
    STTTranscriptResult,
)

if TYPE_CHECKING:
    from engine.providers.registry import ResolvedService

logger = get_logger("providers.stt.custom")

PROFILE_OPENAI_REALTIME = "openai-realtime"
PROFILE_GEMINI_LIVE = "gemini-live"
DEFAULT_PROVIDER_ID = "custom"
MAX_HEADER_VALUE_CHARS = 4096

# RFC 9110 field-name token.
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}$")
# The profiles' own auth headers and handshake headers the WebSocket client must own.
_RESERVED_HEADERS = frozenset(
    {
        "authorization",
        "x-goog-api-key",
        "host",
        "connection",
        "upgrade",
        "origin",
        "content-length",
        "transfer-encoding",
    }
)
_RESERVED_HEADER_PREFIX = "sec-websocket-"
# A public header whose name suggests a credential must be configured as a secret header.
_CREDENTIAL_NAME_HINTS = (
    "auth",
    "token",
    "key",
    "secret",
    "cookie",
    "password",
    "session",
    "signature",
    "credential",
)
_AUTO_LANGUAGE = "auto"

ConnectBase = Callable[..., Awaitable[Any]]


def _config_error(code: ProviderErrorCode, field_name: str) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        ProviderError(
            code=code,
            capability=ProviderCapability.STT,
            message="Custom STT configuration is invalid.",
            retryable=False,
            details={"field": field_name},
        )
    )


def _invalid(field_name: str) -> ProviderRuntimeError:
    return _config_error(ProviderErrorCode.INVALID_REQUEST, field_name)


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CustomSTTSettings:
    """Resolved Custom STT settings. Secret values never appear in repr."""

    protocol_profile: str
    endpoint: str
    model: str
    api_key: str = field(default="", repr=False)
    language_hints: Tuple[str, ...] = ()
    headers: Tuple[Tuple[str, str], ...] = ()
    secret_headers: Tuple[Tuple[str, str], ...] = field(default=(), repr=False)
    provider_id: str = DEFAULT_PROVIDER_ID


def _check_header(name: Any, value: Any, name_field: str, value_field: str, secret: bool) -> None:
    if not isinstance(name, str) or not _HEADER_NAME_RE.match(name):
        raise _invalid(name_field)
    lower = name.lower()
    if lower in _RESERVED_HEADERS or lower.startswith(_RESERVED_HEADER_PREFIX):
        raise _invalid(name_field)
    if not secret and any(hint in lower for hint in _CREDENTIAL_NAME_HINTS):
        raise _invalid(name_field)
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_HEADER_VALUE_CHARS
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise _invalid(value_field)


def validate_settings(settings: CustomSTTSettings) -> None:
    """Reject unusable settings before any adapter or socket exists. Never echoes values."""
    if settings.protocol_profile not in CUSTOM_STT_PROFILES:
        raise _invalid(CUSTOM_STT_PROFILE_FIELD)
    try:
        validate_endpoint(settings.endpoint)
    except ValueError:
        raise _invalid("endpoint") from None
    if not isinstance(settings.model, str) or not settings.model.strip():
        raise _invalid("model")
    if not isinstance(settings.api_key, str) or not settings.api_key.strip():
        raise _config_error(ProviderErrorCode.AUTH_FAILED, "api_key")
    for hint in settings.language_hints:
        if not isinstance(hint, str) or not hint.strip():
            raise _invalid("language")
    seen: List[str] = []
    for kind, pairs, secret in (
        ("header", settings.headers, False),
        ("secret_header", settings.secret_headers, True),
    ):
        for index, (name, value) in enumerate(pairs, start=1):
            prefix = f"{kind}_{index}"
            value_field = prefix if secret else f"{prefix}_value"
            _check_header(name, value, f"{prefix}_name", value_field, secret)
            if name.lower() in seen:
                raise _invalid(f"{prefix}_name")
            seen.append(name.lower())


def _language_hints(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, str) or value.strip().lower() in ("", _AUTO_LANGUAGE):
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


# ── Delegation ───────────────────────────────────────────────────────────────


def _extra_headers(settings: CustomSTTSettings) -> Dict[str, str]:
    merged = dict(settings.headers)
    merged.update(settings.secret_headers)
    return merged


def _build_openai_realtime(
    settings: CustomSTTSettings, stream_ids: Tuple[str, ...], connect_base: Optional[ConnectBase]
) -> StreamingSTTProvider:
    from engine.providers.stt.openai_protocol import OpenAIRealtimeSettings
    from engine.providers.stt.openai_realtime import (
        OpenAIRealtimeSTTProvider,
        RealtimeTransportOptions,
        make_websocket_factory,
    )

    transport = RealtimeTransportOptions()
    base = connect_base or make_websocket_factory(transport.max_message_bytes)
    extra = _extra_headers(settings)

    async def connect(url: str, headers: Dict[str, str], open_timeout: float) -> Any:
        # Reserved names keep extra headers from replacing the profile's auth header.
        return await base(url, {**extra, **headers}, open_timeout)

    delegate_settings = OpenAIRealtimeSettings(
        endpoint=settings.endpoint,
        model=settings.model,
        api_key=settings.api_key,
        languages=settings.language_hints,
    )
    return OpenAIRealtimeSTTProvider(
        delegate_settings,
        provider_id=settings.provider_id,
        stream_ids=stream_ids,
        transport=transport,
        connect_factory=connect,
    )


def _build_gemini_live(
    settings: CustomSTTSettings, stream_ids: Tuple[str, ...], connect_base: Optional[ConnectBase]
) -> StreamingSTTProvider:
    from engine.providers.stt.gemini_live import (
        GeminiLiveSTTConfig,
        GeminiLiveSTTProvider,
        websockets_connector,
    )

    base = connect_base or websockets_connector
    extra = _extra_headers(settings)

    async def connect(url: str, headers: Dict[str, str]) -> Any:
        return await base(url, {**extra, **headers})

    # Gemini Live opens one connection per stream id on first audio; no stream list needed.
    config = GeminiLiveSTTConfig(
        api_key=settings.api_key,
        endpoint=settings.endpoint,
        model=settings.model,
        language_hints=settings.language_hints,
        provider_id=settings.provider_id,
    )
    return GeminiLiveSTTProvider(config, connector=connect)


_DELEGATE_BUILDERS: Dict[
    str,
    Callable[[CustomSTTSettings, Tuple[str, ...], Optional[ConnectBase]], StreamingSTTProvider],
] = {
    PROFILE_OPENAI_REALTIME: _build_openai_realtime,
    PROFILE_GEMINI_LIVE: _build_gemini_live,
}


def _coerce_settings(settings: Any, provider_id: Optional[str]) -> CustomSTTSettings:
    """Accept Custom settings, or a delegate settings object from an EF-007 family builder."""
    if isinstance(settings, CustomSTTSettings):
        return replace(settings, provider_id=provider_id) if provider_id else settings

    from engine.providers.stt.openai_protocol import OpenAIRealtimeSettings

    if isinstance(settings, OpenAIRealtimeSettings):
        return CustomSTTSettings(
            protocol_profile=PROFILE_OPENAI_REALTIME,
            endpoint=settings.endpoint,
            model=settings.model,
            api_key=settings.api_key,
            language_hints=tuple(settings.languages),
            provider_id=provider_id or DEFAULT_PROVIDER_ID,
        )

    from engine.providers.stt.gemini_live import GeminiLiveSTTConfig

    if isinstance(settings, GeminiLiveSTTConfig):
        return CustomSTTSettings(
            protocol_profile=PROFILE_GEMINI_LIVE,
            endpoint=settings.endpoint,
            model=settings.model,
            api_key=settings.api_key,
            language_hints=tuple(settings.language_hints),
            provider_id=provider_id or settings.provider_id,
        )
    raise _invalid("settings")


# ── Provider ─────────────────────────────────────────────────────────────────


class CustomSTTProvider(StreamingSTTProvider):
    """``StreamingSTTProvider`` that forwards every call to its profile's adapter."""

    def __init__(
        self,
        settings: Any,
        *,
        provider_id: Optional[str] = None,
        stream_ids: Sequence[str] = ("default",),
        connect_factory: Optional[ConnectBase] = None,
    ) -> None:
        """Build the delegate. ``connect_factory`` replaces the delegate's socket opener.

        ``settings`` is normally ``CustomSTTSettings``. An ``OpenAIRealtimeSettings`` or
        ``GeminiLiveSTTConfig`` (as EF-007's family builders pass) is also accepted; it has no
        header fields, so a server that needs custom headers cannot be reached that way.
        """
        custom = _coerce_settings(settings, provider_id)
        validate_settings(custom)
        self._settings = custom
        self._delegate = _DELEGATE_BUILDERS[custom.protocol_profile](
            custom, tuple(stream_ids), connect_factory
        )
        logger.info(
            "Custom STT configured: provider=%s profile=%s headers=%d secret_headers=%d",
            custom.provider_id,
            custom.protocol_profile,
            len(custom.headers),
            len(custom.secret_headers),
        )

    @classmethod
    def from_resolved(
        cls,
        resolved: "ResolvedService",
        secrets: Mapping[str, str],
        *,
        stream_ids: Sequence[str] = ("default",),
        connect_factory: Optional[ConnectBase] = None,
    ) -> "CustomSTTProvider":
        """Build from a registry-resolved service and the secrets for its credential refs.

        ``secrets`` is read, never stored or mutated; the caller owns clearing it.
        """
        if getattr(resolved, "capability", None) is not ProviderCapability.STT:
            raise _invalid("capability")
        options = resolved.options
        profile = options.get(CUSTOM_STT_PROFILE_FIELD)
        if profile not in CUSTOM_STT_PROFILES:
            raise _invalid(CUSTOM_STT_PROFILE_FIELD)
        if resolved.protocol_family is not CUSTOM_STT_PROFILES[profile]:
            raise _invalid(CUSTOM_STT_PROFILE_FIELD)

        headers: List[Tuple[str, str]] = []
        for index in range(1, CUSTOM_STT_PUBLIC_HEADER_SLOTS + 1):
            name = _optional_text(options.get(f"header_{index}_name"))
            value = options.get(f"header_{index}_value")
            has_value = isinstance(value, str) and bool(value.strip())
            if not name and not has_value:
                continue
            if not name:
                raise _invalid(f"header_{index}_name")
            if not has_value:
                raise _invalid(f"header_{index}_value")
            headers.append((name, value))

        secret_headers: List[Tuple[str, str]] = []
        for index in range(1, CUSTOM_STT_SECRET_HEADER_SLOTS + 1):
            slot = f"secret_header_{index}"
            name = _optional_text(options.get(f"{slot}_name"))
            referenced = slot in resolved.credential_refs
            if not name and not referenced:
                continue
            if not name:
                raise _invalid(f"{slot}_name")
            value = secrets.get(slot) if referenced else None
            # A secret header value is only ever taken from its credential reference.
            if not isinstance(value, str) or not value.strip():
                raise _config_error(ProviderErrorCode.AUTH_FAILED, slot)
            secret_headers.append((name, value))

        api_key = secrets.get("api_key") if "api_key" in resolved.credential_refs else None
        settings = CustomSTTSettings(
            protocol_profile=profile,
            endpoint=resolved.endpoint,
            model=resolved.model,
            api_key=api_key if isinstance(api_key, str) else "",
            language_hints=_language_hints(options.get("language")),
            headers=tuple(headers),
            secret_headers=tuple(secret_headers),
            provider_id=resolved.provider_id,
        )
        return cls(settings, stream_ids=stream_ids, connect_factory=connect_factory)

    @property
    def provider_id(self) -> str:
        return self._settings.provider_id

    @property
    def protocol_profile(self) -> str:
        return self._settings.protocol_profile

    @property
    def delegate(self) -> StreamingSTTProvider:
        return self._delegate

    async def start(self) -> None:
        await self._delegate.start()

    async def stop(self) -> None:
        await self._delegate.stop()

    async def health_check(self) -> ProviderHealthCheck:
        return await self._delegate.health_check()

    async def send_audio(self, frame: AudioFrame) -> None:
        await self._delegate.send_audio(frame)

    def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        return self._delegate.receive_events()

    async def finalize_stream(self, stream_id: str) -> None:
        await self._delegate.finalize_stream(stream_id)

    async def reset_stream(self, stream_id: str) -> None:
        await self._delegate.reset_stream(stream_id)
