"""Capability-specific provider validation (Test Connection) and its process entrypoint.

Transport: Tauri runs this module as a short-lived child process,
``python -m engine.providers.validation``. The child reads exactly one JSON request from
stdin, validates once, writes exactly one JSON result line to stdout, and exits. Secret
values therefore travel Rust -> pipe -> this process's memory only; they never touch argv,
the environment, the engine socket, or disk.

Request (stdin, UTF-8, at most ``MAX_REQUEST_BYTES``)::

    {"id": "<[A-Za-z0-9._:-]{1,128}>",
     "capability": "stt" | "translation" | "assistant",
     "service": {<services.<capability> dict: providerId, endpoint, modelOverride,
                 options, credentialRefs, ...>},
     "secrets": {"<slot>": "<value>"}}

Result (stdout, one line)::

    {
        "id",
        "capability",
        "providerId",
        "status",
        "latencyMs",
        "testedAt",
        "fingerprint",
        "credentialVersion",
        "errorCode",
    }

``status`` is ``valid``, ``invalid``, or ``unvalidated``. ``errorCode`` is a fixed
identifier (a ``ProviderErrorCode``, ``RegistryErrorCode`` or ``ConfigError`` code), never
provider message text. A stored result counts only while its fingerprint and credential
version still match the service config; see ``current_validation_status``.

Strategies are chosen from the resolved manifest:

- STT (cloud protocol families): the adapter's ``health_check`` opens one session, waits
  for the protocol handshake, and closes it.
- Translation: one ``translate`` call on a short fixed text with a small output cap.
- Assistant: one ``assist`` call on a short fixed message with a small output cap.
- Local runtimes: not validated over the network; reported ``unvalidated``.

Whatever happens, the adapter is stopped in ``finally`` under a bounded timeout, and the
secrets dict is cleared as soon as the adapter is built. Python strings cannot be wiped,
so clearing drops the last references; the process exiting is the real cleanup.
"""

import asyncio
import dataclasses
import io
import json
import logging
import re
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, TextIO, Tuple

from engine.core.config import (
    SERVICE_NAMES,
    VALIDATION_INVALID,
    VALIDATION_UNVALIDATED,
    VALIDATION_VALID,
    ConfigError,
    ServiceConfig,
    is_valid_id_segment,
)
from engine.core.logging import get_logger
from engine.providers.contracts import (
    AssistantProvider,
    BaseProvider,
    StreamingSTTProvider,
    TranslationProvider,
)
from engine.providers.manifests import ProtocolFamily
from engine.providers.registry import (
    ProviderRegistry,
    ProviderRegistryError,
    RegistryErrorCode,
    ResolvedService,
    build_default_registry,
)
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    ConversationMessage,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    TranslationRequest,
    TranslationResult,
)

logger = get_logger("providers.validation")

DEFAULT_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 3.0
MAX_REQUEST_BYTES = 64 * 1024

# Probe payloads are fixed and tiny so a Test Connection costs as little as possible.
PROBE_TEXT = "Hello."
PROBE_MAX_OUTPUT_TOKENS = 16
PROBE_STREAM_ID = "validation"

ERROR_INVALID_REQUEST = ProviderErrorCode.INVALID_REQUEST.value
ERROR_INVALID_RESPONSE = ProviderErrorCode.INVALID_RESPONSE.value
ERROR_INTERNAL = ProviderErrorCode.INTERNAL_ERROR.value
ERROR_TIMEOUT = ProviderErrorCode.TIMEOUT.value
ERROR_MISSING_CREDENTIAL = RegistryErrorCode.MISSING_CREDENTIAL.value
ERROR_FACTORY_UNAVAILABLE = RegistryErrorCode.FACTORY_UNAVAILABLE.value
ERROR_NOT_SUPPORTED = "validation_not_supported"

_REQUEST_FIELDS = frozenset({"id", "capability", "service", "secrets"})
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PROVIDER_ERROR_CODES = frozenset(code.value for code in ProviderErrorCode)


class ValidationFailedError(Exception):
    """A validation step failed with a fixed, safe error code."""

    def __init__(self, code: str) -> None:
        safe = code if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code) else ERROR_INTERNAL
        super().__init__(safe)
        self.code = safe


@dataclass(frozen=True)
class ValidationResult:
    """Everything a Test Connection reports. Holds no secret and no provider text."""

    request_id: Optional[str]
    capability: Optional[str]
    provider_id: Optional[str]
    status: str
    latency_ms: Optional[float]
    tested_at: float
    fingerprint: Optional[str]
    credential_version: Optional[int]
    error_code: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.request_id,
            "capability": self.capability,
            "providerId": self.provider_id,
            "status": self.status,
            "latencyMs": self.latency_ms,
            "testedAt": self.tested_at,
            "fingerprint": self.fingerprint,
            "credentialVersion": self.credential_version,
            "errorCode": self.error_code,
        }


def credential_version(service: ServiceConfig) -> Optional[int]:
    """Highest version across the service's credential refs; None when it has none.

    Replacing any secret bumps its ref version, so the maximum moves whenever one does.
    The fingerprint covers every ref version individually as well.
    """
    if not service.credential_refs:
        return None
    return max(ref.version for ref in service.credential_refs.values())


def current_validation_status(result: ValidationResult, service: ServiceConfig) -> str:
    """``valid``/``invalid`` only while the result still describes ``service``.

    A changed provider, fingerprint, or credential version makes the stored result
    meaningless, so it reads as ``unvalidated`` rather than a stale pass or fail.
    """
    if result.status not in (VALIDATION_VALID, VALIDATION_INVALID):
        return VALIDATION_UNVALIDATED
    if result.capability != service.service or result.provider_id != service.provider_id:
        return VALIDATION_UNVALIDATED
    if result.fingerprint is None or result.fingerprint != service.fingerprint():
        return VALIDATION_UNVALIDATED
    if result.credential_version != credential_version(service):
        return VALIDATION_UNVALIDATED
    return result.status


# ── Strategies ───────────────────────────────────────────────────────────────


class ValidationStrategy(ABC):
    """One capability's minimum proof that a configured provider works."""

    def prepare(self, resolved: ResolvedService) -> ResolvedService:
        """Adjust the resolved service before the adapter is built (e.g. cap output)."""
        return resolved

    @abstractmethod
    async def probe(self, adapter: BaseProvider, resolved: ResolvedService) -> None:
        """Return on success; raise ``ValidationFailedError`` or ``ProviderRuntimeError``."""
        raise NotImplementedError


class HandshakeStrategy(ValidationStrategy):
    """STT: protocol handshake plus clean close, via the adapter's ``health_check``."""

    async def probe(self, adapter: BaseProvider, resolved: ResolvedService) -> None:
        if not isinstance(adapter, StreamingSTTProvider):
            raise ValidationFailedError(ERROR_FACTORY_UNAVAILABLE)
        check = await adapter.health_check()
        if not isinstance(check, ProviderHealthCheck):
            raise ValidationFailedError(ERROR_INVALID_RESPONSE)
        if check.status is ProviderHealthStatus.HEALTHY:
            return
        code = (check.details or {}).get("code")
        raise ValidationFailedError(code if code in _PROVIDER_ERROR_CODES else ERROR_INTERNAL)


def _cap_output(resolved: ResolvedService) -> ResolvedService:
    if "max_output_tokens" not in resolved.options:
        return resolved
    options = dict(resolved.options)
    current = options["max_output_tokens"]
    if isinstance(current, int) and not isinstance(current, bool):
        options["max_output_tokens"] = min(current, PROBE_MAX_OUTPUT_TOKENS)
    else:
        options["max_output_tokens"] = PROBE_MAX_OUTPUT_TOKENS
    return dataclasses.replace(resolved, options=options)


class TranslationProbeStrategy(ValidationStrategy):
    """Translation: one short ``translate`` call with a small output cap."""

    def prepare(self, resolved: ResolvedService) -> ResolvedService:
        return _cap_output(resolved)

    async def probe(self, adapter: BaseProvider, resolved: ResolvedService) -> None:
        if not isinstance(adapter, TranslationProvider):
            raise ValidationFailedError(ERROR_FACTORY_UNAVAILABLE)
        await adapter.start()
        source = resolved.options.get("source_lang")
        target = resolved.options.get("target_lang")
        request = TranslationRequest(
            text=PROBE_TEXT,
            source_lang=source if isinstance(source, str) and source else "auto",
            target_lang=target if isinstance(target, str) and target else "en",
            stream_id=PROBE_STREAM_ID,
        )
        result = await adapter.translate(request)
        text = getattr(result, "translated_text", None)
        if not isinstance(result, TranslationResult) or not isinstance(text, str):
            raise ValidationFailedError(ERROR_INVALID_RESPONSE)
        if not text.strip():
            raise ValidationFailedError(ERROR_INVALID_RESPONSE)


class AssistantProbeStrategy(ValidationStrategy):
    """Assistant: one short ``assist`` call with a small output cap."""

    def prepare(self, resolved: ResolvedService) -> ResolvedService:
        return _cap_output(resolved)

    async def probe(self, adapter: BaseProvider, resolved: ResolvedService) -> None:
        if not isinstance(adapter, AssistantProvider):
            raise ValidationFailedError(ERROR_FACTORY_UNAVAILABLE)
        await adapter.start()
        message = ConversationMessage(
            message_id="validation-probe",
            speaker="validator",
            text=PROBE_TEXT,
            timestamp_ms=0.0,
            stream_id=PROBE_STREAM_ID,
        )
        result = await adapter.assist(AssistRequest(target_message=message))
        if not isinstance(result, AssistResult):
            raise ValidationFailedError(ERROR_INVALID_RESPONSE)


_STRATEGIES: Dict[ProviderCapability, ValidationStrategy] = {
    ProviderCapability.STT: HandshakeStrategy(),
    ProviderCapability.TRANSLATION: TranslationProbeStrategy(),
    ProviderCapability.ASSISTANT: AssistantProbeStrategy(),
}


def select_strategy(resolved: ResolvedService) -> Optional[ValidationStrategy]:
    """Strategy for the manifest's capability, or None where network validation is moot."""
    if resolved.protocol_family is ProtocolFamily.LOCAL:
        return None
    return _STRATEGIES.get(resolved.capability)


# ── Adapter construction ─────────────────────────────────────────────────────

# (factory, resolved service, secrets by slot) -> adapter. Builders must not keep
# ``secrets``; the caller clears it right after the call.
AdapterBuilder = Callable[[Callable[..., Any], ResolvedService, Mapping[str, str]], BaseProvider]


def _language_tuple(resolved: ResolvedService) -> Tuple[str, ...]:
    language = resolved.options.get("language")
    if isinstance(language, str) and language.strip() and language != "auto":
        return (language,)
    return ()


def build_openai_realtime(
    factory: Callable[..., Any], resolved: ResolvedService, secrets: Mapping[str, str]
) -> BaseProvider:
    from engine.providers.stt.openai_protocol import OpenAIRealtimeSettings

    settings = OpenAIRealtimeSettings(
        endpoint=resolved.endpoint,
        model=resolved.model,
        api_key=secrets.get("api_key", ""),
        languages=_language_tuple(resolved),
    )
    return factory(settings, provider_id=resolved.provider_id)


def build_gemini_live(
    factory: Callable[..., Any], resolved: ResolvedService, secrets: Mapping[str, str]
) -> BaseProvider:
    from engine.providers.stt.gemini_live import GeminiLiveSTTConfig

    config = GeminiLiveSTTConfig(
        api_key=secrets.get("api_key", ""),
        endpoint=resolved.endpoint,
        model=resolved.model,
        language_hints=_language_tuple(resolved),
        provider_id=resolved.provider_id,
    )
    return factory(config)


def build_from_resolved(
    factory: Callable[..., Any], resolved: ResolvedService, secrets: Mapping[str, str]
) -> BaseProvider:
    """Generic hook: adapters without a family builder expose ``from_resolved``."""
    hook = getattr(factory, "from_resolved", None)
    if not callable(hook):
        raise ValidationFailedError(ERROR_FACTORY_UNAVAILABLE)
    return hook(resolved, secrets)


DEFAULT_BUILDERS: Dict[ProtocolFamily, AdapterBuilder] = {
    ProtocolFamily.OPENAI_REALTIME: build_openai_realtime,
    ProtocolFamily.GEMINI_LIVE: build_gemini_live,
}


# ── Validation ───────────────────────────────────────────────────────────────


def _check_secrets(resolved: ResolvedService, secrets: Any) -> None:
    if not isinstance(secrets, dict):
        raise ValidationFailedError(ERROR_INVALID_REQUEST)
    for slot, value in secrets.items():
        if slot not in resolved.credential_refs or not isinstance(value, str):
            raise ValidationFailedError(ERROR_INVALID_REQUEST)
    for slot in resolved.credential_refs:
        value = secrets.get(slot)
        if not isinstance(value, str) or not value.strip():
            raise ValidationFailedError(ERROR_MISSING_CREDENTIAL)


async def _stop_quietly(adapter: BaseProvider, provider_id: str) -> None:
    try:
        await asyncio.wait_for(adapter.stop(), STOP_TIMEOUT_S)
    except (Exception, asyncio.CancelledError) as exc:
        # Type only: exception text from network clients can carry URLs or headers.
        logger.debug(
            "Validation adapter stop failed: provider=%s error=%s", provider_id, type(exc).__name__
        )


class _Attempt:
    def __init__(self) -> None:
        self.adapter: Optional[BaseProvider] = None
        self.probe_started: Optional[float] = None


async def validate_service(
    service: ServiceConfig,
    secrets: Dict[str, str],
    *,
    registry: ProviderRegistry,
    builders: Optional[Mapping[ProtocolFamily, AdapterBuilder]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    request_id: Optional[str] = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
) -> ValidationResult:
    """Validate one service config once. Never raises except on cancellation.

    ``secrets`` is cleared before this returns, on every path.
    """
    family_builders = DEFAULT_BUILDERS if builders is None else builders
    fingerprint = service.fingerprint()
    version = credential_version(service)
    attempt = _Attempt()

    def finish(status: str, error_code: Optional[str]) -> ValidationResult:
        latency = None
        if attempt.probe_started is not None:
            latency = round((clock() - attempt.probe_started) * 1000.0, 3)
        return ValidationResult(
            request_id=request_id,
            capability=service.service,
            provider_id=service.provider_id,
            status=status,
            latency_ms=latency,
            tested_at=wall_clock(),
            fingerprint=fingerprint,
            credential_version=version,
            error_code=error_code,
        )

    async def run() -> Optional[ValidationResult]:
        resolved = registry.resolve_service(service)
        strategy = select_strategy(resolved)
        if strategy is None:
            return finish(VALIDATION_UNVALIDATED, ERROR_NOT_SUPPORTED)
        _check_secrets(resolved, secrets)
        resolved = strategy.prepare(resolved)
        # Importing an adapter module can block (SDK imports); keep it off the loop.
        loop = asyncio.get_running_loop()
        factory = await loop.run_in_executor(
            None, registry.load_factory, resolved.capability, resolved.provider_id
        )
        builder = family_builders.get(resolved.protocol_family, build_from_resolved)
        try:
            attempt.adapter = builder(factory, resolved, secrets)
        finally:
            secrets.clear()
        attempt.probe_started = clock()
        await strategy.probe(attempt.adapter, resolved)
        return None

    try:
        outcome = await asyncio.wait_for(run(), timeout_s)
        return outcome if outcome is not None else finish(VALIDATION_VALID, None)
    except asyncio.TimeoutError:
        return finish(VALIDATION_INVALID, ERROR_TIMEOUT)
    except ValidationFailedError as failure:
        return finish(VALIDATION_INVALID, failure.code)
    except ProviderRegistryError as err:
        return finish(VALIDATION_INVALID, err.code.value)
    except ProviderRuntimeError as err:
        return finish(VALIDATION_INVALID, err.error.code.value)
    except Exception as exc:
        logger.warning(
            "Provider validation failed: capability=%s provider=%s error=%s",
            service.service,
            service.provider_id,
            type(exc).__name__,
        )
        return finish(VALIDATION_INVALID, ERROR_INTERNAL)
    finally:
        secrets.clear()
        if attempt.adapter is not None:
            await _stop_quietly(attempt.adapter, service.provider_id)


# ── Process entrypoint ───────────────────────────────────────────────────────


def _unvalidated(
    request_id: Optional[str],
    capability: Optional[str],
    provider_id: Optional[str],
    error_code: str,
    status: str = VALIDATION_UNVALIDATED,
) -> ValidationResult:
    return ValidationResult(
        request_id=request_id,
        capability=capability,
        provider_id=provider_id,
        status=status,
        latency_ms=None,
        tested_at=time.time(),
        fingerprint=None,
        credential_version=None,
        error_code=error_code if _ERROR_CODE_RE.fullmatch(error_code) else ERROR_INTERNAL,
    )


async def run_request(
    raw: str,
    *,
    registry: ProviderRegistry,
    builders: Optional[Mapping[ProtocolFamily, AdapterBuilder]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ValidationResult:
    """Parse one request document and validate it. Echoes only identifiers that are safe."""
    try:
        request = json.loads(raw)
    except ValueError:
        return _unvalidated(None, None, None, ERROR_INVALID_REQUEST)
    if not isinstance(request, dict):
        return _unvalidated(None, None, None, ERROR_INVALID_REQUEST)

    secrets = request.pop("secrets", None)
    try:
        raw_id = request.get("id")
        request_id = (
            raw_id if isinstance(raw_id, str) and _REQUEST_ID_RE.fullmatch(raw_id) else None
        )
        raw_capability = request.get("capability")
        capability = raw_capability if raw_capability in SERVICE_NAMES else None
        data = request.get("service")
        raw_provider = data.get("providerId") if isinstance(data, dict) else None
        provider_id = raw_provider if is_valid_id_segment(raw_provider) else None

        unknown = set(request) - _REQUEST_FIELDS
        if request_id is None or capability is None or unknown or not isinstance(secrets, dict):
            return _unvalidated(request_id, capability, provider_id, ERROR_INVALID_REQUEST)
        try:
            service = ServiceConfig.from_dict(capability, data)
        except ConfigError as err:
            return _unvalidated(
                request_id, capability, provider_id, err.code, status=VALIDATION_INVALID
            )
        return await validate_service(
            service,
            secrets,
            registry=registry,
            builders=builders,
            timeout_s=timeout_s,
            request_id=request_id,
        )
    finally:
        if isinstance(secrets, dict):
            secrets.clear()


def _configure_process_logging() -> None:
    """Nothing reaches stderr: engine loggers, the root logger, and warnings are silenced."""
    null = logging.NullHandler()
    root = logging.getLogger()
    root.handlers = [null]
    echoflux = logging.getLogger("echoflux")
    echoflux.handlers = [null]
    echoflux.propagate = False
    logging.lastResort = None
    logging.captureWarnings(True)
    warnings.simplefilter("ignore")


def _read_request(stream: TextIO) -> Optional[str]:
    """At most ``MAX_REQUEST_BYTES`` of UTF-8; None when oversized or undecodable."""
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        data = buffer.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = stream.read(MAX_REQUEST_BYTES + 1)
    if len(text.encode("utf-8")) > MAX_REQUEST_BYTES:
        return None
    return text


def main(
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    *,
    registry: Optional[ProviderRegistry] = None,
    builders: Optional[Mapping[ProtocolFamily, AdapterBuilder]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    configure_logging: bool = True,
) -> int:
    """Read one request, write one result line. Always exits 0 with a result."""
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    if configure_logging:
        _configure_process_logging()

    # Stray prints from adapters or SDKs must not corrupt the single result line or leak.
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        raw = _read_request(source)
        if raw is None:
            result = _unvalidated(None, None, None, ERROR_INVALID_REQUEST)
        else:
            active = registry if registry is not None else build_default_registry()
            result = asyncio.run(
                run_request(raw, registry=active, builders=builders, timeout_s=timeout_s)
            )
        raw = None
    except BaseException as exc:
        logger.debug("Validation entrypoint failed: error=%s", type(exc).__name__)
        result = _unvalidated(None, None, None, ERROR_INTERNAL)
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err

    sink.write(json.dumps(result.to_dict(), separators=(",", ":")) + "\n")
    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
