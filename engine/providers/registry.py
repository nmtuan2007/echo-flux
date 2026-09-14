"""Capability-safe provider registry with lazy adapter factories.

The registry answers three questions deterministically: which providers serve a capability,
what endpoint/model/options a service config resolves to, and which adapter class to build.
Lookups of an unsupported provider/capability pair always raise the same
``ProviderRegistryError`` code. Errors name providers, capabilities, and field keys only;
option values and secrets never appear in them.

Adapter modules are imported on first ``load_factory`` call, not at registration. Importing
can block, so call it from a worker thread rather than the asyncio event loop.
"""

import importlib
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Tuple, Union

from engine.core.config import (
    CredentialRef,
    ServiceConfig,
    is_secret_key_name,
    is_valid_id_segment,
)
from engine.providers.manifests import (
    BUILTIN_MANIFESTS,
    FieldType,
    ManifestField,
    ProtocolFamily,
    ProviderManifest,
)
from engine.providers.types import ProviderCapability

# Callers may pass the enum or its string value ("stt", "translation", "assistant").
# typing.Union is house style (rule 04); UP007 is the one PEP-604 rule not disabled globally.
CapabilityLike = Union[ProviderCapability, str]  # noqa: UP007

_FACTORY_REF_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")

_ALLOWED_PROTOCOLS: Dict[ProviderCapability, FrozenSet[ProtocolFamily]] = {
    ProviderCapability.STT: frozenset(
        {
            ProtocolFamily.LOCAL,
            ProtocolFamily.OPENAI_REALTIME,
            ProtocolFamily.GEMINI_LIVE,
            ProtocolFamily.DEEPGRAM,
            ProtocolFamily.ASSEMBLYAI_STREAMING,
            ProtocolFamily.AZURE_SPEECH,
        }
    ),
    ProviderCapability.TRANSLATION: frozenset(
        {ProtocolFamily.LOCAL, ProtocolFamily.OPENAI_COMPATIBLE_TEXT, ProtocolFamily.GEMINI_TEXT}
    ),
    # Local Assistant is an OpenAI-compatible server, not an in-process runtime.
    ProviderCapability.ASSISTANT: frozenset(
        {ProtocolFamily.OPENAI_COMPATIBLE_TEXT, ProtocolFamily.GEMINI_TEXT}
    ),
}

# User-configured providers speak a compatible protocol, never a vendor-specific one.
# Custom STT picks a realtime profile (D-011); custom and local text APIs are OpenAI-compatible
# (D-012).
_RESTRICTED_PROVIDER_PROTOCOLS: Dict[str, FrozenSet[ProtocolFamily]] = {
    "custom": frozenset({ProtocolFamily.OPENAI_REALTIME, ProtocolFamily.GEMINI_LIVE}),
    "custom-openai-compatible": frozenset({ProtocolFamily.OPENAI_COMPATIBLE_TEXT}),
    "local-openai-compatible": frozenset({ProtocolFamily.OPENAI_COMPATIBLE_TEXT}),
}


class RegistryErrorCode(str, Enum):
    INVALID_MANIFEST = "invalid_manifest"
    DUPLICATE_PROVIDER = "duplicate_provider"
    UNKNOWN_PROVIDER = "unknown_provider"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    FACTORY_UNAVAILABLE = "factory_unavailable"
    MISSING_ENDPOINT = "missing_endpoint"
    MISSING_MODEL = "missing_model"
    MODEL_NOT_SELECTABLE = "model_not_selectable"
    MISSING_CREDENTIAL = "missing_credential"
    INVALID_CREDENTIAL_REF = "invalid_credential_ref"
    INVALID_OPTION = "invalid_option"
    SECRET_IN_OPTIONS = "secret_in_options"


def _safe_label(value: Any) -> Optional[str]:
    # Ids are echoed only when they pass the vault id rules, so a pasted key never is.
    return value if is_valid_id_segment(value) else None


class ProviderRegistryError(Exception):
    """Deterministic registry failure. Holds identifiers only, never values."""

    def __init__(
        self,
        code: RegistryErrorCode,
        provider_id: Any = None,
        capability: Optional[ProviderCapability] = None,
        field_key: Any = None,
    ) -> None:
        self.code = code
        self.provider_id = _safe_label(provider_id)
        self.capability = capability
        self.field_key = _safe_label(field_key)
        parts = [code.value]
        if capability is not None:
            parts.append(f"capability={capability.value}")
        if self.provider_id is not None:
            parts.append(f"provider={self.provider_id}")
        if self.field_key is not None:
            parts.append(f"field={self.field_key}")
        super().__init__(" ".join(parts))


@dataclass(frozen=True)
class ResolvedService:
    """A service config after manifest defaults and recommendations are applied."""

    capability: ProviderCapability
    provider_id: str
    protocol_family: ProtocolFamily
    factory_ref: str
    endpoint: str
    model: str
    options: Dict[str, Any]
    credential_refs: Dict[str, CredentialRef]
    fingerprint: str


def _coerce_capability(capability: CapabilityLike) -> ProviderCapability:
    if isinstance(capability, ProviderCapability):
        return capability
    try:
        return ProviderCapability(capability)
    except ValueError:
        raise ProviderRegistryError(RegistryErrorCode.UNSUPPORTED_CAPABILITY) from None


def _value_matches(spec: ManifestField, value: Any) -> bool:
    if spec.field_type is FieldType.STRING:
        return isinstance(value, str)
    if spec.field_type is FieldType.BOOLEAN:
        return isinstance(value, bool)
    if spec.field_type is FieldType.CHOICE:
        return isinstance(value, str) and value in spec.choices
    if spec.field_type is FieldType.INTEGER:
        numeric = isinstance(value, int) and not isinstance(value, bool)
    elif spec.field_type is FieldType.NUMBER:
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        return False
    if not numeric:
        return False
    if spec.minimum is not None and value < spec.minimum:
        return False
    return spec.maximum is None or value <= spec.maximum


def _validate_profiles(
    manifest: ProviderManifest,
    restricted: Optional[FrozenSet[ProtocolFamily]],
    invalid: Callable[[], ProviderRegistryError],
) -> None:
    """A profiled manifest picks its family from a bounded, required choice field.

    Every profile family must be allowed for every capability and by the provider's
    restriction, so a profile can never smuggle in an arbitrary protocol or event mapper.
    """
    profiles = manifest.protocol_profiles
    if not isinstance(profiles, dict) or not isinstance(manifest.profile_field, str):
        raise invalid()
    if not profiles:
        if manifest.profile_field:
            raise invalid()
        return
    families = list(profiles.values())
    if families[0] is not manifest.protocol_family:
        raise invalid()
    for name, family in profiles.items():
        if not is_valid_id_segment(name) or not isinstance(family, ProtocolFamily):
            raise invalid()
        if family is ProtocolFamily.LOCAL or (restricted is not None and family not in restricted):
            raise invalid()
        if any(family not in _ALLOWED_PROTOCOLS[cap] for cap in manifest.capabilities):
            raise invalid()
    spec = next((f for f in manifest.fields if f.key == manifest.profile_field), None)
    if (
        spec is None
        or spec.field_type is not FieldType.CHOICE
        or not spec.required
        or spec.default is not None
        or spec.applies_to
        or set(spec.choices) != set(profiles)
        or len(spec.choices) != len(profiles)
    ):
        raise invalid()


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, ProviderManifest):
        raise ProviderRegistryError(RegistryErrorCode.INVALID_MANIFEST)
    pid = manifest.provider_id

    def invalid() -> ProviderRegistryError:
        return ProviderRegistryError(RegistryErrorCode.INVALID_MANIFEST, provider_id=pid)

    if not is_valid_id_segment(pid) or not manifest.capabilities:
        raise invalid()
    if not isinstance(manifest.protocol_family, ProtocolFamily):
        raise invalid()
    if not manifest.endpoint_editable:
        raise invalid()
    restricted = _RESTRICTED_PROVIDER_PROTOCOLS.get(pid)
    if restricted is not None and manifest.protocol_family not in restricted:
        raise invalid()
    if manifest.protocol_family is ProtocolFamily.LOCAL and (
        manifest.model_selectable or manifest.recommended_model
    ):
        raise invalid()
    for capability in manifest.capabilities:
        if not isinstance(capability, ProviderCapability):
            raise invalid()
        if manifest.protocol_family not in _ALLOWED_PROTOCOLS[capability]:
            raise invalid()
    _validate_profiles(manifest, restricted, invalid)
    if set(manifest.factories) != set(manifest.capabilities):
        raise invalid()
    for ref in manifest.factories.values():
        if not isinstance(ref, str) or not _FACTORY_REF_RE.fullmatch(ref):
            raise invalid()
    keys = [spec.key for spec in manifest.fields]
    if len(keys) != len(set(keys)):
        raise invalid()
    for spec in manifest.fields:
        if not is_valid_id_segment(spec.key):
            raise invalid()
        if not spec.applies_to <= manifest.capabilities:
            raise invalid()
        if spec.secret != (spec.field_type is FieldType.SECRET):
            raise invalid()
        if spec.secret and spec.default is not None:
            raise invalid()
        if not spec.secret and is_secret_key_name(spec.key):
            raise invalid()
        if spec.field_type is FieldType.CHOICE and not spec.choices:
            raise invalid()
        if not spec.secret and spec.default is not None and not _value_matches(spec, spec.default):
            raise invalid()


class ProviderRegistry:
    """Thread-safe manifest registry keyed by provider id."""

    def __init__(self, manifests: Iterable[ProviderManifest] = ()) -> None:
        self._lock = threading.Lock()
        self._manifests: Dict[str, ProviderManifest] = {}
        self._factory_cache: Dict[str, Callable[..., Any]] = {}
        for manifest in manifests:
            self.register(manifest)

    def register(self, manifest: ProviderManifest) -> None:
        _validate_manifest(manifest)
        with self._lock:
            if manifest.provider_id in self._manifests:
                raise ProviderRegistryError(
                    RegistryErrorCode.DUPLICATE_PROVIDER, provider_id=manifest.provider_id
                )
            self._manifests[manifest.provider_id] = manifest

    def provider_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._manifests))

    def providers_for(self, capability: CapabilityLike) -> Tuple[ProviderManifest, ...]:
        cap = _coerce_capability(capability)
        with self._lock:
            manifests = list(self._manifests.values())
        return tuple(sorted((m for m in manifests if m.supports(cap)), key=lambda m: m.provider_id))

    def get(self, capability: CapabilityLike, provider_id: str) -> ProviderManifest:
        cap = _coerce_capability(capability)
        with self._lock:
            manifest = self._manifests.get(provider_id) if isinstance(provider_id, str) else None
        if manifest is None:
            raise ProviderRegistryError(
                RegistryErrorCode.UNKNOWN_PROVIDER, provider_id=provider_id, capability=cap
            )
        if not manifest.supports(cap):
            raise ProviderRegistryError(
                RegistryErrorCode.UNSUPPORTED_CAPABILITY, provider_id=provider_id, capability=cap
            )
        return manifest

    def resolve_endpoint(
        self, capability: CapabilityLike, provider_id: str, endpoint: str = ""
    ) -> str:
        manifest = self.get(capability, provider_id)
        override = endpoint.strip() if isinstance(endpoint, str) else ""
        resolved = override or manifest.recommended_endpoint
        if manifest.requires_endpoint and not resolved:
            raise ProviderRegistryError(
                RegistryErrorCode.MISSING_ENDPOINT,
                provider_id=provider_id,
                capability=_coerce_capability(capability),
            )
        return resolved

    def resolve_model(
        self,
        capability: CapabilityLike,
        provider_id: str,
        model_override: Optional[str] = None,
    ) -> str:
        """An empty override falls back to the manifest's recommended model."""
        manifest = self.get(capability, provider_id)
        override = model_override.strip() if isinstance(model_override, str) else ""
        if override and not manifest.model_selectable:
            raise ProviderRegistryError(
                RegistryErrorCode.MODEL_NOT_SELECTABLE,
                provider_id=provider_id,
                capability=_coerce_capability(capability),
            )
        resolved = override or manifest.recommended_model
        # Local runtimes derive a model from their options and some cloud APIs (Azure Speech)
        # have no model parameter; every provider with a selectable model needs a name.
        if not resolved and manifest.model_selectable:
            raise ProviderRegistryError(
                RegistryErrorCode.MISSING_MODEL,
                provider_id=provider_id,
                capability=_coerce_capability(capability),
            )
        return resolved

    def validate_options(
        self,
        capability: CapabilityLike,
        provider_id: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Type-check options against the manifest and fill in defaults."""
        cap = _coerce_capability(capability)
        manifest = self.get(cap, provider_id)

        def fail(code: RegistryErrorCode, key: Any = None) -> ProviderRegistryError:
            return ProviderRegistryError(
                code, provider_id=provider_id, capability=cap, field_key=key
            )

        resolved: Dict[str, Any] = {}
        for key, value in options.items():
            if not isinstance(key, str) or is_secret_key_name(key):
                raise fail(RegistryErrorCode.SECRET_IN_OPTIONS)
            spec = manifest.get_field(key, cap)
            if spec is None:
                raise fail(RegistryErrorCode.INVALID_OPTION, key)
            if spec.secret:
                raise fail(RegistryErrorCode.SECRET_IN_OPTIONS, key)
            if not _value_matches(spec, value):
                raise fail(RegistryErrorCode.INVALID_OPTION, key)
            resolved[key] = value
        for spec in manifest.option_fields_for(cap):
            if spec.key in resolved:
                continue
            if spec.default is not None:
                resolved[spec.key] = spec.default
            elif spec.required:
                raise fail(RegistryErrorCode.INVALID_OPTION, spec.key)
        return resolved

    def resolve_service(self, service: ServiceConfig) -> ResolvedService:
        cap = _coerce_capability(service.service)
        manifest = self.get(cap, service.provider_id)
        pid = manifest.provider_id
        endpoint = self.resolve_endpoint(cap, pid, service.endpoint)
        model = self.resolve_model(cap, pid, service.model_override)
        options = self.validate_options(cap, pid, service.options)
        for slot in service.credential_refs:
            spec = manifest.get_field(slot, cap)
            if spec is None or not spec.secret:
                raise ProviderRegistryError(
                    RegistryErrorCode.INVALID_CREDENTIAL_REF,
                    provider_id=pid,
                    capability=cap,
                    field_key=slot,
                )
        for spec in manifest.secret_fields_for(cap):
            if spec.required and spec.key not in service.credential_refs:
                raise ProviderRegistryError(
                    RegistryErrorCode.MISSING_CREDENTIAL,
                    provider_id=pid,
                    capability=cap,
                    field_key=spec.key,
                )
        return ResolvedService(
            capability=cap,
            provider_id=pid,
            # A profiled provider (Custom STT) speaks the family its validated profile names.
            protocol_family=manifest.protocol_family_for(options),
            factory_ref=manifest.factories[cap],
            endpoint=endpoint,
            model=model,
            options=options,
            credential_refs=dict(service.credential_refs),
            fingerprint=service.fingerprint(),
        )

    def load_factory(self, capability: CapabilityLike, provider_id: str) -> Callable[..., Any]:
        """Import and return the adapter factory. Blocking; keep off the event loop."""
        cap = _coerce_capability(capability)
        ref = self.get(cap, provider_id).factories[cap]
        with self._lock:
            cached = self._factory_cache.get(ref)
        if cached is not None:
            return cached
        module_name, _, attribute = ref.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError):
            factory = None
        if not callable(factory):
            raise ProviderRegistryError(
                RegistryErrorCode.FACTORY_UNAVAILABLE, provider_id=provider_id, capability=cap
            )
        with self._lock:
            self._factory_cache[ref] = factory
        return factory


def build_default_registry() -> ProviderRegistry:
    return ProviderRegistry(BUILTIN_MANIFESTS)
