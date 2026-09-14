"""Pins the approved v1 STT matrix, Custom STT protocol profiles, and STT factory registration.

Nothing here opens a socket or loads a speech SDK: delegates get scripted in-memory
connections, and lazy-import checks run in fresh interpreters so test order cannot hide an
eager import. SECRET and HEADER_SECRET are sentinels that must never appear in errors,
reprs, or logs.
"""

import dataclasses
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import pytest

from engine.core.config import CredentialRef, ServiceConfig
from engine.providers import validation
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.manifests import (
    BUILTIN_MANIFESTS,
    CUSTOM_STT_PROFILES,
    FieldType,
    ManifestField,
    ProtocolFamily,
    ProviderManifest,
)
from engine.providers.registry import (
    ProviderRegistry,
    ProviderRegistryError,
    RegistryErrorCode,
    ResolvedService,
    build_default_registry,
)
from engine.providers.stt import custom as custom_module
from engine.providers.stt import gemini_protocol, openai_protocol
from engine.providers.stt.custom import CustomSTTProvider, CustomSTTSettings
from engine.providers.stt.gemini_live import GeminiLiveSTTProvider
from engine.providers.stt.openai_realtime import OpenAIRealtimeSTTProvider
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

SECRET = "stt-matrix-not-real-EF014-0001"
HEADER_SECRET = "hdr-secret-not-real-EF014-0002"
STT = ProviderCapability.STT
TRANSLATION = ProviderCapability.TRANSLATION
ASSISTANT = ProviderCapability.ASSISTANT
REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ENDPOINT = "wss://stt.example.invalid/realtime"

GEMINI_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# provider id -> (protocol family, factory reference, recommended endpoint, recommended model)
EXPECTED_STT = {
    "local": (ProtocolFamily.LOCAL, "engine.providers.stt.local:LocalSTTProvider", "", ""),
    "openai-realtime": (
        ProtocolFamily.OPENAI_REALTIME,
        "engine.providers.stt.openai_realtime:OpenAIRealtimeSTTProvider",
        "wss://api.openai.com/v1/realtime",
        "gpt-live-transcribe",
    ),
    "gemini-live": (
        ProtocolFamily.GEMINI_LIVE,
        "engine.providers.stt.gemini_live:GeminiLiveSTTProvider",
        GEMINI_ENDPOINT,
        "gemini-3.5-transcribe-live",
    ),
    "deepgram": (
        ProtocolFamily.DEEPGRAM,
        "engine.providers.stt.deepgram:DeepgramSTTProvider",
        "wss://api.deepgram.com/v1/listen",
        "nova-3",
    ),
    "azure-speech": (
        ProtocolFamily.AZURE_SPEECH,
        "engine.providers.stt.azure_speech:AzureSpeechSTTProvider",
        "",
        "",
    ),
    "assemblyai": (
        ProtocolFamily.ASSEMBLYAI_STREAMING,
        "engine.providers.stt.assemblyai:AssemblyAISTTProvider",
        "wss://streaming.assemblyai.com/v3/ws",
        "universal-3-5-pro",
    ),
    "custom": (
        ProtocolFamily.OPENAI_REALTIME,
        "engine.providers.stt.custom:CustomSTTProvider",
        "",
        "",
    ),
}

CLOUD_STT = sorted(set(EXPECTED_STT) - {"local"})
ADAPTER_MODULES = sorted({ref.partition(":")[0] for _, ref, _, _ in EXPECTED_STT.values()})
SDK_AND_NETWORK_MODULES = [
    "websockets",
    "httpx",
    "azure.cognitiveservices",
    "azure.cognitiveservices.speech",
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ref(slot: str, version: int = 1) -> CredentialRef:
    return CredentialRef(config_id="main", slot=slot, version=version)


def _assert_no_secret(*values: Any) -> None:
    for value in values:
        text = value if isinstance(value, str) else repr(value)
        assert SECRET not in text
        assert HEADER_SECRET not in text


def _custom_service(
    profile: Optional[str] = "openai-realtime",
    options: Optional[Dict[str, Any]] = None,
    refs: Sequence[str] = ("api_key",),
    endpoint: str = CUSTOM_ENDPOINT,
    model: str = "rt-model",
) -> ServiceConfig:
    merged: Dict[str, Any] = {} if profile is None else {"protocol_profile": profile}
    merged.update(options or {})
    return ServiceConfig(
        service="stt",
        provider_id="custom",
        endpoint=endpoint,
        model_override=model,
        options=merged,
        credential_refs={slot: _ref(slot) for slot in refs},
    )


HEADER_OPTIONS = {
    "language": "vi",
    "header_1_name": "X-Tenant",
    "header_1_value": "acme",
    "secret_header_1_name": "X-Gateway-Token",
}
HEADER_SECRETS = {"api_key": SECRET, "secret_header_1": HEADER_SECRET}


def _resolved_with_headers(profile: str) -> ResolvedService:
    service = _custom_service(profile, HEADER_OPTIONS, refs=("api_key", "secret_header_1"))
    return build_default_registry().resolve_service(service)


class FakeConnection:
    """Scripted in-memory socket shared by both profile transports."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.sent: List[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if not self.replies:
            raise AssertionError("unexpected recv on scripted connection")
        return self.replies.pop(0)

    async def close(self) -> None:
        self.closed = True


class RecordingConnector:
    """Stands in for the delegate's socket opener; records url and handshake headers."""

    def __init__(self, *replies: str) -> None:
        self.replies = replies
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeConnection] = []

    async def __call__(self, url: str, headers: Dict[str, str], *rest: Any) -> FakeConnection:
        self.calls.append((url, dict(headers)))
        connection = FakeConnection(self.replies)
        self.connections.append(connection)
        return connection


OPENAI_READY = json.dumps({"type": "session.updated", "session": {"id": "sess-1"}})
GEMINI_READY = json.dumps({"setupComplete": {}})
PROFILE_READY = {"openai-realtime": OPENAI_READY, "gemini-live": GEMINI_READY}


@pytest.fixture
def captured_logs():
    records: List[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=logging.DEBUG)
    root = logging.getLogger("echoflux")
    previous = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


# ── Matrix and registration ──────────────────────────────────────────────────


def test_stt_matrix_is_exact_and_each_provider_registered_once():
    registry = build_default_registry()
    assert {m.provider_id for m in registry.providers_for(STT)} == set(EXPECTED_STT)
    ids = [m.provider_id for m in BUILTIN_MANIFESTS]
    for provider_id in EXPECTED_STT:
        assert ids.count(provider_id) == 1, provider_id
    for provider_id in CLOUD_STT:
        assert registry.get(STT, provider_id).capabilities == frozenset({STT})
        for capability in (TRANSLATION, ASSISTANT):
            for call in (registry.get, registry.load_factory):
                with pytest.raises(ProviderRegistryError) as excinfo:
                    call(capability, provider_id)
                assert excinfo.value.code is RegistryErrorCode.UNSUPPORTED_CAPABILITY
    # Local is the one shared manifest, and it never serves the Assistant.
    assert registry.get(STT, "local").capabilities == frozenset({STT, TRANSLATION})


def test_stt_protocols_factories_and_recommendations():
    registry = build_default_registry()
    for provider_id, (family, ref, endpoint, model) in EXPECTED_STT.items():
        manifest = registry.get(STT, provider_id)
        assert manifest.protocol_family is family, provider_id
        assert manifest.factories[STT] == ref, provider_id
        assert manifest.recommended_endpoint == endpoint, provider_id
        assert manifest.recommended_model == model, provider_id
        assert manifest.endpoint_editable, provider_id
    assert registry.get(STT, "custom").requires_endpoint
    assert registry.get(STT, "custom").model_selectable
    assert not registry.get(STT, "azure-speech").model_selectable
    assert not registry.get(STT, "local").model_selectable


def test_stt_factories_resolve_lazily_to_the_streaming_contract():
    registry = build_default_registry()
    for provider_id, (_, ref, _, _) in EXPECTED_STT.items():
        factory = registry.load_factory(STT, provider_id)
        assert isinstance(factory, type) and issubclass(factory, StreamingSTTProvider), ref
        module_name, _, attribute = ref.partition(":")
        assert (factory.__module__, factory.__name__) == (module_name, attribute)
        assert registry.load_factory(STT, provider_id) is factory
    assert registry.load_factory(STT, "custom") is CustomSTTProvider


def test_api_url_is_editable_and_model_override_is_optional():
    registry = build_default_registry()
    edited = "wss://proxy.example.invalid/stt"
    for provider_id in CLOUD_STT:
        manifest = registry.get(STT, provider_id)
        options = {"protocol_profile": "gemini-live"} if provider_id == "custom" else {}
        base = ServiceConfig(
            service="stt",
            provider_id=provider_id,
            options=options,
            credential_refs={"api_key": _ref("api_key")},
        )
        if provider_id == "custom":
            base = base.with_changes(endpoint=edited, model_override="served-model")
        resolved = registry.resolve_service(base)
        assert resolved.endpoint == (
            edited if provider_id == "custom" else manifest.recommended_endpoint
        )
        if provider_id != "custom":
            assert resolved.model == manifest.recommended_model, provider_id
        assert registry.resolve_service(base.with_changes(endpoint=edited)).endpoint == edited
        if manifest.model_selectable:
            override = registry.resolve_service(base.with_changes(model_override="vendor-x"))
            assert override.model == "vendor-x", provider_id
        else:
            with pytest.raises(ProviderRegistryError) as excinfo:
                registry.resolve_service(base.with_changes(model_override="vendor-x"))
            assert excinfo.value.code is RegistryErrorCode.MODEL_NOT_SELECTABLE


def test_every_cloud_stt_provider_requires_a_credential_reference():
    registry = build_default_registry()
    for provider_id in CLOUD_STT:
        api_key = registry.get(STT, provider_id).get_field("api_key", STT)
        assert api_key is not None and api_key.secret and api_key.required, provider_id
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(ServiceConfig(service="stt", provider_id="deepgram"))
    assert excinfo.value.code is RegistryErrorCode.MISSING_CREDENTIAL


# ── Language and provider-aware fields ───────────────────────────────────────


def test_language_auto_detect_is_default_and_hints_are_advanced():
    registry = build_default_registry()
    for provider_id in EXPECTED_STT:
        manifest = registry.get(STT, provider_id)
        language = manifest.get_field("language", STT)
        if provider_id in ("azure-speech", "assemblyai"):
            # Azure identifies only among candidate locales; AssemblyAI takes a code list and
            # a detection flag. Neither has a single hint that could honestly default to auto.
            assert language is None
        else:
            assert language is not None, provider_id
            assert (language.field_type, language.default, language.advanced) == (
                FieldType.STRING,
                "auto",
                True,
            )
        for spec in manifest.option_fields_for(STT):
            assert isinstance(spec.field_type, FieldType)
            if spec.key not in ("region", "protocol_profile"):
                assert spec.advanced, (provider_id, spec.key)

    assert registry.validate_options(STT, "gemini-live", {}) == {"language": "auto"}
    assert registry.validate_options(STT, "openai-realtime", {}) == {"language": "auto"}
    assert registry.validate_options(STT, "deepgram", {}) == {
        "language": "auto",
        "auth_scheme": "token",
    }
    assert registry.validate_options(STT, "assemblyai", {}) == {"auth_mode": "api_key"}


def test_azure_speech_models_candidate_locales_honestly():
    manifest = build_default_registry().get(STT, "azure-speech")
    for key in ("recognition_language", "candidate_languages"):
        spec = manifest.get_field(key, STT)
        assert spec is not None and spec.field_type is FieldType.STRING
        assert spec.advanced and spec.default is None and not spec.required
    mode = manifest.get_field("language_id_mode", STT)
    assert mode.field_type is FieldType.CHOICE and mode.choices == ("AtStart", "Continuous")
    assert (mode.default, mode.advanced) == ("AtStart", True)
    region = manifest.get_field("region", STT)
    assert region.field_type is FieldType.STRING and not region.advanced
    resolved = build_default_registry().resolve_service(
        ServiceConfig(
            service="stt",
            provider_id="azure-speech",
            options={"region": "eastus", "candidate_languages": "en-US,vi-VN"},
            credential_refs={"api_key": _ref("api_key")},
        )
    )
    assert resolved.model == "" and resolved.endpoint == ""
    assert resolved.options == {
        "region": "eastus",
        "candidate_languages": "en-US,vi-VN",
        "language_id_mode": "AtStart",
        "allow_custom_endpoint": False,
    }


@pytest.mark.parametrize(
    ("provider_id", "options"),
    [
        ("openai-realtime", {"transcription_mode": "SMART"}),
        ("openai-realtime", {"delay": "instant"}),
        ("openai-realtime", {"region": "eastus"}),
        ("gemini-live", {"delay": "low"}),
        ("gemini-live", {"transcription_mode": "smart"}),
        ("deepgram", {"utterance_end_ms": 500}),
        ("deepgram", {"language_codes": "en"}),
        ("deepgram", {"smart_format": "yes"}),
        ("deepgram", {"auth_scheme": "basic"}),
        ("assemblyai", {"end_of_turn_confidence_threshold": 1.5}),
        ("assemblyai", {"min_turn_silence_ms": 10}),
        ("assemblyai", {"language": "en"}),
        ("azure-speech", {"language": "en-US"}),
        ("azure-speech", {"language_id_mode": "Sometimes"}),
        ("local", {"region": "eastus"}),
        ("local", {"keyterms": "EchoFlux"}),
        ("custom", {"protocol_profile": "openai-realtime", "header_3_name": "X-Extra"}),
        ("custom", {"protocol_profile": "openai-realtime", "event_mapper": "generic"}),
        ("custom", {"protocol_profile": "openai-realtime", "transcription_mode": "SMART"}),
        ("custom", {"protocol_profile": "gemini-live", "delay": "low"}),
    ],
)
def test_unsupported_or_mistyped_fields_fail_validation(provider_id, options):
    with pytest.raises(ProviderRegistryError) as excinfo:
        build_default_registry().validate_options(STT, provider_id, options)
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION


@pytest.mark.parametrize("provider_id", CLOUD_STT)
def test_secret_values_are_never_accepted_as_options(provider_id):
    registry = build_default_registry()
    for key, value in (("api_key", SECRET), ("token", SECRET), ("secret_header_1", HEADER_SECRET)):
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.validate_options(STT, provider_id, {key: value})
        assert excinfo.value.code in (
            RegistryErrorCode.SECRET_IN_OPTIONS,
            RegistryErrorCode.INVALID_OPTION,
        )
        _assert_no_secret(str(excinfo.value), excinfo.value.field_key)


# ── Custom STT profiles ──────────────────────────────────────────────────────


def test_custom_requires_endpoint_model_profile_and_key():
    registry = build_default_registry()
    cases = [
        (_custom_service(endpoint=""), RegistryErrorCode.MISSING_ENDPOINT, None),
        (_custom_service(model=""), RegistryErrorCode.MISSING_MODEL, None),
        (_custom_service(profile=None), RegistryErrorCode.INVALID_OPTION, "protocol_profile"),
        (_custom_service(refs=()), RegistryErrorCode.MISSING_CREDENTIAL, "api_key"),
    ]
    for service, code, field_key in cases:
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.resolve_service(service)
        assert excinfo.value.code is code
        assert excinfo.value.field_key == field_key


@pytest.mark.parametrize("profile", ["deepgram", "generic", "custom-mapper", "OPENAI-REALTIME", 1])
def test_custom_rejects_profiles_outside_the_bounded_pair(profile):
    with pytest.raises(ProviderRegistryError) as excinfo:
        build_default_registry().resolve_service(_custom_service(profile=profile))
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION
    assert excinfo.value.field_key == "protocol_profile"


def test_custom_profile_selects_the_resolved_protocol_family():
    registry = build_default_registry()
    assert set(CUSTOM_STT_PROFILES) == {"openai-realtime", "gemini-live"}
    for profile, family in CUSTOM_STT_PROFILES.items():
        resolved = registry.resolve_service(_custom_service(profile))
        assert resolved.protocol_family is family
        assert resolved.factory_ref == "engine.providers.stt.custom:CustomSTTProvider"
        assert resolved.options == {"protocol_profile": profile, "language": "auto"}
        assert resolved.model == "rt-model" and resolved.endpoint == CUSTOM_ENDPOINT


def test_custom_headers_keep_public_and_secret_separate():
    registry = build_default_registry()
    manifest = registry.get(STT, "custom")
    assert {spec.key for spec in manifest.secret_fields_for(STT)} == {
        "api_key",
        "secret_header_1",
        "secret_header_2",
    }
    for spec in manifest.secret_fields_for(STT):
        assert spec.field_type is FieldType.SECRET and spec.default is None
    option_keys = {spec.key for spec in manifest.option_fields_for(STT)}
    assert {
        "header_1_name",
        "header_1_value",
        "header_2_name",
        "header_2_value",
        "secret_header_1_name",
        "secret_header_2_name",
    } <= option_keys

    service = _custom_service(options=HEADER_OPTIONS, refs=("api_key", "secret_header_1"))
    resolved = registry.resolve_service(service)
    assert set(resolved.credential_refs) == {"api_key", "secret_header_1"}
    assert "secret_header_1" not in resolved.options
    document = json.dumps(service.to_dict())
    _assert_no_secret(document)
    round_trip = ServiceConfig.from_dict("stt", json.loads(document))
    assert round_trip.fingerprint() == service.fingerprint()

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(_custom_service(refs=("api_key", "secret_header_3")))
    assert excinfo.value.code is RegistryErrorCode.INVALID_CREDENTIAL_REF
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(_custom_service(refs=("api_key", "header_1_value")))
    assert excinfo.value.code is RegistryErrorCode.INVALID_CREDENTIAL_REF


def _profile_field(choices: Sequence[str], **overrides: Any) -> ManifestField:
    values: Dict[str, Any] = {
        "key": "protocol_profile",
        "label": "Protocol",
        "field_type": FieldType.CHOICE,
        "required": True,
        "choices": tuple(choices),
    }
    values.update(overrides)
    return ManifestField(**values)


def _profiled_manifest(
    profiles: Dict[str, ProtocolFamily],
    field: Optional[ManifestField] = None,
    **overrides: Any,
) -> ProviderManifest:
    values: Dict[str, Any] = {
        "provider_id": "custom",
        "display_name": "Custom",
        "capabilities": frozenset({STT}),
        "protocol_family": next(iter(profiles.values()), ProtocolFamily.OPENAI_REALTIME),
        "factories": {STT: "collections:OrderedDict"},
        "protocol_profiles": profiles,
        "profile_field": "protocol_profile",
        "fields": (field or _profile_field(list(profiles)),),
    }
    values.update(overrides)
    return ProviderManifest(**values)


def test_valid_profiled_manifest_registers():
    manifest = _profiled_manifest(dict(CUSTOM_STT_PROFILES))
    registry = ProviderRegistry([manifest])
    for profile, family in CUSTOM_STT_PROFILES.items():
        service = ServiceConfig(
            service="stt",
            provider_id="custom",
            model_override="m",
            options={"protocol_profile": profile},
        )
        assert registry.resolve_service(service).protocol_family is family


_REALTIME = ProtocolFamily.OPENAI_REALTIME


@pytest.mark.parametrize(
    "manifest_factory",
    [
        # Custom may only profile the two realtime families (D-011).
        lambda: _profiled_manifest({"openai-realtime": _REALTIME, "dg": ProtocolFamily.DEEPGRAM}),
        lambda: _profiled_manifest(
            {"openai-realtime": _REALTIME, "mapper": ProtocolFamily.OPENAI_COMPATIBLE_TEXT}
        ),
        lambda: _profiled_manifest({"openai-realtime": _REALTIME, "local": ProtocolFamily.LOCAL}),
        lambda: _profiled_manifest(
            dict(CUSTOM_STT_PROFILES), field=_profile_field(["openai-realtime", "generic"])
        ),
        lambda: _profiled_manifest(
            dict(CUSTOM_STT_PROFILES),
            field=_profile_field(list(CUSTOM_STT_PROFILES), required=False),
        ),
        lambda: _profiled_manifest(
            dict(CUSTOM_STT_PROFILES),
            field=_profile_field(list(CUSTOM_STT_PROFILES), default="openai-realtime"),
        ),
        lambda: _profiled_manifest(
            dict(CUSTOM_STT_PROFILES),
            field=_profile_field(list(CUSTOM_STT_PROFILES), field_type=FieldType.STRING),
        ),
        lambda: _profiled_manifest(dict(CUSTOM_STT_PROFILES), profile_field=""),
        lambda: _profiled_manifest({}, field=_profile_field(["openai-realtime"])),
        lambda: _profiled_manifest(
            dict(CUSTOM_STT_PROFILES), protocol_family=ProtocolFamily.GEMINI_LIVE
        ),
        lambda: _profiled_manifest({"Not Valid": _REALTIME}),
    ],
)
def test_inconsistent_or_unbounded_profiles_are_invalid_manifests(manifest_factory):
    with pytest.raises(ProviderRegistryError) as excinfo:
        ProviderRegistry([manifest_factory()])
    assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST


# ── Custom delegation ────────────────────────────────────────────────────────


async def test_openai_profile_delegates_to_openai_realtime_protocol_with_headers(captured_logs):
    connector = RecordingConnector(OPENAI_READY)
    provider = CustomSTTProvider.from_resolved(
        _resolved_with_headers("openai-realtime"), dict(HEADER_SECRETS), connect_factory=connector
    )
    assert isinstance(provider, StreamingSTTProvider)
    assert type(provider.delegate) is OpenAIRealtimeSTTProvider
    assert (provider.provider_id, provider.protocol_profile) == ("custom", "openai-realtime")
    assert provider.capability is STT

    check = await provider.health_check()
    assert check.status is ProviderHealthStatus.HEALTHY
    url, headers = connector.calls[0]
    assert url == openai_protocol.build_connection_url(CUSTOM_ENDPOINT)
    assert headers == {
        "Authorization": f"Bearer {SECRET}",
        "X-Tenant": "acme",
        "X-Gateway-Token": HEADER_SECRET,
    }
    expected = openai_protocol.build_session_update(
        openai_protocol.OpenAIRealtimeSettings(
            endpoint=CUSTOM_ENDPOINT, model="rt-model", api_key=SECRET, languages=("vi",)
        )
    )
    assert [json.loads(frame) for frame in connector.connections[0].sent] == [expected]
    assert connector.connections[0].closed
    _assert_no_secret(
        repr(provider.delegate), check, "\n".join(r.getMessage() for r in captured_logs)
    )


async def test_gemini_profile_delegates_to_gemini_live_protocol_with_headers(captured_logs):
    connector = RecordingConnector(GEMINI_READY)
    provider = CustomSTTProvider.from_resolved(
        _resolved_with_headers("gemini-live"), dict(HEADER_SECRETS), connect_factory=connector
    )
    assert type(provider.delegate) is GeminiLiveSTTProvider
    assert (provider.provider_id, provider.protocol_profile) == ("custom", "gemini-live")

    check = await provider.health_check()
    assert check.status is ProviderHealthStatus.HEALTHY
    url, headers = connector.calls[0]
    assert url == CUSTOM_ENDPOINT
    assert headers == {
        gemini_protocol.API_KEY_HEADER: SECRET,
        "X-Tenant": "acme",
        "X-Gateway-Token": HEADER_SECRET,
    }
    expected = gemini_protocol.build_setup_message("rt-model", ("vi",))
    assert [json.loads(frame) for frame in connector.connections[0].sent] == [expected]
    assert connector.connections[0].closed
    _assert_no_secret(check, "\n".join(r.getMessage() for r in captured_logs))


class _RecordingDelegate(StreamingSTTProvider):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []
        self.result = STTTranscriptResult(
            stream_id="mic",
            sequence=0,
            provider_event_id="item-1",
            text="xin chao",
            is_final=True,
            revision=1,
            start_time_ms=0.0,
            end_time_ms=10.0,
            monotonic_timestamp=1.0,
        )

    @property
    def provider_id(self) -> str:
        return "custom"

    async def start(self) -> None:
        self.calls.append(("start", None))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def health_check(self) -> ProviderHealthCheck:
        self.calls.append(("health_check", None))
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    async def send_audio(self, frame: AudioFrame) -> None:
        self.calls.append(("send_audio", frame))

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        self.calls.append(("receive_events", None))
        yield self.result

    async def finalize_stream(self, stream_id: str) -> None:
        self.calls.append(("finalize_stream", stream_id))

    async def reset_stream(self, stream_id: str) -> None:
        self.calls.append(("reset_stream", stream_id))


async def test_every_streaming_call_is_forwarded_to_the_delegate(monkeypatch):
    delegate = _RecordingDelegate()
    seen: List[Tuple[CustomSTTSettings, Tuple[str, ...]]] = []

    def build(settings: CustomSTTSettings, stream_ids: Tuple[str, ...], base: Any) -> Any:
        seen.append((settings, stream_ids))
        return delegate

    monkeypatch.setitem(custom_module._DELEGATE_BUILDERS, "openai-realtime", build)
    provider = CustomSTTProvider.from_resolved(
        _resolved_with_headers("openai-realtime"),
        dict(HEADER_SECRETS),
        stream_ids=("mic", "system"),
    )
    frame = AudioFrame(stream_id="mic", sequence=1, pcm_data=b"\x00\x00" * 160)
    await provider.start()
    await provider.send_audio(frame)
    events = [event async for event in provider.receive_events()]
    await provider.finalize_stream("mic")
    await provider.reset_stream("system")
    assert (await provider.health_check()).status is ProviderHealthStatus.HEALTHY
    await provider.stop()

    assert events == [delegate.result]
    assert [name for name, _ in delegate.calls] == [
        "start",
        "send_audio",
        "receive_events",
        "finalize_stream",
        "reset_stream",
        "health_check",
        "stop",
    ]
    assert delegate.calls[1][1] is frame
    settings, stream_ids = seen[0]
    assert stream_ids == ("mic", "system")
    assert settings.headers == (("X-Tenant", "acme"),)
    assert settings.secret_headers == (("X-Gateway-Token", HEADER_SECRET),)
    assert settings.language_hints == ("vi",)
    _assert_no_secret(repr(settings))


def test_custom_module_reuses_protocol_adapters_without_its_own_parser():
    source = (REPO_ROOT / "engine/providers/stt/custom.py").read_text(encoding="utf-8")
    assert "from engine.providers.stt.openai_realtime import" in source
    assert "from engine.providers.stt.gemini_live import" in source
    for marker in ("import json", "json.loads", "def parse", "event_mapper", "ServerEvent"):
        assert marker not in source, marker
    assert set(custom_module._DELEGATE_BUILDERS) == set(CUSTOM_STT_PROFILES)


_FIELD_CASES = [
    ({"header_1_name": "Authorization", "header_1_value": "x"}, (), "header_1_name"),
    ({"header_1_name": "X-Api-Key", "header_1_value": "plain"}, (), "header_1_name"),
    ({"header_1_name": "X-Session-Id", "header_1_value": "plain"}, (), "header_1_name"),
    ({"header_1_name": "Bad Name", "header_1_value": "plain"}, (), "header_1_name"),
    ({"header_1_name": "X-Tenant"}, (), "header_1_value"),
    ({"header_1_value": "acme"}, (), "header_1_name"),
    ({"header_1_name": "X-Tenant", "header_1_value": "a\r\nX-Injected: 1"}, (), "header_1_value"),
    ({"secret_header_1_name": "Authorization"}, ("secret_header_1",), "secret_header_1_name"),
    ({"secret_header_1_name": "x-goog-api-key"}, ("secret_header_1",), "secret_header_1_name"),
    (
        {"secret_header_1_name": "Sec-WebSocket-Protocol"},
        ("secret_header_1",),
        "secret_header_1_name",
    ),
    (
        {"header_1_name": "X-Tenant", "header_1_value": "a", "secret_header_1_name": "x-tenant"},
        ("secret_header_1",),
        "secret_header_1_name",
    ),
    ({}, ("secret_header_1",), "secret_header_1_name"),
]


@pytest.mark.parametrize("profile", sorted(CUSTOM_STT_PROFILES))
@pytest.mark.parametrize(("options", "extra_refs", "field_name"), _FIELD_CASES)
def test_unsafe_or_incomplete_headers_are_rejected_before_any_socket(
    profile, options, extra_refs, field_name
):
    service = _custom_service(profile, options, refs=("api_key", *extra_refs))
    resolved = build_default_registry().resolve_service(service)
    connector = RecordingConnector(PROFILE_READY[profile])
    with pytest.raises(ProviderRuntimeError) as excinfo:
        CustomSTTProvider.from_resolved(resolved, dict(HEADER_SECRETS), connect_factory=connector)
    error = excinfo.value.error
    assert error.code is ProviderErrorCode.INVALID_REQUEST
    assert error.capability is STT
    assert error.details == {"field": field_name}
    assert connector.calls == []
    _assert_no_secret(str(excinfo.value), error)


@pytest.mark.parametrize(
    ("refs", "secrets"),
    [
        # A named secret header without a credential reference never reads a raw value.
        ((), {"api_key": SECRET, "secret_header_1": HEADER_SECRET}),
        (("secret_header_1",), {"api_key": SECRET}),
        (("secret_header_1",), {"api_key": SECRET, "secret_header_1": "  "}),
    ],
)
def test_secret_header_values_come_only_from_credential_references(refs, secrets):
    service = _custom_service(
        options={"secret_header_1_name": "X-Gateway-Token"}, refs=("api_key", *refs)
    )
    resolved = build_default_registry().resolve_service(service)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        CustomSTTProvider.from_resolved(resolved, secrets)
    assert excinfo.value.error.code is ProviderErrorCode.AUTH_FAILED
    assert excinfo.value.error.details == {"field": "secret_header_1"}
    _assert_no_secret(str(excinfo.value))


@pytest.mark.parametrize(
    ("changes", "code", "field_name"),
    [
        (
            {"protocol_profile": "generic-mapper"},
            ProviderErrorCode.INVALID_REQUEST,
            "protocol_profile",
        ),
        (
            {"endpoint": "https://stt.example.invalid/rt"},
            ProviderErrorCode.INVALID_REQUEST,
            "endpoint",
        ),
        (
            {"endpoint": "ws://stt.example.invalid/rt"},
            ProviderErrorCode.INVALID_REQUEST,
            "endpoint",
        ),
        (
            {"endpoint": f"wss://user:{SECRET}@stt.example.invalid/rt"},
            ProviderErrorCode.INVALID_REQUEST,
            "endpoint",
        ),
        ({"model": " "}, ProviderErrorCode.INVALID_REQUEST, "model"),
        ({"api_key": ""}, ProviderErrorCode.AUTH_FAILED, "api_key"),
        ({"language_hints": ("",)}, ProviderErrorCode.INVALID_REQUEST, "language"),
    ],
)
def test_custom_settings_are_validated_before_delegation(changes, code, field_name):
    settings = CustomSTTSettings(
        protocol_profile="gemini-live",
        endpoint=CUSTOM_ENDPOINT,
        model="rt-model",
        api_key=SECRET,
        secret_headers=(("X-Gateway-Token", HEADER_SECRET),),
    )
    _assert_no_secret(repr(settings))
    connector = RecordingConnector(GEMINI_READY)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        CustomSTTProvider(dataclasses.replace(settings, **changes), connect_factory=connector)
    assert excinfo.value.error.code is code
    assert excinfo.value.error.details == {"field": field_name}
    assert connector.calls == []
    _assert_no_secret(str(excinfo.value))


def test_loopback_ws_endpoint_is_allowed_for_self_hosted_servers():
    service = _custom_service("gemini-live", endpoint="ws://127.0.0.1:9000/live")
    resolved = build_default_registry().resolve_service(service)
    provider = CustomSTTProvider.from_resolved(resolved, {"api_key": SECRET})
    assert type(provider.delegate) is GeminiLiveSTTProvider


def test_from_resolved_rejects_mismatched_resolution():
    resolved = build_default_registry().resolve_service(_custom_service("openai-realtime"))
    secrets = {"api_key": SECRET}
    bad = [
        (dataclasses.replace(resolved, capability=TRANSLATION), "capability"),
        (
            dataclasses.replace(resolved, protocol_family=ProtocolFamily.GEMINI_LIVE),
            "protocol_profile",
        ),
        (
            dataclasses.replace(resolved, options={**resolved.options, "protocol_profile": "dg"}),
            "protocol_profile",
        ),
    ]
    for candidate, field_name in bad:
        with pytest.raises(ProviderRuntimeError) as excinfo:
            CustomSTTProvider.from_resolved(candidate, secrets)
        assert excinfo.value.error.details == {"field": field_name}


# ── Test Connection (EF-007) construction ────────────────────────────────────


@pytest.mark.parametrize("profile", sorted(CUSTOM_STT_PROFILES))
async def test_custom_test_connection_builds_through_from_resolved(profile):
    connector = RecordingConnector(PROFILE_READY[profile])

    def builder(factory: Any, resolved: ResolvedService, secrets: Dict[str, str]) -> Any:
        assert factory is CustomSTTProvider
        return factory.from_resolved(resolved, secrets, connect_factory=connector)

    service = _custom_service(profile, HEADER_OPTIONS, refs=("api_key", "secret_header_1"))
    secrets = dict(HEADER_SECRETS)
    result = await validation.validate_service(
        service,
        secrets,
        registry=build_default_registry(),
        builders={family: builder for family in CUSTOM_STT_PROFILES.values()},
        timeout_s=5.0,
    )
    assert (result.status, result.error_code) == ("valid", None)
    assert result.provider_id == "custom" and result.fingerprint == service.fingerprint()
    assert secrets == {}
    assert connector.calls[0][1]["X-Gateway-Token"] == HEADER_SECRET
    _assert_no_secret(json.dumps(result.to_dict()))


def test_default_family_builders_and_generic_hook_construct_custom():
    registry = build_default_registry()
    secrets = {"api_key": SECRET}
    cases = [
        ("openai-realtime", validation.build_openai_realtime, OpenAIRealtimeSTTProvider),
        ("gemini-live", validation.build_gemini_live, GeminiLiveSTTProvider),
    ]
    for profile, builder, delegate_type in cases:
        resolved = registry.resolve_service(_custom_service(profile, {"language": "vi"}))
        for build in (builder, validation.build_from_resolved):
            provider = build(CustomSTTProvider, resolved, secrets)
            assert isinstance(provider, CustomSTTProvider)
            assert type(provider.delegate) is delegate_type
            assert (provider.provider_id, provider.protocol_profile) == ("custom", profile)


@pytest.mark.parametrize(
    ("provider_id", "options"),
    [
        ("deepgram", {}),
        ("assemblyai", {}),
        ("azure-speech", {"region": "eastus", "recognition_language": "en-US"}),
    ],
)
async def test_vendor_streaming_test_connection_reports_factory_unavailable(provider_id, options):
    # EF-007 has no family builder for these protocols and their adapters expose no
    # ``from_resolved`` hook, so Test Connection fails deterministically before any socket.
    service = ServiceConfig(
        service="stt",
        provider_id=provider_id,
        options=options,
        credential_refs={"api_key": _ref("api_key")},
    )
    result = await validation.validate_service(
        service, {"api_key": SECRET}, registry=build_default_registry(), timeout_s=5.0
    )
    assert (result.status, result.error_code) == ("invalid", "factory_unavailable")


async def test_azure_speech_factory_loads_without_sdk_and_reports_it_deterministically():
    factory = build_default_registry().load_factory(STT, "azure-speech")
    from engine.providers.stt.azure_speech import AzureSpeechSTTConfig

    def missing_sdk() -> Any:
        raise ImportError("azure.cognitiveservices.speech")

    config = AzureSpeechSTTConfig(api_key=SECRET, region="eastus", recognition_language="en-US")
    for _ in range(2):
        check = await factory(config, sdk_loader=missing_sdk).health_check()
        assert check.status is ProviderHealthStatus.UNHEALTHY
        assert check.details == {"code": "model_unavailable", "retryable": False}
        _assert_no_secret(check)


# ── Import hygiene ───────────────────────────────────────────────────────────

_IMPORT_PROBE = """
import json, sys
stage = sys.argv[1]
if stage == "registry":
    from engine.core.config import CredentialRef, ServiceConfig
    from engine.providers.registry import build_default_registry
    registry = build_default_registry()
    for manifest in registry.providers_for("stt"):
        registry.get("stt", manifest.provider_id)
    registry.resolve_service(ServiceConfig(
        service="stt", provider_id="custom", endpoint="wss://x.example.invalid/rt",
        model_override="m", options={"protocol_profile": "gemini-live"},
        credential_refs={"api_key": CredentialRef(config_id="main", slot="api_key")},
    ))
elif stage == "package":
    import engine.providers.stt as stt
    assert "CustomSTTProvider" in dir(stt) and "AzureSpeechSTTProvider" in stt.__all__
elif stage == "custom":
    import engine.providers.stt.custom
elif stage == "deepgram_export":
    import engine.providers.stt as stt
    assert stt.DeepgramSTTProvider.__name__ == "DeepgramSTTProvider"
elif stage == "azure_factory":
    from engine.providers.registry import build_default_registry
    build_default_registry().load_factory("stt", "azure-speech")
print(json.dumps(sorted(sys.modules)))
"""


def _adapter(name: str) -> str:
    return f"engine.providers.stt.{name}"


@pytest.mark.parametrize(
    ("stage", "required"),
    [
        ("registry", []),
        ("package", []),
        ("custom", [_adapter("custom")]),
        ("deepgram_export", [_adapter("deepgram")]),
        ("azure_factory", [_adapter("azure_speech")]),
    ],
)
def test_stt_imports_stay_lazy_in_a_fresh_interpreter(stage, required):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, stage],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    forbidden = (set(ADAPTER_MODULES) | set(SDK_AND_NETWORK_MODULES)) - set(required)
    forbidden |= {_adapter("openai_realtime"), _adapter("gemini_live")} - set(required)
    assert not forbidden & loaded, sorted(forbidden & loaded)
    assert set(required) <= loaded
