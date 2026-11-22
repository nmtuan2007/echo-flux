"""Pins the provider registry, schema v2 service config, and v1-to-v2 migration.

No test contacts a provider or uses a real credential. SECRET is a sentinel that must never
appear in serialized config, backups, fingerprints, or error text.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.core.config import (
    SCHEMA_VERSION,
    SERVICE_NAMES,
    ConfigError,
    CredentialRef,
    ServiceConfig,
    strip_endpoint_credentials,
)
from engine.core.config_migration import (
    ConfigMigrationError,
    derive_services,
    migrate_config_file,
    migrate_data,
)
from engine.providers import registry as registry_module
from engine.providers.contracts import (
    AssistantProvider,
    StreamingSTTProvider,
    TranslationProvider,
)
from engine.providers.manifests import (
    BUILTIN_MANIFESTS,
    FieldType,
    ManifestField,
    ProtocolFamily,
    ProviderManifest,
)
from engine.providers.registry import (
    ProviderRegistry,
    ProviderRegistryError,
    RegistryErrorCode,
    build_default_registry,
)
from engine.providers.types import ProviderCapability

SECRET = "test-secret-value-not-real-0001"

STT = ProviderCapability.STT
TRANSLATION = ProviderCapability.TRANSLATION
ASSISTANT = ProviderCapability.ASSISTANT

# Approved v1 provider matrix (master plan): the authoritative upper bound. Registration tasks
# add providers inside it; nothing may appear outside it or under an unapproved capability.
APPROVED_PROVIDERS = {
    STT: {
        "local",
        "openai-realtime",
        "gemini-live",
        "deepgram",
        "azure-speech",
        "assemblyai",
        "custom",
    },
    TRANSLATION: {
        "local",
        "openai",
        "gemini",
        "azure-openai",
        "assemblyai-llm-gateway",
        "custom-openai-compatible",
    },
    ASSISTANT: {
        "openai",
        "gemini",
        "azure-openai",
        "assemblyai-llm-gateway",
        "local-openai-compatible",
        "custom-openai-compatible",
    },
}

SEED_PROVIDERS = {"local", "openai-realtime", "gemini-live", "openai", "custom-openai-compatible"}

FORBIDDEN_PROVIDER_IDS = {"deepl", "google-translate", "azure-translator", "openrouter"}

# Adapters live in the package for the capability they serve.
PLANNED_FACTORY_PACKAGES = {
    STT: "engine.providers.stt.",
    TRANSLATION: "engine.providers.translation.",
    ASSISTANT: "engine.providers.assistant.",
}

BUILTIN_FACTORY_MODULES = {
    ref.partition(":")[0] for manifest in BUILTIN_MANIFESTS for ref in manifest.factories.values()
}

# Approved protocol families per capability. Anything absent from a row is forbidden.
APPROVED_PROTOCOLS = {
    STT: {
        ProtocolFamily.LOCAL,
        ProtocolFamily.OPENAI_REALTIME,
        ProtocolFamily.GEMINI_LIVE,
        ProtocolFamily.DEEPGRAM,
        ProtocolFamily.ASSEMBLYAI_STREAMING,
        ProtocolFamily.AZURE_SPEECH,
    },
    TRANSLATION: {
        ProtocolFamily.LOCAL,
        ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        ProtocolFamily.GEMINI_TEXT,
    },
    ASSISTANT: {ProtocolFamily.OPENAI_COMPATIBLE_TEXT, ProtocolFamily.GEMINI_TEXT},
}


def _api_key_ref(version: int = 1) -> CredentialRef:
    return CredentialRef(config_id="main", slot="api_key", version=version)


def _manifest(**overrides) -> ProviderManifest:
    values = {
        "provider_id": "test-text",
        "display_name": "Test",
        "capabilities": frozenset({TRANSLATION}),
        "protocol_family": ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        "factories": {TRANSLATION: "collections:OrderedDict"},
        "recommended_endpoint": "https://example.invalid/v1",
        "recommended_model": "test-model",
    }
    values.update(overrides)
    return ProviderManifest(**values)


def _legacy_document() -> dict:
    return {
        "engine": {"port": 9001},
        "logging": {"level": "DEBUG"},
        "ui": {"theme": "dark"},
        "asr": {"model_size": "medium", "language": "ja", "device": "cuda"},
        "translation": {
            "enabled": False,
            "backend": "marian",
            "source_lang": "ja",
            "target_lang": "en",
            "model": "Helsinki-NLP/opus-mt-ja-en",
            "model_path": "C:/models/ja-en",
        },
        "llm": {
            "enabled": True,
            "provider_url": "https://openrouter.ai/api/v1",
            "api_key": SECRET,
            "model": "anthropic/claude-test",
        },
    }


def _redacted_legacy_document() -> dict:
    document = _legacy_document()
    document["llm"]["api_key"] = ""
    return document


# --- Registry ---------------------------------------------------------------------------


def test_builtin_providers_stay_inside_the_approved_matrix():
    all_ids = [m.provider_id for m in BUILTIN_MANIFESTS]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) >= SEED_PROVIDERS
    assert not FORBIDDEN_PROVIDER_IDS & set(all_ids)
    for manifest in BUILTIN_MANIFESTS:
        for capability in manifest.capabilities:
            assert manifest.provider_id in APPROVED_PROVIDERS[capability], (
                manifest.provider_id,
                capability,
            )

    # Speech-only vendors never serve text, and there is no local Assistant under "local".
    text_ids = {
        m.provider_id for m in BUILTIN_MANIFESTS if m.supports(TRANSLATION) or m.supports(ASSISTANT)
    }
    assert not {"deepgram", "azure-speech"} & text_ids
    assert "local" not in APPROVED_PROVIDERS[ASSISTANT]
    assert not any(m.provider_id == "local" and m.supports(ASSISTANT) for m in BUILTIN_MANIFESTS)


def test_approved_matrix_itself_matches_the_master_plan():
    # Guards the upper bound against being widened to let a forbidden provider through.
    approved = set().union(*APPROVED_PROVIDERS.values())
    assert not FORBIDDEN_PROVIDER_IDS & approved
    assert not {"deepgram", "azure-speech"} & (
        APPROVED_PROVIDERS[TRANSLATION] | APPROVED_PROVIDERS[ASSISTANT]
    )
    assert approved >= SEED_PROVIDERS


def test_registry_lists_only_providers_supporting_the_capability():
    registry = build_default_registry()
    for capability in ProviderCapability:
        expected = sorted(m.provider_id for m in BUILTIN_MANIFESTS if m.supports(capability))
        for key in (capability, capability.value):
            listed = [m.provider_id for m in registry.providers_for(key)]
            assert listed == expected, capability
        assert all(m.supports(capability) for m in registry.providers_for(capability))

    assert {"local", "openai-realtime", "gemini-live"} <= {
        m.provider_id for m in registry.providers_for(STT)
    }
    assert {"local", "openai", "custom-openai-compatible"} <= {
        m.provider_id for m in registry.providers_for(TRANSLATION)
    }
    assert {"openai", "custom-openai-compatible"} <= {
        m.provider_id for m in registry.providers_for(ASSISTANT)
    }


def test_builtin_protocol_families_are_approved_for_their_capabilities():
    for manifest in BUILTIN_MANIFESTS:
        for capability in manifest.capabilities:
            assert manifest.protocol_family in APPROVED_PROTOCOLS[capability], (
                manifest.provider_id,
                capability,
            )


def _protocol_manifest(capability, family, provider_id="test-protocol"):
    local = family is ProtocolFamily.LOCAL
    return _manifest(
        provider_id=provider_id,
        capabilities=frozenset({capability}),
        protocol_family=family,
        factories={capability: "collections:OrderedDict"},
        model_selectable=not local,
        recommended_model="" if local else "test-model",
    )


@pytest.mark.parametrize("capability", list(ProviderCapability))
@pytest.mark.parametrize("family", list(ProtocolFamily))
def test_protocol_family_capability_table(capability, family):
    manifest = _protocol_manifest(capability, family)
    if family in APPROVED_PROTOCOLS[capability]:
        registry = ProviderRegistry([manifest])
        assert registry.get(capability, "test-protocol").protocol_family is family
    else:
        with pytest.raises(ProviderRegistryError) as excinfo:
            ProviderRegistry([manifest])
        assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST


@pytest.mark.parametrize(
    ("capability", "family"),
    [
        (TRANSLATION, ProtocolFamily.DEEPGRAM),
        (ASSISTANT, ProtocolFamily.AZURE_SPEECH),
        (TRANSLATION, ProtocolFamily.ASSEMBLYAI_STREAMING),
        (STT, ProtocolFamily.GEMINI_TEXT),
        (STT, ProtocolFamily.OPENAI_COMPATIBLE_TEXT),
        (TRANSLATION, ProtocolFamily.OPENAI_REALTIME),
        (ASSISTANT, ProtocolFamily.GEMINI_LIVE),
        (ASSISTANT, ProtocolFamily.LOCAL),
    ],
)
def test_protocol_family_rejections_are_explicit(capability, family):
    with pytest.raises(ProviderRegistryError) as excinfo:
        ProviderRegistry([_protocol_manifest(capability, family)])
    assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST


def test_mixed_capability_manifest_needs_family_approved_for_each():
    # Gemini text may serve both text capabilities, but not STT alongside them.
    both = _manifest(
        capabilities=frozenset({TRANSLATION, ASSISTANT}),
        protocol_family=ProtocolFamily.GEMINI_TEXT,
        factories={TRANSLATION: "collections:OrderedDict", ASSISTANT: "collections:OrderedDict"},
    )
    assert ProviderRegistry([both]).get(ASSISTANT, "test-text").protocol_family is (
        ProtocolFamily.GEMINI_TEXT
    )
    with_stt = _manifest(
        capabilities=frozenset({STT, TRANSLATION}),
        protocol_family=ProtocolFamily.GEMINI_TEXT,
        factories={STT: "collections:OrderedDict", TRANSLATION: "collections:OrderedDict"},
    )
    with pytest.raises(ProviderRegistryError) as excinfo:
        ProviderRegistry([with_stt])
    assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST


@pytest.mark.parametrize(
    ("provider_id", "capability", "allowed"),
    [
        # D-011: Custom STT reuses a realtime protocol profile only.
        ("custom", STT, {ProtocolFamily.OPENAI_REALTIME, ProtocolFamily.GEMINI_LIVE}),
        # D-012: custom and local text APIs are OpenAI-compatible only.
        ("custom-openai-compatible", TRANSLATION, {ProtocolFamily.OPENAI_COMPATIBLE_TEXT}),
        ("custom-openai-compatible", ASSISTANT, {ProtocolFamily.OPENAI_COMPATIBLE_TEXT}),
        ("local-openai-compatible", ASSISTANT, {ProtocolFamily.OPENAI_COMPATIBLE_TEXT}),
    ],
)
def test_custom_providers_are_limited_to_compatible_protocols(provider_id, capability, allowed):
    for family in APPROVED_PROTOCOLS[capability]:
        manifest = _protocol_manifest(capability, family, provider_id=provider_id)
        if family in allowed:
            assert ProviderRegistry([manifest]).get(capability, provider_id)
        else:
            with pytest.raises(ProviderRegistryError) as excinfo:
                ProviderRegistry([manifest])
            assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST, family


def test_unsupported_pair_fails_deterministically():
    registry = build_default_registry()
    messages = set()
    for _ in range(2):
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.get(STT, "openai")
        assert excinfo.value.code is RegistryErrorCode.UNSUPPORTED_CAPABILITY
        messages.add(str(excinfo.value))
    assert messages == {"unsupported_capability capability=stt provider=openai"}

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.get(ASSISTANT, "local")
    assert excinfo.value.code is RegistryErrorCode.UNSUPPORTED_CAPABILITY

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.get(ASSISTANT, "openrouter")
    assert excinfo.value.code is RegistryErrorCode.UNKNOWN_PROVIDER

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.providers_for("vision")
    assert excinfo.value.code is RegistryErrorCode.UNSUPPORTED_CAPABILITY


def test_unknown_provider_error_never_echoes_key_like_ids():
    registry = build_default_registry()
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.get(ASSISTANT, "sk-proj-" + SECRET)
    assert SECRET not in str(excinfo.value)
    assert excinfo.value.provider_id is None


def test_duplicate_registration_is_rejected_and_original_kept():
    registry = ProviderRegistry([_manifest()])
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.register(_manifest(display_name="Imposter"))
    assert excinfo.value.code is RegistryErrorCode.DUPLICATE_PROVIDER
    assert registry.get(TRANSLATION, "test-text").display_name == "Test"


@pytest.mark.parametrize(
    "overrides",
    [
        {"factories": {}},
        {"factories": {TRANSLATION: "not a reference"}},
        {"endpoint_editable": False},
        {"capabilities": frozenset({STT}), "factories": {STT: "collections:OrderedDict"}},
        {
            "capabilities": frozenset({ASSISTANT}),
            "factories": {ASSISTANT: "collections:OrderedDict"},
            "protocol_family": ProtocolFamily.LOCAL,
            "model_selectable": False,
            "recommended_model": "",
        },
        {"protocol_family": ProtocolFamily.LOCAL, "recommended_model": ""},
        {"protocol_family": ProtocolFamily.LOCAL, "model_selectable": False},
        {"capabilities": frozenset()},
        {"provider_id": "Bad Id"},
        {"fields": (ManifestField(key="api_key", label="k", field_type=FieldType.STRING),)},
        {
            "fields": (
                ManifestField(key="mode", label="m", field_type=FieldType.CHOICE, default="x"),
            )
        },
        {
            "fields": (
                ManifestField(
                    key="hint",
                    label="h",
                    field_type=FieldType.STRING,
                    applies_to=frozenset({STT}),
                ),
            )
        },
    ],
)
def test_invalid_manifests_are_rejected(overrides):
    with pytest.raises(ProviderRegistryError) as excinfo:
        ProviderRegistry([_manifest(**overrides)])
    assert excinfo.value.code is RegistryErrorCode.INVALID_MANIFEST


def test_factory_references_point_at_planned_modules():
    for manifest in BUILTIN_MANIFESTS:
        assert set(manifest.factories) == set(manifest.capabilities)
        for capability, ref in manifest.factories.items():
            module_name, sep, attribute = ref.partition(":")
            assert sep and attribute.isidentifier(), ref
            assert module_name.startswith(PLANNED_FACTORY_PACKAGES[capability]), ref
            assert all(part.isidentifier() for part in module_name.split(".")), ref
    local = build_default_registry().get(STT, "local")
    assert local.factories[STT] != local.factories[TRANSLATION]


REPO_ROOT = Path(__file__).resolve().parents[1]

# Libraries an adapter may pull in to reach a provider; a lookup must never load them.
NETWORK_MODULES = ("websockets", "httpx", "aiohttp", "requests", "openai", "google.genai")

CAPABILITY_CONTRACTS = {
    STT: StreamingSTTProvider,
    TRANSLATION: TranslationProvider,
    ASSISTANT: AssistantProvider,
}

_LAZY_IMPORT_PROBE = """
import json, sys
import engine.providers.manifests
import engine.providers.registry as registry_module
from engine.providers.types import ProviderCapability

registry = registry_module.build_default_registry()
for manifest in engine.providers.manifests.BUILTIN_MANIFESTS:
    for capability in manifest.capabilities:
        registry.get(capability, manifest.provider_id)
    for capability in ProviderCapability:
        registry.providers_for(capability)
watched = json.loads(sys.argv[1])
print(json.dumps(sorted(name for name in sys.modules if name in watched)))
"""


def _module_exists(module_name: str) -> bool:
    # find_spec raises when a parent package is missing, e.g. before a later task adds it.
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_registry_lookups_do_not_import_adapters_or_network_libs():
    # A fresh interpreter makes this independent of whatever earlier test modules imported.
    watched = sorted(BUILTIN_FACTORY_MODULES | set(NETWORK_MODULES))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_PROBE, json.dumps(watched)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_factories_are_imported_lazily(monkeypatch):
    calls = []
    real_import = registry_module.importlib.import_module

    def spy(name):
        calls.append(name)
        return real_import(name)

    monkeypatch.setattr(registry_module.importlib, "import_module", spy)
    registry = ProviderRegistry([_manifest()])
    default = build_default_registry()
    for manifest in BUILTIN_MANIFESTS:
        for capability in manifest.capabilities:
            default.get(capability, manifest.provider_id)
    assert calls == []

    from collections import OrderedDict

    assert registry.load_factory(TRANSLATION, "test-text") is OrderedDict
    assert registry.load_factory(TRANSLATION, "test-text") is OrderedDict
    assert calls == ["collections"]


def test_builtin_factories_resolve_or_fail_deterministically():
    registry = build_default_registry()
    for manifest in BUILTIN_MANIFESTS:
        for capability, ref in manifest.factories.items():
            if _module_exists(ref.partition(":")[0]):
                factory = registry.load_factory(capability, manifest.provider_id)
                assert isinstance(factory, type), ref
                assert issubclass(factory, CAPABILITY_CONTRACTS[capability]), ref
                assert registry.load_factory(capability, manifest.provider_id) is factory
            else:
                with pytest.raises(ProviderRegistryError) as excinfo:
                    registry.load_factory(capability, manifest.provider_id)
                assert excinfo.value.code is RegistryErrorCode.FACTORY_UNAVAILABLE, ref


@pytest.mark.parametrize(
    "ref",
    [
        "collections:Nope",
        "engine.providers.ef003_missing_module:Provider",
        "engine.providers.ef003_missing_package.adapter:Provider",
    ],
)
def test_unresolvable_factories_fail_deterministically(ref):
    registry = ProviderRegistry([_manifest(factories={TRANSLATION: ref})])
    for _ in range(2):
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.load_factory(TRANSLATION, "test-text")
        assert excinfo.value.code is RegistryErrorCode.FACTORY_UNAVAILABLE
        assert excinfo.value.provider_id == "test-text"


def _service_for(manifest: ProviderManifest, capability: ProviderCapability, **values):
    refs = {
        spec.key: _api_key_ref() for spec in manifest.secret_fields_for(capability) if spec.required
    }
    return ServiceConfig(
        service=capability.value,
        provider_id=manifest.provider_id,
        enabled=True,
        credential_refs=refs,
        **values,
    )


def _placeholder_value(spec: ManifestField):
    if spec.field_type is FieldType.CHOICE:
        return spec.choices[0]
    if spec.field_type is FieldType.BOOLEAN:
        return False
    if spec.field_type in (FieldType.INTEGER, FieldType.NUMBER):
        value = spec.minimum if spec.minimum is not None else spec.maximum
        value = 1 if value is None else value
        return int(value) if spec.field_type is FieldType.INTEGER else float(value)
    return "placeholder"


def _required_options(manifest: ProviderManifest, capability: ProviderCapability) -> dict:
    # Fill required non-secret fields without a default (e.g. an Azure deployment name).
    return {
        spec.key: _placeholder_value(spec)
        for spec in manifest.option_fields_for(capability)
        if spec.required and spec.default is None
    }


def test_placeholder_options_satisfy_required_fields():
    manifest = _manifest(
        fields=(
            ManifestField(key="deployment", label="d", field_type=FieldType.STRING, required=True),
            ManifestField(
                key="region", label="r", field_type=FieldType.CHOICE, required=True, choices=("a",)
            ),
            ManifestField(
                key="rate", label="n", field_type=FieldType.INTEGER, required=True, minimum=8000
            ),
        )
    )
    registry = ProviderRegistry([manifest])
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.validate_options(TRANSLATION, "test-text", {})
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION
    options = _required_options(manifest, TRANSLATION)
    assert registry.validate_options(TRANSLATION, "test-text", options) == options


def test_api_url_is_editable_for_every_provider():
    registry = build_default_registry()
    custom_url = "https://edited.example.invalid/custom"
    for manifest in BUILTIN_MANIFESTS:
        assert manifest.endpoint_editable
        model = "chosen-model" if manifest.model_selectable else ""
        for capability in manifest.capabilities:
            service = _service_for(
                manifest,
                capability,
                endpoint=custom_url,
                model_override=model,
                options=_required_options(manifest, capability),
            )
            assert registry.resolve_service(service).endpoint == custom_url


def test_recommended_models_and_endpoints_match_references():
    registry = build_default_registry()
    openai_rt = registry.get(STT, "openai-realtime")
    assert openai_rt.recommended_model == "gpt-live-transcribe"
    assert openai_rt.recommended_endpoint.startswith("wss://api.openai.com/")
    gemini = registry.get(STT, "gemini-live")
    assert gemini.recommended_model == "gemini-3.5-transcribe-live"
    assert gemini.recommended_endpoint == (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )
    openai_text = registry.get(ASSISTANT, "openai")
    assert openai_text.recommended_endpoint == "https://api.openai.com/v1"
    assert openai_text.recommended_model == "gpt-5.6-terra"


def test_empty_model_override_resolves_to_manifest_recommendation():
    registry = build_default_registry()
    for capability, provider_id in ((ASSISTANT, "openai"), (STT, "gemini-live")):
        manifest = registry.get(capability, provider_id)
        for override in ("", "   "):
            service = _service_for(manifest, capability, model_override=override)
            resolved = registry.resolve_service(service)
            assert resolved.model == manifest.recommended_model
            assert resolved.endpoint == manifest.recommended_endpoint

    openai = registry.get(ASSISTANT, "openai")
    explicit = _service_for(openai, ASSISTANT, model_override="vendor-model-x")
    assert registry.resolve_service(explicit).model == "vendor-model-x"


def test_local_provider_exposes_no_model_selection():
    registry = build_default_registry()
    local = registry.get(STT, "local")
    assert local.model_selectable is False
    assert local.recommended_model == ""
    assert local.get_field("model_path") is None
    for capability in (STT, TRANSLATION):
        assert (
            registry.resolve_service(
                ServiceConfig(service=capability.value, provider_id="local")
            ).model
            == ""
        )
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.resolve_model(capability, "local", "large-v3")
        assert excinfo.value.code is RegistryErrorCode.MODEL_NOT_SELECTABLE


def test_custom_providers_require_endpoint_and_model():
    registry = build_default_registry()
    custom = registry.get(ASSISTANT, "custom-openai-compatible")
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(_service_for(custom, ASSISTANT, model_override="m"))
    assert excinfo.value.code is RegistryErrorCode.MISSING_ENDPOINT

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(
            _service_for(custom, ASSISTANT, endpoint="https://llm.example.invalid/v1")
        )
    assert excinfo.value.code is RegistryErrorCode.MISSING_MODEL

    # Self-hosted servers may run without a key.
    keyless = _service_for(
        custom, ASSISTANT, endpoint="http://127.0.0.1:1234/v1", model_override="m"
    )
    assert registry.resolve_service(keyless).credential_refs == {}


def test_language_defaults_to_auto_with_advanced_hints():
    registry = build_default_registry()
    for manifest in BUILTIN_MANIFESTS:
        for capability in manifest.capabilities:
            for key in ("language", "source_lang"):
                spec = manifest.get_field(key, capability)
                if spec is not None:
                    assert spec.default == "auto", (manifest.provider_id, key)
                    assert spec.advanced is True, (manifest.provider_id, key)
            target = manifest.get_field("target_lang", capability)
            if target is not None:
                assert target.advanced is False

    stt = registry.validate_options(STT, "local", {"device": "cpu"})
    assert stt == {
        "language": "auto",
        "quality_preset": "balanced",
        "device": "cpu",
        "compute_type": "float16",
    }
    translation = registry.validate_options(TRANSLATION, "local", {})
    assert translation == {"source_lang": "auto", "target_lang": "vi"}


def test_options_are_typed_scoped_and_defaulted():
    registry = build_default_registry()
    for bad in (
        {"device": "quantum"},
        {"unknown_option": 1},
        {"language": 5},
        {"target_lang": "en"},
        {"model_path": "C:/models"},
    ):
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.validate_options(STT, "local", bad)
        assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.validate_options(TRANSLATION, "local", {"device": "cpu"})
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.validate_options(ASSISTANT, "openai", {"temperature": 9.0})
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION
    assert "9.0" not in str(excinfo.value)

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.validate_options(ASSISTANT, "openai", {"api_key": SECRET})
    assert excinfo.value.code is RegistryErrorCode.SECRET_IN_OPTIONS
    assert SECRET not in str(excinfo.value)


def test_cloud_provider_requires_credential_reference():
    registry = build_default_registry()
    service = ServiceConfig(service="assistant", provider_id="openai", enabled=True)
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(service)
    assert excinfo.value.code is RegistryErrorCode.MISSING_CREDENTIAL

    stray = ServiceConfig(
        service="stt",
        provider_id="local",
        credential_refs={"api_key": _api_key_ref()},
    )
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(stray)
    assert excinfo.value.code is RegistryErrorCode.INVALID_CREDENTIAL_REF


# --- Schema v2 config, fingerprint, validation ------------------------------------------


def test_default_services_follow_d003_and_are_registered(make_config):
    config = make_config()
    registry = build_default_registry()
    assert config.schema_version == SCHEMA_VERSION == 2
    assert config.get("schema_version") == 2
    assert config.issues == []
    for name in SERVICE_NAMES:
        service = config.service(name)
        assert service.service == name
        registry.get(name, service.provider_id)
        assert set(config.get(f"services.{name}")) == {
            "enabled",
            "providerId",
            "endpoint",
            "modelOverride",
            "options",
            "credentialRefs",
            "validation",
        }
    assert (config.service("stt").provider_id, config.service("stt").enabled) == ("local", True)
    translation = config.service("translation")
    assert (translation.provider_id, translation.enabled) == ("local", True)
    assistant = config.service("assistant")
    assert (assistant.provider_id, assistant.enabled) == ("openai", False)
    for service in (config.service("stt"), translation):
        registry.resolve_service(service)
    # Legacy runtime keys are untouched.
    assert config.get("asr.model_size") == "small"
    assert config.get("translation.enabled") is False


def test_unknown_service_is_rejected(make_config):
    config = make_config()
    with pytest.raises(ConfigError):
        config.service("vision")


def test_fingerprint_tracks_public_config_and_credential_version():
    base = ServiceConfig(
        service="assistant",
        provider_id="openai",
        endpoint="https://api.openai.com/v1",
        options={"temperature": 0.2},
        credential_refs={"api_key": _api_key_ref(1)},
    )
    fingerprint = base.fingerprint()
    assert fingerprint == ServiceConfig.from_dict("assistant", base.to_dict()).fingerprint()

    changed = [
        base.with_changes(provider_id="custom-openai-compatible"),
        base.with_changes(endpoint="https://proxy.example.invalid/v1"),
        base.with_changes(model_override="vendor-other"),
        base.with_changes(options={"temperature": 0.9}),
        base.with_changes(credential_refs={"api_key": _api_key_ref(2)}),
    ]
    assert all(c.fingerprint() != fingerprint for c in changed)
    assert base.with_changes(enabled=True).fingerprint() == fingerprint
    assert base.with_validation_result(True).fingerprint() == fingerprint
    assert SECRET not in json.dumps(base.to_dict())


def test_relevant_changes_invalidate_validation(make_config):
    config = make_config()
    config.update_service("assistant", credential_refs={"api_key": _api_key_ref(1)})
    validated = config.record_validation("assistant", True, checked_at=1.0)
    assert validated.validation_state() == "valid"

    toggled = config.update_service("assistant", enabled=True)
    assert toggled.validation_state() == "valid"

    rotated = config.update_service("assistant", credential_refs={"api_key": _api_key_ref(2)})
    assert rotated.validation_state() == "unvalidated"
    assert rotated.validation.fingerprint is None

    config.record_validation("assistant", False, error_code="auth_failed")
    assert config.service("assistant").validation_state() == "invalid"
    # A change that bypasses update_service (hand edit, dotted set) is detected as stale.
    config.set("services.assistant.endpoint", "https://proxy.example.invalid/v1")
    assert config.service("assistant").validation_state() == "stale"


def test_validation_does_not_store_free_text():
    service = ServiceConfig(service="assistant", provider_id="openai")
    with pytest.raises(ConfigError):
        service.with_validation_result(False, error_code=f"Invalid key {SECRET}")


@pytest.mark.parametrize(
    "section",
    [
        {"providerId": "openai", "apiKey": SECRET},
        {"providerId": "openai", "options": {"api_key": SECRET}},
        {
            "providerId": "openai",
            "credentialRefs": {"api_key": {"configId": "main", "slot": "api_key", "value": SECRET}},
        },
        {"providerId": "openai", "endpoint": f"https://user:{SECRET}@api.openai.com/v1"},
        {
            "providerId": "custom-openai-compatible",
            "endpoint": f"https://x.invalid/v1?key={SECRET}",
        },
        {
            "providerId": "openai",
            "credentialRefs": {"api_key": {"configId": "sk-proj-abc", "slot": "api_key"}},
        },
    ],
)
def test_secret_values_are_rejected_without_echo(section):
    with pytest.raises(ConfigError) as excinfo:
        ServiceConfig.from_dict("assistant", section)
    assert SECRET not in str(excinfo.value)
    assert "sk-proj" not in str(excinfo.value)


def test_config_with_secret_in_services_degrades_and_never_saves_it(make_config, tmp_path):
    document = {
        "schema_version": 2,
        "services": {"assistant": {"providerId": "openai", "apiKey": SECRET}},
        "llm": {"provider_url": f"https://user:{SECRET}@llm.example.invalid/v1"},
    }
    config = make_config(document, dotenv=f"ECHOFLUX_LLM_API_KEY={SECRET}\n")
    assert config.issues == ["services.assistant:secret_value_rejected"]
    assert config.service("assistant").provider_id == "openai"
    assert SECRET not in json.dumps(config.get("services"))
    assert SECRET not in " ".join(config.issues)
    # The legacy runtime can still read the env-provided key in memory...
    assert config.get("llm.api_key") == SECRET

    config.save()
    # ...but it is never written to disk, nor are URL credentials.
    written = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert SECRET not in written
    saved = json.loads(written)
    assert saved["schema_version"] == 2
    assert saved["llm"]["provider_url"] == "https://llm.example.invalid/v1"


def test_strip_endpoint_credentials_keeps_everything_else():
    url = f"https://user:{SECRET}@llm.example.invalid:8443/v1?key={SECRET}&region=eu#frag"
    assert strip_endpoint_credentials(url) == "https://llm.example.invalid:8443/v1?region=eu#frag"
    clean = "https://api.openai.com/v1?x=1"
    assert strip_endpoint_credentials(clean) == clean


# --- Migration --------------------------------------------------------------------------


def test_migration_maps_legacy_sections_and_drops_raw_key():
    legacy = _legacy_document()
    legacy["llm.api_key"] = SECRET
    migrated, result = migrate_data(legacy)

    assert result.migrated and result.from_version == 1 and result.to_version == 2
    assert result.legacy_secret_detected is True
    assert "llm.api_key:requires_vault_import" in result.notes
    assert SECRET not in json.dumps(migrated)
    assert SECRET not in " ".join(result.notes)
    assert "api_key" not in migrated["llm"]
    assert "llm.api_key" not in migrated

    # Unrelated and legacy flat settings survive untouched for the transitional runtime.
    assert migrated["engine"] == {"port": 9001}
    assert migrated["logging"] == {"level": "DEBUG"}
    assert migrated["ui"] == {"theme": "dark"}
    assert migrated["asr"] == legacy["asr"]
    assert migrated["translation"] == legacy["translation"]
    assert migrated["llm"]["model"] == "anthropic/claude-test"

    services = migrated["services"]
    assert migrated["schema_version"] == 2
    stt = services["stt"]
    assert (stt["providerId"], stt["enabled"]) == ("local", True)
    # Local model choices are never carried into modelOverride (D-032).
    assert stt["modelOverride"] == ""
    assert stt["options"] == {"language": "ja", "device": "cuda"}
    translation = services["translation"]
    assert translation["providerId"] == "local"
    # The legacy file explicitly disabled translation; that choice is preserved.
    assert translation["enabled"] is False
    assert translation["modelOverride"] == ""
    assert translation["options"] == {"source_lang": "ja", "target_lang": "en"}
    assistant = services["assistant"]
    assert assistant["enabled"] is True
    assert assistant["providerId"] == "custom-openai-compatible"
    assert assistant["endpoint"] == "https://openrouter.ai/api/v1"
    assert assistant["modelOverride"] == "anthropic/claude-test"
    assert assistant["credentialRefs"] == {}
    assert assistant["validation"]["status"] == "unvalidated"
    # Input document is not mutated.
    assert legacy["llm"]["api_key"] == SECRET


def test_migration_maps_providers_by_legacy_values():
    registry = build_default_registry()
    cases = [
        ({"translation": {"backend": "online"}}, "translation", "local", True),
        ({"translation": {"enabled": False}}, "translation", "local", False),
        ({}, "translation", "local", True),
        (
            {"llm": {"enabled": True, "provider_url": "https://api.openai.com/v1"}},
            "assistant",
            "openai",
            True,
        ),
        (
            {"llm": {"enabled": True, "provider_url": "https://openrouter.ai/api/v1"}},
            "assistant",
            "custom-openai-compatible",
            True,
        ),
        (
            {"llm": {"provider_url": "http://localhost:1234/v1"}},
            "assistant",
            "custom-openai-compatible",
            False,
        ),
        ({"llm": {"enabled": True, "model": "x"}}, "assistant", "openai", False),
        ({}, "assistant", "openai", False),
    ]
    for document, name, expected_provider, expected_enabled in cases:
        services = derive_services(document)
        assert services[name]["providerId"] == expected_provider, document
        assert services[name]["enabled"] is expected_enabled, document
        registry.get(name, services[name]["providerId"])
    assert derive_services({})["assistant"]["endpoint"] == ""
    _, result = migrate_data({"translation": {"backend": "online"}})
    assert "translation.backend:mapped_to_local" in result.notes


def test_migration_strips_credentials_embedded_in_provider_url():
    document = {
        "llm": {"provider_url": f"https://user:{SECRET}@llm.example.invalid/v1?key={SECRET}&x=1"}
    }
    migrated, result = migrate_data(document)
    assert result.legacy_secret_detected is True
    assert "llm.provider_url:credentials_removed" in result.notes
    assistant = migrated["services"]["assistant"]
    assert assistant["providerId"] == "custom-openai-compatible"
    assert assistant["endpoint"] == "https://llm.example.invalid/v1?x=1"
    assert migrated["llm"]["provider_url"] == "https://llm.example.invalid/v1?x=1"
    assert SECRET not in json.dumps(migrated)


def test_file_migration_writes_redacted_backup_first_and_is_rerunnable(tmp_path, make_config):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_legacy_document()), encoding="utf-8")

    first = migrate_config_file(config_path)
    backup = tmp_path / "config.v1.backup.json"
    assert first.migrated is True
    assert first.legacy_secret_detected is True
    assert first.backup_path == backup
    backup_bytes = backup.read_bytes()
    assert SECRET.encode() not in backup_bytes
    assert json.loads(backup_bytes) == _redacted_legacy_document()
    migrated_bytes = config_path.read_bytes()
    assert SECRET.encode() not in migrated_bytes
    assert json.loads(migrated_bytes)["schema_version"] == 2
    assert not (tmp_path / "config.json.migrating").exists()

    second = migrate_config_file(config_path)
    assert second.migrated is False
    assert second.backup_path is None
    assert config_path.read_bytes() == migrated_bytes
    assert sorted(p.name for p in tmp_path.glob("config.v1.backup*")) == [backup.name]

    # The migrated file loads cleanly and resolves through the registry.
    config = make_config()
    assert config.issues == []
    assert config.loaded_schema_version == 2
    translation = build_default_registry().resolve_service(config.service("translation"))
    assert translation.options == {"source_lang": "ja", "target_lang": "en"}


def test_backup_redacts_dotted_key_and_url_credentials(tmp_path):
    config_path = tmp_path / "config.json"
    document = {
        "llm.api_key": SECRET,
        "llm.provider_url": f"https://user:{SECRET}@llm.example.invalid/v1",
        "llm": {"api_key": SECRET, "provider_url": f"https://llm.example.invalid/v1?key={SECRET}"},
        "vad": {"threshold": 0.7},
    }
    config_path.write_text(json.dumps(document), encoding="utf-8")

    result = migrate_config_file(config_path)
    backup_text = result.backup_path.read_text(encoding="utf-8")
    assert SECRET not in backup_text
    assert json.loads(backup_text) == {
        "llm.api_key": "",
        "llm.provider_url": "https://llm.example.invalid/v1",
        "llm": {"api_key": "", "provider_url": "https://llm.example.invalid/v1"},
        "vad": {"threshold": 0.7},
    }
    assert result.legacy_secret_detected is True


def test_backup_without_secrets_is_byte_identical(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"asr": {"model_size": "tiny"},   "vad": {"enabled": false}}', "utf-8")
    original = config_path.read_bytes()
    result = migrate_config_file(config_path)
    assert result.legacy_secret_detected is False
    assert result.backup_path.read_bytes() == original


def test_backup_never_overwrites_a_different_existing_backup(tmp_path):
    config_path = tmp_path / "config.json"
    existing = tmp_path / "config.v1.backup.json"
    existing.write_text('{"older": true}', encoding="utf-8")
    config_path.write_text(json.dumps({"asr": {"model_size": "tiny"}}), encoding="utf-8")
    original = config_path.read_bytes()

    result = migrate_config_file(config_path)
    assert result.backup_path == tmp_path / "config.v1.backup.1.json"
    assert result.backup_path.read_bytes() == original
    assert existing.read_text(encoding="utf-8") == '{"older": true}'


def test_interrupted_run_reuses_backup_with_same_redacted_content(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_legacy_document()), encoding="utf-8")
    backup = tmp_path / "config.v1.backup.json"
    # An earlier run wrote the redacted backup, then stopped before replacing the config.
    backup.write_text(json.dumps(_redacted_legacy_document()), encoding="utf-8")

    result = migrate_config_file(config_path)
    assert result.backup_path == backup
    assert not (tmp_path / "config.v1.backup.1.json").exists()
    assert SECRET not in backup.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (json.dumps({"schema_version": 99}), "unsupported_schema_version"),
        (json.dumps({"schema_version": "two"}), "invalid_schema_version"),
        ('{"llm": {"api_key": "' + SECRET + '"', "invalid_json"),
    ],
)
def test_file_migration_refuses_unknown_input_without_touching_it(tmp_path, content, code):
    config_path = tmp_path / "config.json"
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigMigrationError) as excinfo:
        migrate_config_file(config_path)
    assert excinfo.value.code == code
    assert SECRET not in str(excinfo.value)
    assert config_path.read_text(encoding="utf-8") == content
    assert not list(tmp_path.glob("config.v*.backup*"))


def test_missing_file_migration_is_a_no_op(tmp_path):
    result = migrate_config_file(tmp_path / "absent.json")
    assert result.migrated is False
    assert not list(tmp_path.iterdir())


def test_legacy_config_stays_readable_and_save_backs_up(make_config, tmp_path):
    config = make_config(_legacy_document())
    assert config.loaded_schema_version == 1
    assert config.issues == []
    # Legacy runtime keys, including the in-memory key, read exactly as before.
    assert config.get("asr.model_size") == "medium"
    assert config.get("translation.model") == "Helsinki-NLP/opus-mt-ja-en"
    assert config.get("llm.api_key") == SECRET
    assert config.get("engine.port") == 9001
    # v2 services are derived in memory without the secret.
    assert config.service("assistant").provider_id == "custom-openai-compatible"
    assert config.service("stt").model_override == ""
    assert SECRET not in json.dumps(config.get("services"))
    assert not list(tmp_path.glob("config.v1.backup*"))

    config.save()
    backup = tmp_path / "config.v1.backup.json"
    assert SECRET not in backup.read_text(encoding="utf-8")
    assert json.loads(backup.read_text(encoding="utf-8")) == _redacted_legacy_document()
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert "api_key" not in saved["llm"]
    assert config.loaded_schema_version == 2

    config.save()
    assert sorted(p.name for p in tmp_path.glob("config.v1.backup*")) == [backup.name]
