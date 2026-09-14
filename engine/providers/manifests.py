"""Built-in provider manifests and their dynamic field definitions.

A manifest is static metadata: which capabilities a provider serves, which wire protocol it
speaks, what endpoint and model it recommends, and which typed fields the settings UI may
render. The adapter itself is referenced by a ``"module:attribute"`` string so nothing here
imports network clients or ML runtimes; ``engine.providers.registry`` resolves the reference
only when a session actually needs the adapter.

Secret fields describe credential slots. Their values live in the OS vault and reach config
only as ``credentialRefs``, never as options.

The Translation and Assistant matrices are complete. STT is still seeded partially: Deepgram,
Azure Speech, AssemblyAI, and Custom STT (a single ``custom`` provider with a protocol
profile) are added by the tasks that implement them.

Recommended endpoints and models were checked against official provider documentation on
2026-09-14. They are the only place provider defaults live; adapters receive resolved values.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

from engine.providers.types import ProviderCapability


class ProtocolFamily(str, Enum):
    """Wire protocol an adapter speaks. Custom providers reuse one of the cloud families.

    Which capabilities may use each family is enforced by ``engine.providers.registry``.
    """

    LOCAL = "local"
    OPENAI_REALTIME = "openai_realtime"
    GEMINI_LIVE = "gemini_live"
    OPENAI_COMPATIBLE_TEXT = "openai_compatible_text"
    # Vendor-specific protocols with no custom-compatible variant (D-011, D-012).
    GEMINI_TEXT = "gemini_text"
    DEEPGRAM = "deepgram"
    ASSEMBLYAI_STREAMING = "assemblyai_streaming"
    AZURE_SPEECH = "azure_speech"


class FieldType(str, Enum):
    """Value type of a manifest field, used for option validation and UI rendering."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    SECRET = "secret"


@dataclass(frozen=True)
class ManifestField:
    """One typed, provider-specific setting."""

    key: str
    label: str
    field_type: FieldType
    default: Any = None
    required: bool = False
    advanced: bool = False
    secret: bool = False
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    description: str = ""
    # Capabilities this field applies to; empty means every capability of the provider.
    applies_to: FrozenSet[ProviderCapability] = frozenset()

    def applies(self, capability: ProviderCapability) -> bool:
        return not self.applies_to or capability in self.applies_to


@dataclass(frozen=True)
class ProviderManifest:
    """Registry entry describing one provider implementation."""

    provider_id: str
    display_name: str
    capabilities: FrozenSet[ProviderCapability]
    protocol_family: ProtocolFamily
    # Lazy adapter references per capability, as "package.module:Attribute".
    factories: Dict[ProviderCapability, str]
    recommended_endpoint: str = ""
    recommended_model: str = ""
    # Every provider's API URL is user-editable; kept explicit so the registry can enforce it.
    endpoint_editable: bool = True
    requires_endpoint: bool = False
    # Local runtimes use curated model packs, never a free-form model name (D-032).
    model_selectable: bool = True
    fields: Tuple[ManifestField, ...] = field(default_factory=tuple)

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    def fields_for(self, capability: ProviderCapability) -> Tuple[ManifestField, ...]:
        return tuple(f for f in self.fields if f.applies(capability))

    def option_fields_for(self, capability: ProviderCapability) -> Tuple[ManifestField, ...]:
        return tuple(f for f in self.fields_for(capability) if not f.secret)

    def secret_fields_for(self, capability: ProviderCapability) -> Tuple[ManifestField, ...]:
        return tuple(f for f in self.fields_for(capability) if f.secret)

    def get_field(
        self, key: str, capability: Optional[ProviderCapability] = None
    ) -> Optional[ManifestField]:
        for candidate in self.fields:
            if candidate.key == key and (capability is None or candidate.applies(capability)):
                return candidate
        return None


_STT = ProviderCapability.STT
_TRANSLATION = ProviderCapability.TRANSLATION
_ASSISTANT = ProviderCapability.ASSISTANT


def _api_key(required: bool) -> ManifestField:
    return ManifestField(
        key="api_key",
        label="API key",
        field_type=FieldType.SECRET,
        required=required,
        secret=True,
        description="Stored in the OS credential vault; config keeps only a reference.",
    )


# Language is auto-detected by default; explicit hints are an Advanced setting (D-013).
_LANGUAGE_HINT = ManifestField(
    key="language",
    label="Language hint",
    field_type=FieldType.STRING,
    default="auto",
    advanced=True,
    applies_to=frozenset({_STT}),
)

_SOURCE_LANG = ManifestField(
    key="source_lang",
    label="Source language",
    field_type=FieldType.STRING,
    default="auto",
    advanced=True,
    applies_to=frozenset({_TRANSLATION}),
)

_TARGET_LANG = ManifestField(
    key="target_lang",
    label="Target language",
    field_type=FieldType.STRING,
    default="vi",
    required=True,
    applies_to=frozenset({_TRANSLATION}),
)

_TEMPERATURE = ManifestField(
    key="temperature",
    label="Temperature",
    field_type=FieldType.NUMBER,
    default=0.3,
    advanced=True,
    minimum=0.0,
    maximum=2.0,
)

_MAX_OUTPUT_TOKENS = ManifestField(
    key="max_output_tokens",
    label="Max output tokens",
    field_type=FieldType.INTEGER,
    default=1000,
    advanced=True,
    minimum=1,
)

# Planned adapter locations. Until the adapter tasks land these modules do not exist, and
# resolving a reference reports FACTORY_UNAVAILABLE instead of crashing.
_OPENAI_COMPATIBLE_FACTORIES = {
    _TRANSLATION: (
        "engine.providers.translation.openai_compatible:OpenAICompatibleTranslationProvider"
    ),
    _ASSISTANT: "engine.providers.assistant.openai_compatible:OpenAICompatibleAssistantProvider",
}

_GEMINI_FACTORIES = {
    _TRANSLATION: "engine.providers.translation.gemini:GeminiTranslationProvider",
    _ASSISTANT: "engine.providers.assistant.gemini:GeminiAssistantProvider",
}

_TEXT_API_OPTION_FIELDS = (_SOURCE_LANG, _TARGET_LANG, _TEMPERATURE, _MAX_OUTPUT_TOKENS)

# Azure selects the model through a per-resource deployment, so these stay explicit fields.
_AZURE_DEPLOYMENT = ManifestField(
    key="deployment",
    label="Deployment name",
    field_type=FieldType.STRING,
    required=True,
    description="Azure OpenAI deployment; sent as the model on the v1 surface.",
)

_AZURE_API_VERSION = ManifestField(
    key="api_version",
    label="API version",
    field_type=FieldType.STRING,
    default="v1",
    advanced=True,
    description="v1 or preview use the /openai/v1 surface; a dated version uses the "
    "deployment-scoped path.",
)


BUILTIN_MANIFESTS: Tuple[ProviderManifest, ...] = (
    ProviderManifest(
        provider_id="local",
        display_name="Local",
        capabilities=frozenset({_STT, _TRANSLATION}),
        protocol_family=ProtocolFamily.LOCAL,
        factories={
            _STT: "engine.providers.stt.local:LocalSTTProvider",
            _TRANSLATION: "engine.providers.translation.local:LocalTranslationProvider",
        },
        model_selectable=False,
        fields=(
            _LANGUAGE_HINT,
            ManifestField(
                key="device",
                label="Device",
                field_type=FieldType.CHOICE,
                default="auto",
                advanced=True,
                choices=("auto", "cpu", "cuda"),
                applies_to=frozenset({_STT}),
            ),
            ManifestField(
                key="compute_type",
                label="Compute type",
                field_type=FieldType.CHOICE,
                default="float16",
                advanced=True,
                choices=("float16", "int8_float16", "int8", "float32"),
                applies_to=frozenset({_STT}),
            ),
            _SOURCE_LANG,
            _TARGET_LANG,
        ),
    ),
    ProviderManifest(
        provider_id="openai-realtime",
        display_name="OpenAI Realtime",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.OPENAI_REALTIME,
        factories={_STT: "engine.providers.stt.openai_realtime:OpenAIRealtimeSTTProvider"},
        recommended_endpoint="wss://api.openai.com/v1/realtime",
        recommended_model="gpt-live-transcribe",
        fields=(_api_key(required=True), _LANGUAGE_HINT),
    ),
    ProviderManifest(
        provider_id="gemini-live",
        display_name="Gemini Live",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.GEMINI_LIVE,
        factories={_STT: "engine.providers.stt.gemini_live:GeminiLiveSTTProvider"},
        recommended_endpoint=(
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        ),
        recommended_model="gemini-3.5-transcribe-live",
        fields=(_api_key(required=True), _LANGUAGE_HINT),
    ),
    ProviderManifest(
        provider_id="openai",
        display_name="OpenAI",
        capabilities=frozenset({_TRANSLATION, _ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories=dict(_OPENAI_COMPATIBLE_FACTORIES),
        recommended_endpoint="https://api.openai.com/v1",
        recommended_model="gpt-5.6-terra",
        fields=(_api_key(required=True), *_TEXT_API_OPTION_FIELDS),
    ),
    ProviderManifest(
        provider_id="custom-openai-compatible",
        display_name="Custom (OpenAI-compatible)",
        capabilities=frozenset({_TRANSLATION, _ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories=dict(_OPENAI_COMPATIBLE_FACTORIES),
        requires_endpoint=True,
        # Self-hosted compatible servers often run without authentication.
        fields=(_api_key(required=False), *_TEXT_API_OPTION_FIELDS),
    ),
    ProviderManifest(
        provider_id="gemini",
        display_name="Gemini",
        capabilities=frozenset({_TRANSLATION, _ASSISTANT}),
        protocol_family=ProtocolFamily.GEMINI_TEXT,
        factories=dict(_GEMINI_FACTORIES),
        recommended_endpoint="https://generativelanguage.googleapis.com/v1beta",
        recommended_model="gemini-3.8-flash",
        fields=(_api_key(required=True), *_TEXT_API_OPTION_FIELDS),
    ),
    ProviderManifest(
        provider_id="azure-openai",
        display_name="Azure OpenAI",
        capabilities=frozenset({_TRANSLATION, _ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories=dict(_OPENAI_COMPATIBLE_FACTORIES),
        # The resource endpoint is unique per Azure resource, so nothing is recommended.
        requires_endpoint=True,
        recommended_model="gpt-5.6-terra",
        fields=(
            _api_key(required=True),
            _AZURE_DEPLOYMENT,
            _AZURE_API_VERSION,
            *_TEXT_API_OPTION_FIELDS,
        ),
    ),
    ProviderManifest(
        provider_id="assemblyai-llm-gateway",
        display_name="AssemblyAI LLM Gateway",
        capabilities=frozenset({_TRANSLATION, _ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories=dict(_OPENAI_COMPATIBLE_FACTORIES),
        recommended_endpoint="https://llm-gateway.assemblyai.com/v1",
        recommended_model="gpt-5.6-terra",
        fields=(_api_key(required=True), *_TEXT_API_OPTION_FIELDS),
    ),
    ProviderManifest(
        provider_id="local-openai-compatible",
        display_name="Local (OpenAI-compatible)",
        capabilities=frozenset({_ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories={_ASSISTANT: _OPENAI_COMPATIBLE_FACTORIES[_ASSISTANT]},
        # Ollama's default address; LM Studio, vLLM, and others are set by editing it.
        recommended_endpoint="http://localhost:11434/v1",
        requires_endpoint=True,
        fields=(_api_key(required=False), _TEMPERATURE, _MAX_OUTPUT_TOKENS),
    ),
)
