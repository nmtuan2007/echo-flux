"""Built-in provider manifests and their dynamic field definitions.

A manifest is static metadata: which capabilities a provider serves, which wire protocol it
speaks, what endpoint and model it recommends, and which typed fields the settings UI may
render. The adapter itself is referenced by a ``"module:attribute"`` string so nothing here
imports network clients or ML runtimes; ``engine.providers.registry`` resolves the reference
only when a session actually needs the adapter.

Secret fields describe credential slots. Their values live in the OS vault and reach config
only as ``credentialRefs``, never as options.

All three matrices are complete. Custom STT is a single ``custom`` provider whose
``protocol_profile`` option selects one of two realtime protocols (D-011); the registry
resolves the profile to its protocol family, and nothing else can be plugged in.

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
    # Local runtimes use curated model packs, never a free-form model name (D-032). Providers
    # without a model concept (Azure Speech) also turn this off and resolve to an empty model.
    model_selectable: bool = True
    fields: Tuple[ManifestField, ...] = field(default_factory=tuple)
    # Bounded protocol profiles: the value of ``profile_field`` picks the family the resolved
    # service speaks. ``protocol_family`` is then the family of the first-listed profile.
    protocol_profiles: Dict[str, ProtocolFamily] = field(default_factory=dict)
    profile_field: str = ""

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    def protocol_family_for(self, options: Dict[str, Any]) -> ProtocolFamily:
        """Family for resolved options; unprofiled manifests always use ``protocol_family``."""
        if not self.protocol_profiles:
            return self.protocol_family
        return self.protocol_profiles[options[self.profile_field]]

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


def _stt_field(key: str, label: str, field_type: FieldType, **values: Any) -> ManifestField:
    """Optional, Advanced STT setting. Without a default nothing is sent, so the provider's
    own default applies."""
    values.setdefault("advanced", True)
    return ManifestField(
        key=key, label=label, field_type=field_type, applies_to=frozenset({_STT}), **values
    )


# ── STT provider-specific fields ─────────────────────────────────────────────

_OPENAI_REALTIME_FIELDS = (
    _stt_field(
        "delay",
        "Latency/accuracy trade-off",
        FieldType.CHOICE,
        choices=("minimal", "low", "medium", "high", "xhigh"),
    ),
    _stt_field(
        "noise_reduction",
        "Noise reduction",
        FieldType.CHOICE,
        choices=("near_field", "far_field"),
    ),
    _stt_field("prompt", "Transcription prompt", FieldType.STRING),
    _stt_field(
        "keywords",
        "Keywords",
        FieldType.STRING,
        description="Comma-separated terms to bias recognition toward.",
    ),
)

_GEMINI_LIVE_FIELDS = (
    _stt_field(
        "transcription_mode",
        "Transcription mode",
        FieldType.CHOICE,
        choices=("SMART", "VERBATIM"),
    ),
    _stt_field(
        "custom_vocabulary",
        "Custom vocabulary",
        FieldType.STRING,
        description="Comma-separated terms, at most 1000.",
    ),
)

_DEEPGRAM_FIELDS = (
    _stt_field(
        "language",
        "Language hint",
        FieldType.STRING,
        default="auto",
        description="auto sends no language, so the model default applies; "
        "multi enables multilingual code-switching.",
    ),
    _stt_field(
        "auth_scheme",
        "Credential type",
        FieldType.CHOICE,
        default="token",
        choices=("token", "bearer"),
        description="token for an API key, bearer for a temporary JWT.",
    ),
    _stt_field("endpointing_ms", "Endpointing silence (ms)", FieldType.INTEGER, minimum=0),
    _stt_field("utterance_end_ms", "Utterance end (ms)", FieldType.INTEGER, minimum=1000),
    _stt_field("smart_format", "Smart formatting", FieldType.BOOLEAN),
    _stt_field("punctuate", "Punctuation", FieldType.BOOLEAN),
    _stt_field(
        "keyterms",
        "Key terms",
        FieldType.STRING,
        description="Comma-separated terms to boost.",
    ),
)

_ASSEMBLYAI_FIELDS = (
    _stt_field(
        "auth_mode",
        "Credential type",
        FieldType.CHOICE,
        default="api_key",
        choices=("api_key", "temporary_token"),
    ),
    _stt_field("format_turns", "Format turns", FieldType.BOOLEAN),
    _stt_field(
        "language_detection",
        "Language detection",
        FieldType.BOOLEAN,
        description="Report the detected language of each turn.",
    ),
    _stt_field(
        "language_codes",
        "Expected languages",
        FieldType.STRING,
        description="Comma-separated language codes the session may contain.",
    ),
    _stt_field(
        "keyterms_prompt",
        "Key terms",
        FieldType.STRING,
        description="Comma-separated terms, at most 100.",
    ),
    _stt_field(
        "end_of_turn_confidence_threshold",
        "End-of-turn confidence",
        FieldType.NUMBER,
        minimum=0.0,
        maximum=1.0,
    ),
    _stt_field(
        "min_turn_silence_ms",
        "Minimum turn silence (ms)",
        FieldType.INTEGER,
        minimum=50,
        maximum=10000,
    ),
    _stt_field(
        "max_turn_silence_ms",
        "Maximum turn silence (ms)",
        FieldType.INTEGER,
        minimum=50,
        maximum=60000,
    ),
    _stt_field(
        "inactivity_timeout_s", "Inactivity timeout (s)", FieldType.INTEGER, minimum=5, maximum=3600
    ),
)

# Azure Speech cannot auto-detect from an open set: identification needs explicit candidate
# locales. So there is no ``language`` field pretending to default to auto (D-013).
_AZURE_SPEECH_FIELDS = (
    _stt_field(
        "region",
        "Region",
        FieldType.STRING,
        advanced=False,
        description="Speech resource region, e.g. eastus. Required unless the API URL is set.",
    ),
    _stt_field(
        "recognition_language",
        "Recognition language",
        FieldType.STRING,
        description="One full locale such as en-US. Set this or candidate languages.",
    ),
    _stt_field(
        "candidate_languages",
        "Candidate languages",
        FieldType.STRING,
        description="Comma-separated full locales for language identification, one per base "
        "language: at most 4 at start or 10 continuous.",
    ),
    _stt_field(
        "language_id_mode",
        "Language identification mode",
        FieldType.CHOICE,
        default="AtStart",
        choices=("AtStart", "Continuous"),
        description="Continuous identification requires the API URL to be set.",
    ),
    _stt_field(
        "allow_custom_endpoint",
        "Allow custom host",
        FieldType.BOOLEAN,
        default=False,
        description="Accept an API URL that is neither a regional nor a Speech resource host.",
    ),
)

# ── Custom STT: bounded protocol profiles with separated headers ─────────────

CUSTOM_STT_PROFILES: Dict[str, ProtocolFamily] = {
    "openai-realtime": ProtocolFamily.OPENAI_REALTIME,
    "gemini-live": ProtocolFamily.GEMINI_LIVE,
}
CUSTOM_STT_PROFILE_FIELD = "protocol_profile"
CUSTOM_STT_PUBLIC_HEADER_SLOTS = 2
CUSTOM_STT_SECRET_HEADER_SLOTS = 2


def _custom_header_fields() -> Tuple[ManifestField, ...]:
    fields = []
    for index in range(1, CUSTOM_STT_PUBLIC_HEADER_SLOTS + 1):
        fields.append(_stt_field(f"header_{index}_name", f"Header {index} name", FieldType.STRING))
        fields.append(
            _stt_field(
                f"header_{index}_value",
                f"Header {index} value",
                FieldType.STRING,
                description="Stored in plain config. Put credentials in a secret header.",
            )
        )
    for index in range(1, CUSTOM_STT_SECRET_HEADER_SLOTS + 1):
        fields.append(
            _stt_field(
                f"secret_header_{index}_name", f"Secret header {index} name", FieldType.STRING
            )
        )
        fields.append(
            ManifestField(
                key=f"secret_header_{index}",
                label=f"Secret header {index} value",
                field_type=FieldType.SECRET,
                secret=True,
                advanced=True,
                applies_to=frozenset({_STT}),
                description="Stored in the OS credential vault; config keeps only a reference.",
            )
        )
    return tuple(fields)


_CUSTOM_STT_FIELDS = (
    ManifestField(
        key=CUSTOM_STT_PROFILE_FIELD,
        label="Protocol",
        field_type=FieldType.CHOICE,
        required=True,
        choices=tuple(CUSTOM_STT_PROFILES),
        description="Wire protocol the server implements.",
    ),
    # Both profile adapters authenticate every session, so the key is required.
    _api_key(required=True),
    _LANGUAGE_HINT,
    *_custom_header_fields(),
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
                key="quality_preset",
                label="Speed / Accuracy",
                field_type=FieldType.CHOICE,
                default="balanced",
                advanced=True,
                choices=("fast", "balanced", "accurate"),
                description=(
                    "Fast uses tiny (~75 MB), balanced uses small (~500 MB), "
                    "accurate uses medium (~1.5 GB)."
                ),
                applies_to=frozenset({_STT}),
            ),
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
        # The adapter appends ``intent=transcription`` unless the edited URL already sets it.
        recommended_endpoint="wss://api.openai.com/v1/realtime",
        recommended_model="gpt-live-transcribe",
        fields=(_api_key(required=True), _LANGUAGE_HINT, *_OPENAI_REALTIME_FIELDS),
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
        fields=(_api_key(required=True), _LANGUAGE_HINT, *_GEMINI_LIVE_FIELDS),
    ),
    ProviderManifest(
        provider_id="deepgram",
        display_name="Deepgram",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.DEEPGRAM,
        factories={_STT: "engine.providers.stt.deepgram:DeepgramSTTProvider"},
        recommended_endpoint="wss://api.deepgram.com/v1/listen",
        recommended_model="nova-3",
        fields=(_api_key(required=True), *_DEEPGRAM_FIELDS),
    ),
    ProviderManifest(
        provider_id="azure-speech",
        display_name="Azure Speech",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.AZURE_SPEECH,
        # Optional SDK (pyproject extra ``azure-speech``), imported only when a session starts.
        factories={_STT: "engine.providers.stt.azure_speech:AzureSpeechSTTProvider"},
        # Endpoints are per region or resource, and the SDK has no model parameter.
        model_selectable=False,
        fields=(_api_key(required=True), *_AZURE_SPEECH_FIELDS),
    ),
    ProviderManifest(
        provider_id="assemblyai",
        display_name="AssemblyAI",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.ASSEMBLYAI_STREAMING,
        factories={_STT: "engine.providers.stt.assemblyai:AssemblyAISTTProvider"},
        recommended_endpoint="wss://streaming.assemblyai.com/v3/ws",
        recommended_model="universal-3-5-pro",
        fields=(_api_key(required=True), *_ASSEMBLYAI_FIELDS),
    ),
    ProviderManifest(
        provider_id="custom",
        display_name="Custom (realtime-compatible)",
        capabilities=frozenset({_STT}),
        protocol_family=ProtocolFamily.OPENAI_REALTIME,
        protocol_profiles=dict(CUSTOM_STT_PROFILES),
        profile_field=CUSTOM_STT_PROFILE_FIELD,
        factories={_STT: "engine.providers.stt.custom:CustomSTTProvider"},
        requires_endpoint=True,
        fields=_CUSTOM_STT_FIELDS,
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
