import { LEGACY_CREDENTIALS } from "../lib/secureStore";
import type {
  FieldType,
  ManifestField,
  ProviderCapability,
  ProviderManifest,
  ServiceConfig,
  ServiceName,
  ServiceOptionValue,
  ServicesConfig,
  ValidationMetadata,
} from "./types";

// Provider manifest catalog, config sanitization, field validation, and connection test
// derivation helpers. Kept separate from settingsSlice so Zustand store logic stays concise.

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Mirrors _SECRET_KEY_NAMES in engine/core/config.py (compared lowercased, "_"/"-" removed).
const SECRET_OPTION_KEY_NAMES = new Set([
  "apikey",
  "secret",
  "secretkey",
  "clientsecret",
  "token",
  "accesstoken",
  "authtoken",
  "password",
  "authorization",
  "bearer",
  "value",
]);

export function isSecretKeyName(key: string): boolean {
  return SECRET_OPTION_KEY_NAMES.has(key.toLowerCase().replace(/[_-]/g, ""));
}

export function unvalidated(): ValidationMetadata {
  return { status: "unvalidated", fingerprint: null, checkedAt: null, errorCode: null };
}

function defaultService(enabled: boolean, providerId: string): ServiceConfig {
  return {
    enabled,
    providerId,
    endpoint: "",
    modelOverride: "",
    options: {},
    credentialRefs: {},
    validation: unvalidated(),
  };
}

// Matches _DEFAULT_SERVICES in engine/core/config.py.
export function defaultServices(): ServicesConfig {
  return {
    stt: defaultService(true, "local"),
    translation: defaultService(true, "local"),
    assistant: defaultService(false, "openai"),
  };
}

export const SERVICE_NAMES: readonly ServiceName[] = ["stt", "translation", "assistant"];

export const PRESET_TARGET_LANGUAGES: readonly { code: string; label: string }[] = [
  { code: "vi", label: "Vietnamese (vi)" },
  { code: "en", label: "English (en)" },
  { code: "ja", label: "Japanese (ja)" },
  { code: "ko", label: "Korean (ko)" },
  { code: "zh", label: "Chinese (zh)" },
  { code: "fr", label: "French (fr)" },
  { code: "de", label: "German (de)" },
  { code: "es", label: "Spanish (es)" },
  { code: "th", label: "Thai (th)" },
  { code: "id", label: "Indonesian (id)" },
];

export const PRESET_INPUT_LANGUAGES: readonly { code: string; label: string }[] = [
  { code: "auto", label: "Auto-detect" },
  ...PRESET_TARGET_LANGUAGES,
];

// Mirrors _LEGACY_MARIAN_PAIRS in engine/local_models/catalog.py.
const LOCAL_MARIAN_PAIRS = new Set([
  "en->vi",
  "en->zh",
  "en->ja",
  "en->ko",
  "en->de",
  "en->fr",
  "en->es",
  "vi->en",
]);

export function isLocalMarianPairSupported(
  sourceLang: string | undefined | null,
  targetLang: string | undefined | null,
): boolean {
  const src = (sourceLang ?? "").trim().toLowerCase();
  const tgt = (targetLang ?? "").trim().toLowerCase();
  if (!tgt || tgt === "auto") return false;
  if (!src || src === "auto") {
    let matches = 0;
    for (const pair of LOCAL_MARIAN_PAIRS) {
      if (pair.endsWith(`->${tgt}`)) matches += 1;
    }
    return matches === 1;
  }
  return LOCAL_MARIAN_PAIRS.has(`${src}->${tgt}`);
}

export function resolveEffectiveTranslationSourceLang(
  services?: ServicesConfig | null,
  config?: { sourceLang?: string; language?: string } | null,
): string {
  const explicit = services?.translation?.options?.source_lang;
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim();
  const sttLang = services?.stt?.options?.language;
  if (typeof sttLang === "string" && sttLang.trim() && sttLang.trim().toLowerCase() !== "auto") {
    return sttLang.trim();
  }
  const cfgLang = config?.language;
  if (typeof cfgLang === "string" && cfgLang.trim() && cfgLang.trim().toLowerCase() !== "auto") {
    return cfgLang.trim();
  }
  return "auto";
}

// ---------------------------------------------------------------------------
// Provider catalog (fallback until get_provider_manifests loads from engine)
// ---------------------------------------------------------------------------

type FieldSpec = Partial<Omit<ManifestField, "key" | "label" | "fieldType">>;

function manifestField(
  key: string,
  label: string,
  fieldType: FieldType,
  spec: FieldSpec = {},
): ManifestField {
  return {
    key,
    label,
    fieldType,
    default: null,
    required: false,
    advanced: false,
    secret: fieldType === "secret",
    choices: [],
    minimum: null,
    maximum: null,
    description: "",
    appliesTo: [],
    ...spec,
  };
}

const VAULT_NOTE = "Stored in the OS credential vault; config keeps only a reference.";

const apiKeyField = (required: boolean) =>
  manifestField("api_key", "API key", "secret", { required, description: VAULT_NOTE });

const LANGUAGE_HINT = manifestField("language", "Language hint", "string", {
  default: "auto",
  advanced: true,
  appliesTo: ["stt"],
});

const SOURCE_LANG = manifestField("source_lang", "Source language", "string", {
  default: "auto",
  advanced: true,
  appliesTo: ["translation"],
});

const TARGET_LANG = manifestField("target_lang", "Target language", "string", {
  default: "vi",
  required: true,
  appliesTo: ["translation"],
});

const TEMPERATURE = manifestField("temperature", "Temperature", "number", {
  default: 0.3,
  advanced: true,
  minimum: 0,
  maximum: 2,
});

const MAX_OUTPUT_TOKENS = manifestField("max_output_tokens", "Max output tokens", "integer", {
  default: 1000,
  advanced: true,
  minimum: 1,
});

const TEXT_API_OPTION_FIELDS = [SOURCE_LANG, TARGET_LANG, TEMPERATURE, MAX_OUTPUT_TOKENS];

const AZURE_DEPLOYMENT = manifestField("deployment", "Deployment name", "string", {
  required: true,
  description: "Azure OpenAI deployment; sent as the model on the v1 surface.",
});

const AZURE_API_VERSION = manifestField("api_version", "API version", "string", {
  default: "v1",
  advanced: true,
  description:
    "v1 or preview use the /openai/v1 surface; a dated version uses the deployment-scoped path.",
});

const sttField = (key: string, label: string, fieldType: FieldType, spec: FieldSpec = {}) =>
  manifestField(key, label, fieldType, { advanced: true, ...spec, appliesTo: ["stt"] });

const OPENAI_REALTIME_FIELDS = [
  sttField("delay", "Latency/accuracy trade-off", "choice", {
    choices: ["minimal", "low", "medium", "high", "xhigh"],
  }),
  sttField("noise_reduction", "Noise reduction", "choice", {
    choices: ["near_field", "far_field"],
  }),
  sttField("prompt", "Transcription prompt", "string"),
  sttField("keywords", "Keywords", "string", {
    description: "Comma-separated terms to bias recognition toward.",
  }),
];

const GEMINI_LIVE_FIELDS = [
  sttField("transcription_mode", "Transcription mode", "choice", {
    choices: ["SMART", "VERBATIM"],
  }),
  sttField("custom_vocabulary", "Custom vocabulary", "string", {
    description: "Comma-separated terms, at most 1000.",
  }),
];

const DEEPGRAM_FIELDS = [
  sttField("language", "Language hint", "string", {
    default: "auto",
    description:
      "auto sends no language, so the model default applies; multi enables multilingual " +
      "code-switching.",
  }),
  sttField("auth_scheme", "Credential type", "choice", {
    default: "token",
    choices: ["token", "bearer"],
    description: "token for an API key, bearer for a temporary JWT.",
  }),
  sttField("endpointing_ms", "Endpointing silence (ms)", "integer", { minimum: 0 }),
  sttField("utterance_end_ms", "Utterance end (ms)", "integer", { minimum: 1000 }),
  sttField("smart_format", "Smart formatting", "boolean"),
  sttField("punctuate", "Punctuation", "boolean"),
  sttField("keyterms", "Key terms", "string", { description: "Comma-separated terms to boost." }),
];

const ASSEMBLYAI_FIELDS = [
  sttField("auth_mode", "Credential type", "choice", {
    default: "api_key",
    choices: ["api_key", "temporary_token"],
  }),
  sttField("format_turns", "Format turns", "boolean"),
  sttField("language_detection", "Language detection", "boolean", {
    description: "Report the detected language of each turn.",
  }),
  sttField("language_codes", "Expected languages", "string", {
    description: "Comma-separated language codes the session may contain.",
  }),
  sttField("keyterms_prompt", "Key terms", "string", {
    description: "Comma-separated terms, at most 100.",
  }),
  sttField("end_of_turn_confidence_threshold", "End-of-turn confidence", "number", {
    minimum: 0,
    maximum: 1,
  }),
  sttField("min_turn_silence_ms", "Minimum turn silence (ms)", "integer", {
    minimum: 50,
    maximum: 10000,
  }),
  sttField("max_turn_silence_ms", "Maximum turn silence (ms)", "integer", {
    minimum: 50,
    maximum: 60000,
  }),
  sttField("inactivity_timeout_s", "Inactivity timeout (s)", "integer", {
    minimum: 5,
    maximum: 3600,
  }),
];

const AZURE_SPEECH_FIELDS = [
  sttField("region", "Region", "string", {
    advanced: false,
    description: "Speech resource region, e.g. eastus. Required unless the API URL is set.",
  }),
  sttField("recognition_language", "Recognition language", "string", {
    description: "One full locale such as en-US. Set this or candidate languages.",
  }),
  sttField("candidate_languages", "Candidate languages", "string", {
    description:
      "Comma-separated full locales for language identification, one per base language: " +
      "at most 4 at start or 10 continuous.",
  }),
  sttField("language_id_mode", "Language identification mode", "choice", {
    default: "AtStart",
    choices: ["AtStart", "Continuous"],
    description: "Continuous identification requires the API URL to be set.",
  }),
  sttField("allow_custom_endpoint", "Allow custom host", "boolean", {
    default: false,
    description: "Accept an API URL that is neither a regional nor a Speech resource host.",
  }),
];

export const CUSTOM_STT_PROFILE_FIELD = "protocol_profile";
const CUSTOM_STT_HEADER_SLOTS = [1, 2];

const CUSTOM_STT_FIELDS = [
  manifestField(CUSTOM_STT_PROFILE_FIELD, "Protocol", "choice", {
    required: true,
    choices: ["openai-realtime", "gemini-live"],
    description: "Wire protocol the server implements.",
  }),
  apiKeyField(true),
  LANGUAGE_HINT,
  ...CUSTOM_STT_HEADER_SLOTS.flatMap((index) => [
    sttField(`header_${index}_name`, `Header ${index} name`, "string"),
    sttField(`header_${index}_value`, `Header ${index} value`, "string", {
      description: "Stored in plain config. Put credentials in a secret header.",
    }),
  ]),
  ...CUSTOM_STT_HEADER_SLOTS.flatMap((index) => [
    sttField(`secret_header_${index}_name`, `Secret header ${index} name`, "string"),
    sttField(`secret_header_${index}`, `Secret header ${index} value`, "secret", {
      description: VAULT_NOTE,
    }),
  ]),
];

type ManifestSpec = Pick<
  ProviderManifest,
  "providerId" | "displayName" | "capabilities" | "protocolFamily" | "fields"
> &
  Partial<ProviderManifest>;

function manifest(spec: ManifestSpec): ProviderManifest {
  return {
    recommendedEndpoint: "",
    recommendedModel: "",
    endpointEditable: true,
    requiresEndpoint: false,
    modelSelectable: true,
    ...spec,
  };
}

export const PROVIDER_CATALOG: readonly ProviderManifest[] = [
  manifest({
    providerId: "local",
    displayName: "Local",
    capabilities: ["stt", "translation"],
    protocolFamily: "local",
    modelSelectable: false,
    fields: [
      LANGUAGE_HINT,
      manifestField("quality_preset", "Speed / Accuracy", "choice", {
        default: "balanced",
        advanced: false,
        choices: ["fast", "balanced", "accurate"],
        description:
          "Fast uses tiny (~75 MB), balanced uses small (~500 MB), accurate uses medium (~1.5 GB).",
        appliesTo: ["stt"],
      }),
      manifestField("device", "Device", "choice", {
        default: "auto",
        advanced: true,
        choices: ["auto", "cpu", "cuda"],
        appliesTo: ["stt"],
      }),
      manifestField("compute_type", "Compute type", "choice", {
        default: "float16",
        advanced: true,
        choices: ["float16", "int8_float16", "int8", "float32"],
        appliesTo: ["stt"],
      }),
      SOURCE_LANG,
      TARGET_LANG,
    ],
  }),
  manifest({
    providerId: "openai-realtime",
    displayName: "OpenAI Realtime",
    capabilities: ["stt"],
    protocolFamily: "openai_realtime",
    recommendedEndpoint: "wss://api.openai.com/v1/realtime",
    recommendedModel: "gpt-live-transcribe",
    fields: [apiKeyField(true), LANGUAGE_HINT, ...OPENAI_REALTIME_FIELDS],
  }),
  manifest({
    providerId: "gemini-live",
    displayName: "Gemini Live",
    capabilities: ["stt"],
    protocolFamily: "gemini_live",
    recommendedEndpoint:
      "wss://generativelanguage.googleapis.com/ws/" +
      "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
    recommendedModel: "gemini-3.5-transcribe-live",
    fields: [apiKeyField(true), LANGUAGE_HINT, ...GEMINI_LIVE_FIELDS],
  }),
  manifest({
    providerId: "deepgram",
    displayName: "Deepgram",
    capabilities: ["stt"],
    protocolFamily: "deepgram",
    recommendedEndpoint: "wss://api.deepgram.com/v1/listen",
    recommendedModel: "nova-3",
    fields: [apiKeyField(true), ...DEEPGRAM_FIELDS],
  }),
  manifest({
    providerId: "azure-speech",
    displayName: "Azure Speech",
    capabilities: ["stt"],
    protocolFamily: "azure_speech",
    modelSelectable: false,
    fields: [apiKeyField(true), ...AZURE_SPEECH_FIELDS],
  }),
  manifest({
    providerId: "assemblyai",
    displayName: "AssemblyAI",
    capabilities: ["stt"],
    protocolFamily: "assemblyai_streaming",
    recommendedEndpoint: "wss://streaming.assemblyai.com/v3/ws",
    recommendedModel: "universal-3-5-pro",
    fields: [apiKeyField(true), ...ASSEMBLYAI_FIELDS],
  }),
  manifest({
    providerId: "custom",
    displayName: "Custom (realtime-compatible)",
    capabilities: ["stt"],
    protocolFamily: "openai_realtime",
    protocolProfiles: { "openai-realtime": "openai_realtime", "gemini-live": "gemini_live" },
    profileField: CUSTOM_STT_PROFILE_FIELD,
    requiresEndpoint: true,
    fields: CUSTOM_STT_FIELDS,
  }),
  manifest({
    providerId: "openai",
    displayName: "OpenAI",
    capabilities: ["translation", "assistant"],
    protocolFamily: "openai_compatible_text",
    recommendedEndpoint: "https://api.openai.com/v1",
    recommendedModel: "gpt-5.6-terra",
    fields: [apiKeyField(true), ...TEXT_API_OPTION_FIELDS],
  }),
  manifest({
    providerId: "gemini",
    displayName: "Gemini",
    capabilities: ["translation", "assistant"],
    protocolFamily: "gemini_text",
    recommendedEndpoint: "https://generativelanguage.googleapis.com/v1beta",
    recommendedModel: "gemini-3.8-flash",
    fields: [apiKeyField(true), ...TEXT_API_OPTION_FIELDS],
  }),
  manifest({
    providerId: "azure-openai",
    displayName: "Azure OpenAI",
    capabilities: ["translation", "assistant"],
    protocolFamily: "openai_compatible_text",
    requiresEndpoint: true,
    recommendedModel: "gpt-5.6-terra",
    fields: [apiKeyField(true), AZURE_DEPLOYMENT, AZURE_API_VERSION, ...TEXT_API_OPTION_FIELDS],
  }),
  manifest({
    providerId: "assemblyai-llm-gateway",
    displayName: "AssemblyAI LLM Gateway",
    capabilities: ["translation", "assistant"],
    protocolFamily: "openai_compatible_text",
    recommendedEndpoint: "https://llm-gateway.assemblyai.com/v1",
    recommendedModel: "gpt-5.6-terra",
    fields: [apiKeyField(true), ...TEXT_API_OPTION_FIELDS],
  }),
  manifest({
    providerId: "local-openai-compatible",
    displayName: "Local (OpenAI-compatible)",
    capabilities: ["assistant"],
    protocolFamily: "openai_compatible_text",
    recommendedEndpoint: "http://localhost:11434/v1",
    requiresEndpoint: true,
    fields: [apiKeyField(false), TEMPERATURE, MAX_OUTPUT_TOKENS],
  }),
  manifest({
    providerId: "custom-openai-compatible",
    displayName: "Custom (OpenAI-compatible)",
    capabilities: ["translation", "assistant"],
    protocolFamily: "openai_compatible_text",
    requiresEndpoint: true,
    fields: [apiKeyField(false), ...TEXT_API_OPTION_FIELDS],
  }),
];

let runtimeCatalog: readonly ProviderManifest[] = PROVIDER_CATALOG;

export function setRuntimeCatalog(catalog: readonly ProviderManifest[]): void {
  runtimeCatalog = catalog;
}

export function normalizeManifest(raw: any): ProviderManifest | null {
  if (!raw || typeof raw.provider_id !== "string") return null;
  const fields: ManifestField[] = Array.isArray(raw.fields)
    ? raw.fields.map((f: any) => ({
        key: String(f.key),
        label: String(f.label || f.key),
        fieldType: (f.field_type || "string") as FieldType,
        default: f.default ?? null,
        required: Boolean(f.required),
        advanced: Boolean(f.advanced),
        secret: Boolean(f.secret),
        choices: Array.isArray(f.choices) ? f.choices.map(String) : [],
        minimum: typeof f.minimum === "number" ? f.minimum : null,
        maximum: typeof f.maximum === "number" ? f.maximum : null,
        description: String(f.description || ""),
        appliesTo: Array.isArray(f.applies_to) ? f.applies_to : [],
      }))
    : [];
  return {
    providerId: raw.provider_id,
    displayName: String(raw.display_name || raw.provider_id),
    capabilities: Array.isArray(raw.capabilities) ? raw.capabilities : [],
    protocolFamily: raw.protocol_family,
    recommendedEndpoint: String(raw.recommended_endpoint || ""),
    recommendedModel: String(raw.recommended_model || ""),
    endpointEditable: raw.endpoint_editable !== false,
    requiresEndpoint: Boolean(raw.requires_endpoint),
    modelSelectable: raw.model_selectable !== false,
    fields,
    ...(raw.protocol_profiles ? { protocolProfiles: raw.protocol_profiles } : {}),
    ...(raw.profile_field ? { profileField: raw.profile_field } : {}),
  };
}

export function providersFor(capability: ProviderCapability): ProviderManifest[] {
  return runtimeCatalog.filter((m) => m.capabilities.includes(capability));
}

export function findProvider(
  capability: ProviderCapability,
  providerId: string,
): ProviderManifest | undefined {
  return providersFor(capability).find((m) => m.providerId === providerId);
}

export function fieldsFor(manifest: ProviderManifest, capability: ProviderCapability): ManifestField[] {
  return manifest.fields.filter((f) => f.appliesTo.length === 0 || f.appliesTo.includes(capability));
}

export function modelRequired(manifest: ProviderManifest): boolean {
  return manifest.modelSelectable && !manifest.recommendedModel;
}

// ---------------------------------------------------------------------------
// Credential ids (src-tauri/src/secure_store.rs is_valid_segment)
// ---------------------------------------------------------------------------

const CONFIG_ID_RE = /^[a-z0-9][a-z0-9_-]{0,47}$/;
const KEY_LIKE_PREFIXES = ["sk-", "sk_", "aiza", "gsk_", "xai-", "ghp_", "bearer"];

export function isValidConfigId(value: string): boolean {
  if (!CONFIG_ID_RE.test(value)) return false;
  if (KEY_LIKE_PREFIXES.some((prefix) => value.startsWith(prefix))) return false;
  return value.split(/[-_]/).every((run) => run.length <= 24);
}

export function credentialConfigId(providerId: string): string | null {
  return isValidConfigId(providerId) ? providerId : null;
}

// ---------------------------------------------------------------------------
// Zero-secrets v2 services sanitization (EF-025)
// ---------------------------------------------------------------------------

const SECRET_KEY_PATTERNS = ["sk-", "Bearer ", "token-", "key-", "hf_"];

export function sanitizeServiceConfig(raw: unknown, fallback: ServiceConfig): ServiceConfig {
  if (!isRecord(raw)) return fallback;

  const enabled = typeof raw.enabled === "boolean" ? raw.enabled : fallback.enabled;
  const providerId =
    typeof raw.providerId === "string" && isValidConfigId(raw.providerId)
      ? raw.providerId
      : fallback.providerId;
  const endpoint = typeof raw.endpoint === "string" ? raw.endpoint.trim() : "";
  const modelOverride = typeof raw.modelOverride === "string" ? raw.modelOverride.trim() : "";

  const options: Record<string, ServiceOptionValue> = {};
  if (isRecord(raw.options)) {
    for (const [key, value] of Object.entries(raw.options)) {
      if (isSecretKeyName(key)) continue;
      if (typeof value === "string") {
        if (SECRET_KEY_PATTERNS.some((p) => value.startsWith(p))) continue;
        if (KEY_LIKE_PREFIXES.some((p) => value.toLowerCase().startsWith(p))) continue;
        options[key] = value;
      } else if (typeof value === "number" || typeof value === "boolean" || value === null) {
        options[key] = value;
      }
    }
  }

  const credentialRefs: Record<string, { configId: string; slot: string; version: number }> = {};
  if (isRecord(raw.credentialRefs)) {
    for (const [slot, ref] of Object.entries(raw.credentialRefs)) {
      if (!isRecord(ref)) continue;
      const configId = ref.configId;
      const innerSlot = ref.slot;
      const version = ref.version;
      if (
        typeof configId === "string" &&
        isValidConfigId(configId) &&
        typeof innerSlot === "string" &&
        innerSlot === slot &&
        typeof version === "number" &&
        Number.isInteger(version) &&
        version >= 0
      ) {
        credentialRefs[slot] = { configId, slot, version };
      }
    }
  }

  let validation = unvalidated();
  if (isRecord(raw.validation)) {
    const status = raw.validation.status;
    if (status === "valid" || status === "invalid" || status === "unvalidated") {
      const fingerprint =
        typeof raw.validation.fingerprint === "string" && raw.validation.fingerprint.length <= 128
          ? raw.validation.fingerprint
          : null;
      const checkedAt =
        typeof raw.validation.checkedAt === "number" && raw.validation.checkedAt > 0
          ? raw.validation.checkedAt
          : null;
      const errorCode =
        typeof raw.validation.errorCode === "string" && /^[a-z][a-z0-9_]{0,63}$/.test(raw.validation.errorCode)
          ? raw.validation.errorCode
          : null;
      validation = { status, fingerprint, checkedAt, errorCode };
    }
  }

  return {
    enabled,
    providerId,
    endpoint,
    modelOverride,
    options,
    credentialRefs,
    validation,
  };
}

export function sanitizeServices(raw: unknown): ServicesConfig {
  const defaults = defaultServices();
  if (!isRecord(raw)) return defaults;
  return {
    stt: sanitizeServiceConfig(raw.stt, defaults.stt),
    translation: sanitizeServiceConfig(raw.translation, defaults.translation),
    assistant: sanitizeServiceConfig(raw.assistant, defaults.assistant),
  };
}

// ---------------------------------------------------------------------------
// Field input and required-field checks
// ---------------------------------------------------------------------------

export type FieldInput = string | boolean;

export interface ParsedFieldInput {
  value: ServiceOptionValue | undefined;
  error: string | null;
}

const INTEGER_RE = /^-?\d+$/;
const NUMBER_RE = /^-?(\d+\.?\d*|\.\d+)$/;

function rangeError(field: ManifestField, value: number): string | null {
  if (field.minimum !== null && value < field.minimum) return `Must be at least ${field.minimum}.`;
  if (field.maximum !== null && value > field.maximum) return `Must be at most ${field.maximum}.`;
  return null;
}

export function parseFieldInput(field: ManifestField, raw: FieldInput): ParsedFieldInput {
  if (field.secret) return { value: undefined, error: "Secret values are write-only." };
  if (field.fieldType === "boolean") {
    if (typeof raw === "boolean") return { value: raw, error: null };
    if (raw === "true" || raw === "false") return { value: raw === "true", error: null };
    return raw === "" ? { value: undefined, error: null } : { value: undefined, error: "Choose on or off." };
  }
  const text = typeof raw === "string" ? raw : String(raw);
  if (field.fieldType === "string") return { value: text === "" ? undefined : text, error: null };
  if (field.fieldType === "choice") {
    if (text === "") return { value: undefined, error: null };
    return field.choices.includes(text)
      ? { value: text, error: null }
      : { value: undefined, error: "Choose one of the listed values." };
  }
  const trimmed = text.trim();
  if (trimmed === "") return { value: undefined, error: null };
  const integer = field.fieldType === "integer";
  if (!(integer ? INTEGER_RE : NUMBER_RE).test(trimmed)) {
    return { value: undefined, error: integer ? "Enter a whole number." : "Enter a number." };
  }
  const value = Number(trimmed);
  const error = rangeError(field, value);
  return error ? { value: undefined, error } : { value, error: null };
}

export function serviceIssues(
  service: ServiceName,
  config: ServiceConfig,
  drafts: Record<string, string> = {},
): Record<string, string> {
  const issues: Record<string, string> = {};
  const provider = findProvider(service, config.providerId);
  if (!provider) {
    issues.provider = "Choose a provider.";
    return issues;
  }
  if (provider.requiresEndpoint && !config.endpoint.trim() && !provider.recommendedEndpoint) {
    issues.endpoint = "An API URL is required for this provider.";
  }
  if (modelRequired(provider) && !config.modelOverride.trim()) {
    issues.model = "A model is required for this provider.";
  }
  for (const field of fieldsFor(provider, service)) {
    if (field.secret) {
      if (field.required && !config.credentialRefs[field.key]) {
        issues[field.key] = "Required. Save a value to the system vault.";
      }
      continue;
    }
    const draft = drafts[field.key];
    if (draft !== undefined) {
      const { error } = parseFieldInput(field, draft);
      if (error) {
        issues[field.key] = error;
        continue;
      }
    }
    if (field.required && field.default === null && config.options[field.key] === undefined) {
      issues[field.key] = "Required.";
    }
  }
  return issues;
}

// ---------------------------------------------------------------------------
// Test Connection and credential UI state
// ---------------------------------------------------------------------------

export type TestConnectionPhase = "idle" | "loading" | "retry" | "success" | "stale" | "error";

export interface ConnectionTestUi {
  inFlight: boolean;
  retry: boolean;
  testedInput: string | null;
}

export interface CredentialOpState {
  pending: boolean;
  errorCode: string | null;
}

export const FINGERPRINTED_FIELDS = [
  "providerId",
  "endpoint",
  "modelOverride",
  "options",
  "credentialRefs",
] as const;

export function fingerprintInput(config: ServiceConfig): string {
  return JSON.stringify(FINGERPRINTED_FIELDS.map((field) => config[field]));
}

export function deriveTestPhase(config: ServiceConfig, ui: ConnectionTestUi): TestConnectionPhase {
  if (ui.inFlight) return ui.retry ? "retry" : "loading";
  const { validation } = config;
  if (validation.status === "valid") return "success";
  if (validation.status === "invalid" || validation.errorCode) return "error";
  if (ui.testedInput !== null && ui.testedInput !== fingerprintInput(config)) return "stale";
  return "idle";
}

const TEST_ERROR_MESSAGES: Record<string, string> = {
  auth_failed: "The provider rejected the saved credential.",
  rate_limited: "The provider is rate limiting requests. Try again shortly.",
  quota_exceeded: "The provider account has no remaining quota.",
  network_unavailable: "The provider could not be reached. Check the API URL and network.",
  timeout: "The provider did not answer in time.",
  connection_closed: "The provider closed the connection.",
  invalid_request: "The provider did not accept this configuration.",
  invalid_response: "The provider sent a response EchoFlux could not read.",
  model_unavailable: "The model is not available for this account.",
  unsupported_language: "The provider does not support this language setting.",
  internal_error: "The provider reported an internal error.",
  cancelled: "The connection test was cancelled.",
  missing_endpoint: "An API URL is required for this provider.",
  missing_model: "A model is required for this provider.",
  model_not_selectable: "This provider does not accept a model override.",
  missing_credential: "A required credential is not saved.",
  invalid_credential_ref: "A saved credential does not belong to this provider.",
  invalid_option: "A provider setting has an invalid value.",
  secret_in_options: "A secret was found in plain settings and was refused.",
  unknown_provider: "This provider is not known to the engine.",
  unsupported_capability: "This provider does not support this service.",
  factory_unavailable: "This provider is not available in this engine build yet.",
  validation_not_supported: "This provider cannot be tested yet.",
  engine_not_running: "The engine is not running.",
  bridge_failed: "Test Connection needs the desktop app, and the bridge did not respond.",
  vault_error: "A saved credential could not be read from the system vault.",
  vault_failure: "A saved credential could not be read from the system vault.",
};

export function describeTestError(code: string | null): string {
  if (!code) return "The provider did not accept the connection test.";
  return `${TEST_ERROR_MESSAGES[code] ?? "The connection test failed."} (${code})`;
}

const CREDENTIAL_ERROR_MESSAGES: Record<string, string> = {
  empty_secret: "Enter a value before saving.",
  invalid_config_id: "This provider id cannot be used as a vault entry name.",
  vault_unconfirmed: "The system vault did not confirm the saved value.",
  bridge_failed: "The system vault needs the desktop app, and the bridge did not respond.",
};

export function describeCredentialError(code: string): string {
  return `${CREDENTIAL_ERROR_MESSAGES[code] ?? "The system vault did not accept the change."} (${code})`;
}

function idleTestUi(): ConnectionTestUi {
  return { inFlight: false, retry: false, testedInput: null };
}

export function initialServiceUiState(): {
  connectionTests: Record<ServiceName, ConnectionTestUi>;
  credentialOps: Record<ServiceName, Record<string, CredentialOpState>>;
  optionDrafts: Record<ServiceName, Record<string, string>>;
} {
  return {
    connectionTests: { stt: idleTestUi(), translation: idleTestUi(), assistant: idleTestUi() },
    credentialOps: { stt: {}, translation: {}, assistant: {} },
    optionDrafts: { stt: {}, translation: {}, assistant: {} },
  };
}

export function fingerprintedFieldsDiffer(a: ServiceConfig, b: ServiceConfig): boolean {
  return FINGERPRINTED_FIELDS.some((field) => JSON.stringify(a[field]) !== JSON.stringify(b[field]));
}

export function withoutSecretOptions(
  options: Record<string, ServiceOptionValue>,
): Record<string, ServiceOptionValue> {
  const out: Record<string, ServiceOptionValue> = {};
  for (const [key, value] of Object.entries(options)) {
    if (!isSecretKeyName(key)) out[key] = value;
  }
  return out;
}

export function invalidateService(services: ServicesConfig, service: ServiceName): ServicesConfig {
  return { ...services, [service]: { ...services[service], validation: unvalidated() } };
}

export function withLegacyAssistantRef(
  services: ServicesConfig,
  configured: boolean,
  bumpVersion: boolean,
): ServicesConfig {
  const { configId, slot } = LEGACY_CREDENTIALS.llmApiKey;
  const assistant = services.assistant;
  const current = assistant.credentialRefs[slot];
  let credentialRefs: ServiceConfig["credentialRefs"];
  if (configured) {
    if (current && !bumpVersion) return services;
    credentialRefs = {
      ...assistant.credentialRefs,
      [slot]: { configId, slot, version: current ? current.version + 1 : 1 },
    };
  } else {
    if (!current) return services;
    credentialRefs = { ...assistant.credentialRefs };
    delete credentialRefs[slot];
  }
  return {
    ...services,
    assistant: { ...assistant, credentialRefs, validation: unvalidated() },
  };
}
