import type { StateStorage } from "zustand/middleware";
import { errorCodeOf, forwardCommand, testProviderConnection } from "../lib/engineBridge";
import {
  deleteCredential,
  getCredentialStatus,
  LEGACY_CREDENTIALS,
  LEGACY_SECRET_FIELDS,
  migrateLegacySecret,
  scrubLegacyStorageSecrets,
  writeCredential,
  type CredentialLocator,
  type LegacyMigrationOutcome,
  type LegacySecretField,
} from "../lib/secureStore";
import type { EngineSliceCreator } from "./engineStore";
import type {
  AppView,
  CloudConsentState,
  EngineConfig,
  FieldType,
  HardwareInfo,
  ManifestField,
  PackStatusItem,
  ProviderCapability,
  ProviderManifest,
  ServiceConfig,
  ServiceName,
  ServiceOptionValue,
  ServicesConfig,
  Theme,
  ValidationMetadata,
} from "./types";
import { CLOUD_CONSENT_VERSION } from "./types";

// Public settings, credential references/configured state, and provider validation state.
// Raw secrets reach the OS vault through lib/secureStore and are never persisted.

export const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "en",
  device: "auto",
  translationEnabled: true,
  translationBackend: "online",
  translationModel: "",
  sourceLang: "en",
  targetLang: "vi",
  vadEnabled: true,
  audioSource: "microphone",
  micDeviceId: "",
  speakerDeviceId: "",
  llmEnabled: false,
  llmProviderUrl: "https://openrouter.ai/api/v1",
  llmApiKey: "",
  llmModel: "openai/gpt-4o-mini",
  stealthMode: false,
  hfToken: "",
};

// Legacy raw secret fields. Never persisted. Until Settings moves to write-only secret
// inputs they can hold a typed draft in memory; the secure runtime commits that draft to the
// vault (bootstrap, Start, leaving Settings) and clears it.
export const LEGACY_SECRET_CONFIG_KEYS = ["llmApiKey", "hfToken"] as const;

type LegacySecretConfigKey = (typeof LEGACY_SECRET_CONFIG_KEYS)[number];
export type PersistedConfigKey = Exclude<keyof EngineConfig, LegacySecretConfigKey>;
export type PersistedConfig = Pick<EngineConfig, PersistedConfigKey>;

// A Record rather than an array so adding an EngineConfig field fails to compile until it
// is classified as persisted here or as a secret above.
const PERSISTED_CONFIG_FLAGS: Record<PersistedConfigKey, true> = {
  host: true,
  port: true,
  modelSize: true,
  language: true,
  device: true,
  translationEnabled: true,
  translationBackend: true,
  translationModel: true,
  sourceLang: true,
  targetLang: true,
  vadEnabled: true,
  audioSource: true,
  micDeviceId: true,
  speakerDeviceId: true,
  llmEnabled: true,
  llmProviderUrl: true,
  llmModel: true,
  stealthMode: true,
};

export const PERSISTED_CONFIG_KEYS = Object.keys(PERSISTED_CONFIG_FLAGS) as PersistedConfigKey[];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Builds a fresh object from the whitelist, so unknown or secret-shaped keys never pass.
export function sanitizeConfig(raw: unknown): Partial<PersistedConfig> {
  if (!isRecord(raw)) return {};
  const out: Record<string, unknown> = {};
  for (const key of PERSISTED_CONFIG_KEYS) {
    const value = raw[key];
    if (value !== undefined && typeof value === typeof DEFAULT_CONFIG[key]) {
      out[key] = value;
    }
  }
  return out as Partial<PersistedConfig>;
}

// Full EngineConfig shape with every legacy secret blanked, for anything that leaves the
// store (the overlay sync event, the public start config).
export function publicConfigSnapshot(config: EngineConfig): EngineConfig {
  const out = { ...config };
  for (const key of LEGACY_SECRET_CONFIG_KEYS) out[key] = "";
  return out;
}

export function sanitizeTheme(raw: unknown): Theme | undefined {
  return raw === "dark" || raw === "light" ? raw : undefined;
}

export function preferredTheme(): Theme {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
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

// ---------------------------------------------------------------------------
// Provider catalog
//
// A hand-kept mirror of BUILTIN_MANIFESTS in engine/providers/manifests.py: no engine command
// delivers manifests to the UI yet. Any change there must be copied here. Order is display
// order within each capability.
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

// Optional, Advanced STT setting; without a default the provider's own default applies.
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

export function providersFor(capability: ProviderCapability): ProviderManifest[] {
  return PROVIDER_CATALOG.filter((m) => m.capabilities.includes(capability));
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

// A selectable model with no recommendation (the custom providers) has to be named.
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

// One vault entry per capability + provider + slot; the capability is part of the vault key.
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

  // Sanitize options: primitives only, absolutely ZERO secret keys or secret-like values
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

  // Sanitize credentialRefs: preserve only { configId, slot, version }, never secret data
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

  // Sanitize validation: outcome status and fingerprint only, never free text
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
  // undefined clears the option so the manifest or provider default applies.
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

// Keyed by "provider", "endpoint", "model", or a field key. Empty means the card is testable.
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
  // The running test follows a failed one.
  retry: boolean;
  // Fingerprinted inputs of the last completed test; a mismatch means its result is stale.
  testedInput: string | null;
}

export interface CredentialOpState {
  pending: boolean;
  errorCode: string | null;
}

function fingerprintInput(config: ServiceConfig): string {
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

// Fixed text per code: nothing a provider or the bridge returned is ever shown except the
// code itself, which has already passed ERROR_CODE_RE.
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

export function initialServiceUiState(): Pick<
  SettingsSlice,
  "connectionTests" | "credentialOps" | "optionDrafts"
> {
  return {
    connectionTests: { stt: idleTestUi(), translation: idleTestUi(), assistant: idleTestUi() },
    credentialOps: { stt: {}, translation: {}, assistant: {} },
    optionDrafts: { stt: {}, translation: {}, assistant: {} },
  };
}

export type ServiceConfigChanges = Partial<Omit<ServiceConfig, "validation">>;

// Same set as _FINGERPRINTED_CHANGES in the engine; `enabled` does not invalidate a test.
const FINGERPRINTED_FIELDS = [
  "providerId",
  "endpoint",
  "modelOverride",
  "options",
  "credentialRefs",
] as const;

function fingerprintedFieldsDiffer(a: ServiceConfig, b: ServiceConfig): boolean {
  return FINGERPRINTED_FIELDS.some((field) => JSON.stringify(a[field]) !== JSON.stringify(b[field]));
}

function withoutSecretOptions(
  options: Record<string, ServiceOptionValue>,
): Record<string, ServiceOptionValue> {
  const out: Record<string, ServiceOptionValue> = {};
  for (const [key, value] of Object.entries(options)) {
    if (!isSecretKeyName(key)) out[key] = value;
  }
  return out;
}

// Legacy flat fields that feed the assistant: editing any of them voids its last test.
const ASSISTANT_LEGACY_FIELDS = ["llmApiKey", "llmProviderUrl", "llmModel"] as const;

function invalidateService(services: ServicesConfig, service: ServiceName): ServicesConfig {
  return { ...services, [service]: { ...services[service], validation: unvalidated() } };
}

// Points services.assistant at the migrated legacy LLM key. A new version on every write is
// how a credential change reaches the validation fingerprint without the value itself.
function withLegacyAssistantRef(
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

export type SecretStatus = Record<LegacySecretField, boolean>;

export interface SettingsSlice {
  config: EngineConfig;
  theme: Theme;
  activeView: AppView;
  // v2 public service config. In memory only until the engine becomes its source of truth.
  services: ServicesConfig;
  // Whether the OS vault holds each legacy secret. The value itself never enters the store.
  secretStatus: SecretStatus;

  setActiveView: (view: AppView) => void;
  setTheme: (theme: Theme) => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
  updateServiceConfig: (service: ServiceName, changes: ServiceConfigChanges) => void;
  setServiceValidation: (service: ServiceName, validation: ValidationMetadata) => void;
  // Moves any raw legacy secret held in memory into the vault. Defaults to localStorage for
  // the legacy blob cleanup.
  commitLegacySecrets: (storage?: StateStorage) => Promise<LegacyMigrationOutcome[]>;
  refreshCredentialStatus: () => Promise<void>;
  testServiceConnection: (service: ServiceName) => Promise<ValidationMetadata | null>;

  // Settings-card UI state. Never persisted and never holds a secret value.
  connectionTests: Record<ServiceName, ConnectionTestUi>;
  // Keyed by credential slot.
  credentialOps: Record<ServiceName, Record<string, CredentialOpState>>;
  // Raw text of numeric entries that do not parse yet, keyed by option key.
  optionDrafts: Record<ServiceName, Record<string, string>>;

  // Switches provider and resets endpoint, model, options, and credential refs to the new
  // provider's defaults, then reads which of its secret slots the vault already holds.
  selectServiceProvider: (service: ServiceName, providerId: string) => void;
  setServiceOptionInput: (service: ServiceName, key: string, raw: FieldInput) => void;
  // The secret is an argument only: it goes to the vault and is never written to state.
  saveServiceSecret: (service: ServiceName, slot: string, secret: string) => Promise<boolean>;
  deleteServiceSecret: (service: ServiceName, slot: string) => Promise<boolean>;
  refreshServiceCredentials: (service: ServiceName) => Promise<void>;

  // EF-026 Onboarding & Cloud Consent
  onboardingCompleted: boolean;
  hasConsentedToCloud: boolean;
  cloudConsentVersion: number;
  consent: CloudConsentState;
  packStatus: Record<string, PackStatusItem>;
  hardwareInfo: HardwareInfo | null;
  isPackChecking: boolean;

  setOnboardingCompleted: (completed: boolean) => void;
  setHasConsentedToCloud: (consented: boolean) => void;
  acceptCloudConsent: (consents?: Partial<CloudConsentState>) => void;
  revokeCloudConsent: () => void;
  checkPackReadiness: (capability?: "stt" | "translation") => Promise<void>;
  installLocalPack: (packId: string) => Promise<boolean>;
  cancelLocalPack: (packId: string) => Promise<boolean>;
  retryLocalPack: (packId: string) => Promise<boolean>;
}

export const createSettingsSlice: EngineSliceCreator<SettingsSlice> = (set, get) => {
  // Per store instance: a second caller joins the running migration instead of racing it.
  let commitInFlight: Promise<LegacyMigrationOutcome[]> | null = null;

  const setTestUi = (service: ServiceName, patch: Partial<ConnectionTestUi>) => {
    set((state) => ({
      connectionTests: {
        ...state.connectionTests,
        [service]: { ...state.connectionTests[service], ...patch },
      },
    }));
  };

  const setCredentialOp = (service: ServiceName, slot: string, op: CredentialOpState | null) => {
    set((state) => {
      const ops = { ...state.credentialOps[service] };
      if (op) ops[slot] = op;
      else delete ops[slot];
      return { credentialOps: { ...state.credentialOps, [service]: ops } };
    });
  };

  // Only secret slots the current provider declares for this capability can be written.
  const secretFieldOf = (service: ServiceName, slot: string): ManifestField | undefined => {
    const provider = findProvider(service, get().services[service].providerId);
    return provider ? fieldsFor(provider, service).find((f) => f.secret && f.key === slot) : undefined;
  };

  // Goes through updateServiceConfig, so any ref change voids the last validation.
  const setCredentialRef = (
    service: ServiceName,
    slot: string,
    ref: ServiceConfig["credentialRefs"][string] | null,
  ) => {
    const credentialRefs = { ...get().services[service].credentialRefs };
    if (ref) credentialRefs[slot] = ref;
    else delete credentialRefs[slot];
    get().updateServiceConfig(service, { credentialRefs });
  };

  const runCommit = async (storage?: StateStorage): Promise<LegacyMigrationOutcome[]> => {
    const outcomes: LegacyMigrationOutcome[] = [];
    const migrated: LegacySecretField[] = [];
    for (const field of LEGACY_SECRET_FIELDS) {
      const raw = get().config[field];
      const outcome = await migrateLegacySecret(field, raw);
      outcomes.push(outcome);
      if (outcome.status !== "migrated") continue;
      migrated.push(field);
      set((state) => ({
        // A newer draft typed during the write is left for the next commit.
        config: state.config[field] === raw ? { ...state.config, [field]: "" } : state.config,
        secretStatus: { ...state.secretStatus, [field]: true },
        services:
          field === "llmApiKey"
            ? withLegacyAssistantRef(state.services, true, true)
            : state.services,
      }));
    }

    if (migrated.length > 0) {
      try {
        scrubLegacyStorageSecrets(storage ?? window.localStorage, migrated);
      } catch {
        // Storage unavailable: nothing left to scrub from it either.
      }
    }

    const failedCodes = outcomes
      .filter((outcome) => outcome.status === "failed")
      .map((outcome) => outcome.errorCode ?? "vault_failure");
    if (failedCodes.length > 0) {
      set({
        appError:
          "A saved secret could not be moved to the system vault and will be retried " +
          `(${failedCodes.join(", ")})`,
      });
    }
    return outcomes;
  };

  const initialOnboarding =
    typeof window !== "undefined" && window.localStorage?.getItem("echoflux:onboarding_completed") === "true";
  const initialConsent =
    typeof window !== "undefined" && window.localStorage?.getItem("echoflux:cloud_consent") === "true";

  const defaultConsentState: CloudConsentState = {
    version: CLOUD_CONSENT_VERSION,
    accepted: initialConsent,
    acceptedAt: initialConsent ? Date.now() : null,
    revoked: false,
    revokedAt: null,
    stt: initialConsent,
    translation: initialConsent,
    assistant: initialConsent,
    summary: initialConsent,
  };

  return {
    config: { ...DEFAULT_CONFIG },
    theme: preferredTheme(),
    activeView: "transcript",
    services: defaultServices(),
    secretStatus: { llmApiKey: false, hfToken: false },
    onboardingCompleted: initialOnboarding,
    hasConsentedToCloud: initialConsent,
    cloudConsentVersion: CLOUD_CONSENT_VERSION,
    consent: defaultConsentState,
    packStatus: {},
    hardwareInfo: null,
    isPackChecking: false,

    setActiveView: (view) => {
      const { activeView, transport, config } = get();
      set({ activeView: view });
      const leavingSettings = activeView === "settings" && view !== "settings";
      if (
        leavingSettings &&
        transport === "secure" &&
        LEGACY_SECRET_FIELDS.some((field) => config[field])
      ) {
        void get().commitLegacySecrets();
      }
    },

    setTheme: (theme) => {
      set({ theme });
    },

    updateConfig: (partial) => {
      set((state) => {
        const config = { ...state.config, ...partial };
        const assistantChanged = ASSISTANT_LEGACY_FIELDS.some(
          (key) => config[key] !== state.config[key],
        );
        return assistantChanged
          ? { config, services: invalidateService(state.services, "assistant") }
          : { config };
      });
    },

    updateServiceConfig: (service, changes) => {
      set((state) => {
        const current = state.services[service];
        const next: ServiceConfig = {
          ...current,
          ...changes,
          options: changes.options ? withoutSecretOptions(changes.options) : current.options,
          validation: current.validation,
        };
        if (fingerprintedFieldsDiffer(current, next)) next.validation = unvalidated();
        return { services: { ...state.services, [service]: next } };
      });
    },

    setServiceValidation: (service, validation) => {
      set((state) => ({
        services: { ...state.services, [service]: { ...state.services[service], validation } },
      }));
    },

    commitLegacySecrets: (storage) => {
      if (!commitInFlight) {
        commitInFlight = runCommit(storage).finally(() => {
          commitInFlight = null;
        });
      }
      return commitInFlight;
    },

    refreshCredentialStatus: async () => {
      for (const field of LEGACY_SECRET_FIELDS) {
        let configured: boolean;
        try {
          ({ configured } = await getCredentialStatus(LEGACY_CREDENTIALS[field]));
        } catch {
          continue; // Unknown is not "absent"; keep what the store already believes.
        }
        set((state) => ({
          secretStatus: { ...state.secretStatus, [field]: configured },
          services:
            field === "llmApiKey"
              ? withLegacyAssistantRef(state.services, configured, false)
              : state.services,
        }));
      }
    },

    testServiceConnection: async (service) => {
      if (get().connectionTests[service].inFlight) return null;
      const snapshot = get().services[service];
      const retry = deriveTestPhase(snapshot, get().connectionTests[service]) === "error";
      setTestUi(service, { inFlight: true, retry });
      let validation: ValidationMetadata;
      try {
        const result = await testProviderConnection(service, snapshot);
        validation = {
          status: result.status,
          fingerprint: result.fingerprint,
          checkedAt: result.testedAt,
          errorCode: result.errorCode,
        };
      } catch (error) {
        // A bridge failure says nothing about the provider, so it is not "invalid".
        validation = { ...unvalidated(), errorCode: errorCodeOf(error) };
      }
      // The config changed while the test ran; its result describes something else.
      if (fingerprintedFieldsDiffer(get().services[service], snapshot)) {
        setTestUi(service, { inFlight: false, retry: false });
        return null;
      }
      get().setServiceValidation(service, validation);
      setTestUi(service, { inFlight: false, retry: false, testedInput: fingerprintInput(snapshot) });
      return validation;
    },

    ...initialServiceUiState(),

    selectServiceProvider: (service, providerId) => {
      if (get().running || !findProvider(service, providerId)) return;
      if (get().services[service].providerId === providerId) return;
      get().updateServiceConfig(service, {
        providerId,
        endpoint: "",
        modelOverride: "",
        options: {},
        credentialRefs: {},
      });
      set((state) => ({
        optionDrafts: { ...state.optionDrafts, [service]: {} },
        credentialOps: { ...state.credentialOps, [service]: {} },
      }));
      void get().refreshServiceCredentials(service);
    },

    setServiceOptionInput: (service, key, raw) => {
      if (get().running) return;
      const current = get().services[service];
      const provider = findProvider(service, current.providerId);
      const field = provider
        ? fieldsFor(provider, service).find((f) => f.key === key && !f.secret)
        : undefined;
      if (!field) return;
      const { value, error } = parseFieldInput(field, raw);
      // An unparseable entry clears the option, so what gets tested is never a stale value
      // the input no longer shows.
      const options = { ...current.options };
      if (value === undefined) delete options[key];
      else options[key] = value;
      get().updateServiceConfig(service, { options });
      set((state) => {
        const drafts = { ...state.optionDrafts[service] };
        if (error) drafts[key] = String(raw);
        else delete drafts[key];
        return { optionDrafts: { ...state.optionDrafts, [service]: drafts } };
      });
    },

    saveServiceSecret: async (service, slot, secret) => {
      const providerId = get().services[service].providerId;
      if (get().running || !secretFieldOf(service, slot)) return false;
      if (get().credentialOps[service][slot]?.pending) return false;
      if (!secret) {
        setCredentialOp(service, slot, { pending: false, errorCode: "empty_secret" });
        return false;
      }
      const configId = credentialConfigId(providerId);
      if (!configId) {
        setCredentialOp(service, slot, { pending: false, errorCode: "invalid_config_id" });
        return false;
      }
      const locator: CredentialLocator = { capability: service, configId, slot };
      setCredentialOp(service, slot, { pending: true, errorCode: null });
      try {
        const { configured } = await writeCredential(locator, secret);
        if (!configured) {
          setCredentialOp(service, slot, { pending: false, errorCode: "vault_unconfirmed" });
          return false;
        }
      } catch (error) {
        setCredentialOp(service, slot, {
          pending: false,
          errorCode: errorCodeOf(error, "vault_failure"),
        });
        return false;
      }
      setCredentialOp(service, slot, null);
      // The provider changed while the vault call ran; the current refs name another entry.
      if (get().services[service].providerId !== providerId) return true;
      const previous = get().services[service].credentialRefs[slot];
      // A new version is how the change reaches the validation fingerprint.
      setCredentialRef(service, slot, { configId, slot, version: (previous?.version ?? 0) + 1 });
      return true;
    },

    deleteServiceSecret: async (service, slot) => {
      const snapshot = get().services[service];
      if (get().running || !secretFieldOf(service, slot)) return false;
      if (get().credentialOps[service][slot]?.pending) return false;
      // A ref may still name the migrated legacy entry rather than the provider's own.
      const configId = snapshot.credentialRefs[slot]?.configId ?? credentialConfigId(snapshot.providerId);
      if (!configId) {
        setCredentialOp(service, slot, { pending: false, errorCode: "invalid_config_id" });
        return false;
      }
      setCredentialOp(service, slot, { pending: true, errorCode: null });
      try {
        const { configured } = await deleteCredential({ capability: service, configId, slot });
        if (configured) {
          setCredentialOp(service, slot, { pending: false, errorCode: "vault_unconfirmed" });
          return false;
        }
      } catch (error) {
        setCredentialOp(service, slot, {
          pending: false,
          errorCode: errorCodeOf(error, "vault_failure"),
        });
        return false;
      }
      setCredentialOp(service, slot, null);
      const legacy = LEGACY_CREDENTIALS.llmApiKey;
      if (service === legacy.capability && configId === legacy.configId && slot === legacy.slot) {
        set((state) => ({ secretStatus: { ...state.secretStatus, llmApiKey: false } }));
      }
      const current = get().services[service];
      if (current.providerId === snapshot.providerId && current.credentialRefs[slot]?.configId === configId) {
        setCredentialRef(service, slot, null);
      }
      return true;
    },

    refreshServiceCredentials: async (service) => {
      const providerId = get().services[service].providerId;
      const provider = findProvider(service, providerId);
      if (!provider) return;
      for (const field of fieldsFor(provider, service)) {
        if (!field.secret) continue;
        const ref = get().services[service].credentialRefs[field.key];
        const configId = ref?.configId ?? credentialConfigId(providerId);
        if (!configId) continue;
        let configured: boolean;
        try {
          ({ configured } = await getCredentialStatus({ capability: service, configId, slot: field.key }));
        } catch {
          continue; // Unknown is not "absent"; keep what the store already believes.
        }
        const now = get().services[service];
        if (now.providerId !== providerId) return;
        const currentRef = now.credentialRefs[field.key];
        if (configured && !currentRef) {
          setCredentialRef(service, field.key, { configId, slot: field.key, version: 1 });
        } else if (!configured && currentRef?.configId === configId) {
          setCredentialRef(service, field.key, null);
        }
      }
    },

    setOnboardingCompleted: (completed: boolean) => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem("echoflux:onboarding_completed", String(completed));
        }
      } catch {}
      set({ onboardingCompleted: completed });
    },

    setHasConsentedToCloud: (consented: boolean) => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem("echoflux:cloud_consent", String(consented));
        }
      } catch {}
      set((state) => ({
        hasConsentedToCloud: consented,
        consent: {
          ...state.consent,
          accepted: consented,
          acceptedAt: consented ? Date.now() : state.consent.acceptedAt,
          revoked: !consented,
          revokedAt: !consented ? Date.now() : null,
        },
      }));
    },

    acceptCloudConsent: (consents?: Partial<CloudConsentState>) => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem("echoflux:cloud_consent", "true");
        }
      } catch {}
      set((state) => ({
        hasConsentedToCloud: true,
        cloudConsentVersion: CLOUD_CONSENT_VERSION,
        consent: {
          ...state.consent,
          ...consents,
          version: CLOUD_CONSENT_VERSION,
          accepted: true,
          acceptedAt: Date.now(),
          revoked: false,
          revokedAt: null,
        },
      }));
    },

    revokeCloudConsent: () => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem("echoflux:cloud_consent", "false");
        }
      } catch {}
      set((state) => ({
        hasConsentedToCloud: false,
        consent: {
          ...state.consent,
          accepted: false,
          revoked: true,
          revokedAt: Date.now(),
          stt: false,
          translation: false,
          assistant: false,
          summary: false,
        },
      }));
    },

    checkPackReadiness: async (capability?: "stt" | "translation") => {
      set({ isPackChecking: true });
      try {
        const caps: ("stt" | "translation")[] = capability ? [capability] : ["stt", "translation"];
        for (const cap of caps) {
          try {
            const res = await forwardCommand({ type: "pack_check_readiness", capability: cap });
            const statusMsg = res.messages.find(
              (m) => m.type === "pack_status" || (m as any).ready !== undefined || (m as any).packs,
            );
            if (statusMsg) {
              const data = ((statusMsg as any).data as any) || statusMsg;
              const packId = (data.pack_id as string) || (cap === "stt" ? "whisper-base" : "marian-en-vi");
              const ready = Boolean(data.ready);
              set((state) => ({
                packStatus: {
                  ...state.packStatus,
                  [cap]: {
                    packId,
                    kind: cap,
                    displayName: cap === "stt" ? "Whisper Base (Local STT)" : "Marian MT (Local Translation)",
                    version: (data.version as string) || "1.0",
                    sizeBytes: (data.size_bytes as number) || (cap === "stt" ? 145000000 : 250000000),
                    isReady: ready,
                    state: ready ? "ready" : "unknown",
                    downloadPercent: ready ? 100 : 0,
                    downloadedBytes: ready ? 145000000 : 0,
                    totalBytes: (data.size_bytes as number) || (cap === "stt" ? 145000000 : 250000000),
                    issues: Array.isArray(data.issues) ? data.issues : [],
                    tier: (data.tier as string) || "standard",
                  },
                },
              }));
            }
          } catch {
            // bridge unavailable or non-Tauri
          }
        }
        try {
          const statusRes = await forwardCommand({ type: "pack_status" });
          const first = statusRes.messages[0] as any;
          if (first?.hardware) {
            set({
              hardwareInfo: {
                cpuCores: first.hardware.cpu_cores || 8,
                ramGb: first.hardware.ram_gb || 16,
                platform: first.hardware.platform || "windows",
                architecture: first.hardware.architecture || "x86_64",
                hasGpu: Boolean(first.hardware.has_gpu),
              },
            });
          }
        } catch {}
      } finally {
        set({ isPackChecking: false });
      }
    },

    installLocalPack: async (packId: string) => {
      try {
        await forwardCommand({ type: "pack_install", pack_id: packId });
        return true;
      } catch {
        return false;
      }
    },

    cancelLocalPack: async (packId: string) => {
      try {
        await forwardCommand({ type: "pack_cancel", pack_id: packId });
        return true;
      } catch {
        return false;
      }
    },

    retryLocalPack: async (packId: string) => {
      try {
        await forwardCommand({ type: "pack_retry", pack_id: packId });
        return true;
      } catch {
        return false;
      }
    },
  };
};
