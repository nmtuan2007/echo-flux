// Public store types. Nothing in this file may describe a raw secret value: v2 config
// carries credential references only, mirroring engine/core/config.py.

// ---------------------------------------------------------------------------
// Transcript and conversation
// ---------------------------------------------------------------------------

export interface SuggestionOption {
  strategy: string;
  text: string;
}

export interface SuggestionResult {
  options?: SuggestionOption[];
  error?: string;
}

export type TranscriptSource = "mic" | "system" | "both" | null;

export interface TranscriptEntry {
  id: string;
  text: string;
  translation: string | null;
  isFinal: boolean;
  timestamp: number;
  source?: TranscriptSource;
  suggestions?: SuggestionResult; // persisted with the entry
  // Only set when the engine sends one; the legacy runtime never does.
  revision?: number;
}

export interface PartialTranscript {
  text: string;
  translation: string | null;
  source: TranscriptSource;
}

export interface Conversation {
  id: string;
  name: string;
  date: number;
  entries: TranscriptEntry[];
  summary?: string; // persisted meeting summary
}

// ---------------------------------------------------------------------------
// Devices and models (legacy runtime)
// ---------------------------------------------------------------------------

export interface AudioDeviceInfo {
  id: string;
  name: string;
}

export interface HubSearchResult {
  id: string;
  downloads: number;
  tags: string[];
  task: string;
}

export interface ModelItem {
  id: string;
  name: string;
  size_mb: number;
  runtime?: string;
  is_downloaded: boolean;
}

export type ModelType = "asr" | "translation";

// ---------------------------------------------------------------------------
// Legacy flat config (current runtime)
// ---------------------------------------------------------------------------

export interface EngineConfig {
  host: string;
  port: number;
  modelSize: string;
  language: string;
  device: string;
  translationEnabled: boolean;
  translationBackend: "online" | "marian";
  translationModel: string;
  sourceLang: string;
  targetLang: string;
  vadEnabled: boolean;
  // Dual audio capture
  audioSource: "microphone" | "system" | "both";
  micDeviceId: string;
  speakerDeviceId: string;
  // LLM / AI Assistant
  llmEnabled: boolean;
  llmProviderUrl: string;
  // Legacy raw secrets. Held in memory for the current runtime, never persisted.
  llmApiKey: string;
  llmModel: string;
  stealthMode: boolean;
  hfToken: string;
}

export type AppView = "transcript" | "settings" | "history" | "model_manager";

export type Theme = "dark" | "light";

// ---------------------------------------------------------------------------
// v2 provider/service config (engine/core/config.py, schema_version 2)
// ---------------------------------------------------------------------------

export const CONFIG_SCHEMA_VERSION = 2;

export type ServiceName = "stt" | "translation" | "assistant";

export type ProviderCapability = ServiceName;

export type StoredValidationStatus = "unvalidated" | "valid" | "invalid";

// "stale" is derived (fingerprint mismatch) and never stored.
export type ValidationState = StoredValidationStatus | "stale";

export interface CredentialRef {
  configId: string;
  slot: string;
  version: number;
}

export interface ValidationMetadata {
  status: StoredValidationStatus;
  fingerprint: string | null;
  checkedAt: number | null;
  // A ProviderErrorCode value; free text is never stored because it can echo secrets.
  errorCode: string | null;
}

export type ServiceOptionValue = string | number | boolean | null;

export interface ServiceConfig {
  enabled: boolean;
  providerId: string;
  endpoint: string;
  modelOverride: string;
  options: Record<string, ServiceOptionValue>;
  // Keyed by slot; each ref's `slot` equals its key.
  credentialRefs: Record<string, CredentialRef>;
  validation: ValidationMetadata;
}

export type ServicesConfig = Record<ServiceName, ServiceConfig>;

export interface PublicConfigV2 {
  schema_version: typeof CONFIG_SCHEMA_VERSION;
  services: ServicesConfig;
}

// ---------------------------------------------------------------------------
// Provider manifests (engine/providers/manifests.py)
// ---------------------------------------------------------------------------

export const PROTOCOL_FAMILIES = [
  "local",
  "openai_realtime",
  "gemini_live",
  "openai_compatible_text",
  "gemini_text",
  "deepgram",
  "assemblyai_streaming",
  "azure_speech",
] as const;

export type ProtocolFamily = (typeof PROTOCOL_FAMILIES)[number];

export type FieldType = "string" | "integer" | "number" | "boolean" | "choice" | "secret";

export interface ManifestField {
  key: string;
  label: string;
  fieldType: FieldType;
  default: ServiceOptionValue;
  required: boolean;
  advanced: boolean;
  // Secret fields describe a credential slot; their values never enter the store.
  secret: boolean;
  choices: string[];
  minimum: number | null;
  maximum: number | null;
  description: string;
  // Empty means every capability of the provider.
  appliesTo: ProviderCapability[];
}

export interface ProviderManifest {
  providerId: string;
  displayName: string;
  capabilities: ProviderCapability[];
  protocolFamily: ProtocolFamily;
  recommendedEndpoint: string;
  recommendedModel: string;
  endpointEditable: boolean;
  requiresEndpoint: boolean;
  modelSelectable: boolean;
  fields: ManifestField[];
  // Bounded protocol profiles (Custom STT): the option named by `profileField` picks the
  // family, and `protocolFamily` is then the first-listed profile's family.
  protocolProfiles?: Record<string, ProtocolFamily>;
  profileField?: string;
}

// ---------------------------------------------------------------------------
// Legacy application messages (.agent/rules/03-architecture.md)
// ---------------------------------------------------------------------------

export type EngineEventType =
  | "partial"
  | "final"
  | "translation_update"
  | "status"
  | "error"
  | "devices_list"
  | "models_list"
  | "download_progress"
  | "model_action_result"
  | "hub_search_results"
  | "suggestion_result"
  | "llm_chunk"
  | "llm_done";

export type EngineCommandType =
  | "start"
  | "stop"
  | "list_devices"
  | "request_models_list"
  | "download_model"
  | "delete_model"
  | "search_hub"
  | "request_suggestion"
  | "request_summary";

// Inbound messages are untrusted JSON; handlers read fields defensively.
export interface EngineMessage {
  type: EngineEventType | string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Authenticated transport frames (engine/server/protocol.py)
//
// Typed only. The runtime still speaks the legacy unauthenticated socket; switching
// over is a later task.
// ---------------------------------------------------------------------------

export const TRANSPORT_PROTOCOL_VERSION = 1;

export type TransportRole = "control" | "event";

export interface AuthFrame {
  op: "auth";
  role: TransportRole;
  token: string;
  protocol: typeof TRANSPORT_PROTOCOL_VERSION;
}

export interface AuthOkFrame {
  op: "auth_ok";
  role: TransportRole;
  protocol: typeof TRANSPORT_PROTOCOL_VERSION;
}

export interface ControlRequestFrame {
  op: "request";
  // Must match [A-Za-z0-9._:-]{1,128}.
  id: string;
  command: { type: EngineCommandType; [key: string]: unknown };
}

export interface ControlResponseFrame {
  op: "response";
  id: string;
  message: EngineMessage;
}

export interface ControlCompleteFrame {
  op: "complete";
  id: string;
}

export interface EventFrame {
  op: "event";
  event: EngineMessage;
}

export interface TransportErrorFrame {
  op: "error";
  code: string;
  message: string;
  id?: string;
}

export type ClientTransportFrame = AuthFrame | ControlRequestFrame;

export type ServerTransportFrame =
  | AuthOkFrame
  | ControlResponseFrame
  | ControlCompleteFrame
  | EventFrame
  | TransportErrorFrame;
