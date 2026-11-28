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

// An interval of the session timeline whose transcript may be missing or repeated
// (engine/session/transcript_ledger.py ContinuityWarning). Milliseconds on the session clock.
export interface ContinuityInterval {
  source: string;
  reason: string;
  startMs: number;
  endMs: number;
  epoch: number | null;
}

// Per stream (mic/system) STT provider state, derived from session transcript metadata.
// Fallback is sticky: once a stream falls back to Local it stays there for the session.
export interface StreamSttStatus {
  source: string;
  provider: string | null;
  fallbackFrom: string | null;
  continuity: ContinuityInterval[];
  continuityCount: number;
}

export interface SuggestedReply {
  strategy: string;
  text: string;
  reasoning?: string;
}

export interface ClarifyingQuestion {
  question: string;
  purpose?: string;
}

// provider: the Assistant provider failed. structured: the engine refused or could not run
// the request. interrupted: the request ended without its suggestion_result (bridge failure).
export type AssistErrorKind = "structured" | "provider" | "interrupted";

export interface AssistError {
  kind: AssistErrorKind;
  code: string | null;
  providerCode: string | null;
  // Fixed engine text or a UI constant; never provider free text.
  message: string;
  retryable: boolean;
}

export interface AssistOutcome {
  requestId: string;
  assistId: string | null;
  conversationId: string | null;
  suggestedReplies: SuggestedReply[];
  clarifyingQuestions: ClarifyingQuestion[];
  providerId: string | null;
  error: AssistError | null;
}

// One in-flight manual Assist request. Ephemeral, never persisted.
export interface AssistRequestState {
  requestId: string;
  conversationId: string;
  messageId: string;
}

export type AssistPhase = "hidden" | "idle" | "unavailable" | "loading" | "success" | "empty" | "error";

export type AssistUnavailableReason = "no_conversation" | "disconnected";

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
  // Session metadata (EF-016). Absent on legacy transcripts.
  itemId?: string;
  sttProvider?: string;
  sttFallbackFrom?: string;
  continuity?: ContinuityInterval;
  // Engine conversation record ids. Nothing sends them yet; Assist stays unavailable without.
  conversationId?: string;
  messageId?: string;
  // Translation authority: a final translation is never replaced by a partial one.
  translationFinal?: boolean;
  translationRevision?: number;
  translationProvider?: string;
  translationFallback?: boolean;
  translationError?: { code: string; message: string } | null;
  startMs?: number | null;
  endMs?: number | null;
  assist?: AssistOutcome;
}

export interface PartialTranscript {
  text: string;
  translation: string | null;
  source: TranscriptSource;
  itemId?: string;
  revision?: number;
  sttProvider?: string;
}

export interface AskTurn {
  id: string;
  question: string;
  answer: string;
  createdAtMs?: number;
}

export type OverlayFontSize = "sm" | "md" | "lg";

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

export type AppView = "transcript" | "settings" | "history" | "profiles";

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
  | "audio_level"
  | "devices_list"
  | "suggestion_result"
  | "llm_chunk"
  | "llm_done";

export type EngineCommandType =
  | "start"
  | "stop"
  | "list_devices"
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

// ---------------------------------------------------------------------------
// Context Profiles (EF-020 / EF-027)
// ---------------------------------------------------------------------------

export const MAX_PROFILE_NAME_CHARS = 120;
export const MAX_PROFILE_CONTENT_BYTES = 64 * 1024; // 64 KiB
export const MAX_PROFILES_PER_SESSION = 8;
export const MAX_SESSION_PROFILE_BYTES = 256 * 1024; // 256 KiB
export const MAX_TITLE_CHARS = 200;
export const MAX_LABEL_CHARS = 60;

export type ProfileContentFormat = "text" | "markdown";

export interface ContextProfile {
  profileId: string;
  name: string;
  content: string;
  contentFormat: ProfileContentFormat;
  revision: number;
  createdAtMs: number;
  updatedAtMs: number;
}

export interface ContextProfileInfo {
  profileId: string;
  name: string;
  contentFormat: ProfileContentFormat;
  revision: number;
  contentBytes: number;
  updatedAtMs: number;
}

export interface SessionProfileSnapshot {
  sessionId: string;
  position: number;
  profileId: string;
  name: string;
  content: string;
  contentFormat: ProfileContentFormat;
  profileRevision: number;
  contentSha256: string;
  capturedAtMs: number;
}

// ---------------------------------------------------------------------------
// Onboarding & Cloud Consent (EF-026)
// ---------------------------------------------------------------------------

export const CURRENT_CONSENT_VERSION = 1;
export const CLOUD_CONSENT_VERSION = 1;

export interface CloudConsentState {
  version: number;
  accepted: boolean;
  acceptedAt: number | null;
  revoked: boolean;
  revokedAt: number | null;
  stt: boolean;
  translation: boolean;
  assistant: boolean;
  summary: boolean;
}

export type PackReadinessState =
  | "unknown"
  | "checking"
  | "ready"
  | "downloading"
  | "paused"
  | "retry"
  | "corrupt"
  | "incompatible"
  | "offline";

export interface PackStatusItem {
  packId: string;
  kind: "stt" | "translation";
  displayName: string;
  version: string;
  sizeBytes: number;
  isReady: boolean;
  state: PackReadinessState;
  downloadPercent: number;
  downloadedBytes: number;
  totalBytes: number;
  issues: string[];
  tier: string;
  languageSupported?: boolean;
}

export interface HardwareInfo {
  cpuCores: number;
  ramGb: number;
  platform: string;
  architecture: string;
  hasGpu: boolean;
}

// ---------------------------------------------------------------------------
// Summary & SQLite-backed History (EF-029)
// ---------------------------------------------------------------------------

export interface SummaryWarning {
  reason: string;
  omitted_message_count: number;
  first_omitted_timestamp_ms?: number;
  last_omitted_timestamp_ms?: number;
  used_tokens?: number;
  max_tokens?: number;
  included_message_count?: number;
}

export interface StoredSummaryRecord {
  summary_id?: string;
  summary_markdown: string;
  created_at_ms: number;
  warning?: SummaryWarning | null;
}

export interface ConversationRecord {
  conversation_id: string;
  title: string;
  created_at_ms: number;
  updated_at_ms: number;
  source_labels?: { mic: string; system: string };
  import_id?: string | null;
  message_count?: number;
  has_summary?: boolean;
}
