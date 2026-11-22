import { describeCommandError, errorCodeOf, forwardCommand, isTauriRuntime } from "../lib/engineBridge";
import type { EngineSliceCreator, EngineState } from "./engineStore";
import type { EngineCommands } from "./sessionSlice";
import { findProvider, publicConfigSnapshot } from "./settingsSlice";
import type {
  AssistError,
  AssistOutcome,
  AssistPhase,
  AssistRequestState,
  AssistUnavailableReason,
  ClarifyingQuestion,
  ContinuityInterval,
  EngineMessage,
  PartialTranscript,
  StreamSttStatus,
  SuggestedReply,
  SuggestionOption,
  SuggestionResult,
  TranscriptEntry,
  TranscriptSource,
} from "./types";

// Transcript, async translation, STT provider status, and manual Assist state.
//
// Session transcript keys (EF-016) are optional and untrusted, and absent on the legacy
// runtime, so every read below is type-checked.

let idCounter = 0;

function nextId(): string {
  idCounter += 1;
  return `t-${Date.now()}-${idCounter}`;
}

export const LOCAL_STT_PROVIDER_ID = "local";

// Display order for concurrent partials. Fixed so a stream that finalizes and restarts does
// not jump above the other one.
export const STREAM_ORDER: readonly string[] = ["mic", "system", "both", "default"];

const MAX_FINALIZED_ITEMS = 512;
const MAX_STREAM_CONTINUITY = 20;

// Same shape as engine conversation record ids (engine/server/protocol.py).
const RECORD_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordId(value: unknown): string | undefined {
  return typeof value === "string" && RECORD_ID_RE.test(value) ? value : undefined;
}

function revisionOf(message: EngineMessage): number | undefined {
  return typeof message.revision === "number" ? message.revision : undefined;
}

// ES2021-compatible reverse linear search helper
function findLast(arr: TranscriptEntry[], pred: (e: TranscriptEntry) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i;
  }
  return -1;
}

interface TranscriptMeta {
  itemId?: string;
  revision?: number;
  sttProvider?: string;
  retracted: boolean;
  continuity?: ContinuityInterval;
  conversationId?: string;
  messageId?: string;
}

function continuityOf(message: EngineMessage, source: string): ContinuityInterval | undefined {
  const raw = message.continuity;
  if (!isRecord(raw)) return undefined;
  const startMs = numberOrNull(raw.start_ms);
  const endMs = numberOrNull(raw.end_ms);
  if (startMs === null || endMs === null) return undefined;
  return {
    source,
    reason: typeof raw.reason === "string" ? raw.reason : "unknown",
    startMs,
    endMs,
    epoch: numberOrNull(raw.epoch),
  };
}

function readTranscriptMeta(message: EngineMessage, source: string): TranscriptMeta {
  return {
    itemId: nonEmptyString(message.item_id),
    revision: revisionOf(message),
    sttProvider: nonEmptyString(message.stt_provider),
    retracted: message.retracted === true,
    continuity: continuityOf(message, source),
    conversationId: recordId(message.conversation_id),
    messageId: recordId(message.message_id),
  };
}

function continuityKey(interval: ContinuityInterval): string {
  return `${interval.source}:${interval.epoch ?? ""}:${interval.startMs}:${interval.endMs}`;
}

// Folds one transcript event's provider and continuity metadata into the stream status.
// Returns null when nothing changed, so high-frequency partials do not rewrite this map.
function nextStreamStatus(
  streams: Record<string, StreamSttStatus>,
  source: string,
  meta: TranscriptMeta,
): Record<string, StreamSttStatus> | null {
  // A retraction closes an item of a connection that may already be gone; its provider is
  // not the stream's current one.
  const provider = meta.retracted ? undefined : meta.sttProvider;
  if (!provider && !meta.continuity) return null;

  const prev: StreamSttStatus = streams[source] ?? {
    source,
    provider: null,
    fallbackFrom: null,
    continuity: [],
    continuityCount: 0,
  };
  let next = prev;

  // Fallback lasts for the rest of the session, so nothing may move a stream off Local.
  if (provider && provider !== prev.provider && prev.fallbackFrom === null) {
    const fellBack =
      prev.provider !== null &&
      prev.provider !== LOCAL_STT_PROVIDER_ID &&
      provider === LOCAL_STT_PROVIDER_ID;
    next = { ...next, provider, fallbackFrom: fellBack ? prev.provider : null };
  }

  if (meta.continuity) {
    const key = continuityKey(meta.continuity);
    if (!prev.continuity.some((c) => continuityKey(c) === key)) {
      next = {
        ...next,
        continuity: [...prev.continuity, meta.continuity].slice(-MAX_STREAM_CONTINUITY),
        continuityCount: prev.continuityCount + 1,
      };
    }
  }

  return next === prev ? null : { ...streams, [source]: next };
}

// A partial whose item already finalized, or an event for a different item than the one
// currently shown, must not touch the visible partial.
function partialMatchesItem(partial: PartialTranscript | undefined, itemId?: string): boolean {
  return !partial || !itemId || !partial.itemId || partial.itemId === itemId;
}

function withFinalizedItem(
  finalized: Record<string, number>,
  itemId: string,
  revision: number,
): Record<string, number> {
  const next = { ...finalized, [itemId]: revision };
  const keys = Object.keys(next);
  for (let i = 0; i < keys.length - MAX_FINALIZED_ITEMS; i++) delete next[keys[i]];
  return next;
}

export function transcriptResetState(): Pick<
  TranscriptSlice,
  | "entries"
  | "partials"
  | "suggestionLoading"
  | "assistRequests"
  | "sttStreams"
  | "finalizedItems"
  | "engineConversationId"
> {
  return {
    entries: [],
    partials: {},
    suggestionLoading: {},
    assistRequests: {},
    sttStreams: {},
    finalizedItems: {},
    engineConversationId: null,
  };
}

// Returns null when the update is stale or rejected, so no state write occurs.
export function applyPartial(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> | null {
  const newPartial = (message.text as string) || "";
  const streamId = (message.source as string) || "default";
  const meta = readTranscriptMeta(message, streamId);
  const current = state.partials[streamId];

  if (meta.itemId !== undefined) {
    // Late partial for an item that already has its final text.
    if (meta.itemId in state.finalizedItems) return null;
    if (
      current?.itemId === meta.itemId &&
      meta.revision !== undefined &&
      current.revision !== undefined &&
      meta.revision <= current.revision
    ) {
      return null;
    }
  }

  const revisioned = meta.itemId !== undefined && meta.revision !== undefined;
  const currentText = current?.text || "";
  // Legacy partials carry no revision, so a sudden large shrink is treated as flicker.
  if (
    !revisioned &&
    !(
      newPartial.length >= currentText.length ||
      newPartial.length >= currentText.length * 0.6 ||
      currentText === ""
    )
  ) {
    return null;
  }

  const update: Partial<EngineState> = {
    partials: {
      ...state.partials,
      [streamId]: {
        text: newPartial,
        translation: (message.translation as string) ?? null,
        source: (message.source as TranscriptSource) ?? null,
        itemId: meta.itemId,
        revision: meta.revision,
        sttProvider: meta.sttProvider,
      },
    },
  };
  const streams = nextStreamStatus(state.sttStreams, streamId, meta);
  if (streams) update.sttStreams = streams;
  if (meta.conversationId && meta.conversationId !== state.engineConversationId) {
    update.engineConversationId = meta.conversationId;
  }
  if (message.translation_backend) {
    update.activeTranslationBackend = message.translation_backend as string;
  }
  return update;
}

export function applyFinal(state: EngineState, message: EngineMessage): Partial<EngineState> {
  const finalText = (message.text as string) || "";
  const streamId = (message.source as string) || "default";
  const meta = readTranscriptMeta(message, streamId);

  const update: Partial<EngineState> = {};
  if (message.translation_backend) {
    update.activeTranslationBackend = message.translation_backend as string;
  }

  if (partialMatchesItem(state.partials[streamId], meta.itemId)) {
    const newPartials = { ...state.partials };
    delete newPartials[streamId];
    update.partials = newPartials;
  }

  const streams = nextStreamStatus(state.sttStreams, streamId, meta);
  if (streams) update.sttStreams = streams;
  const sttStreams = streams ?? state.sttStreams;
  if (meta.conversationId && meta.conversationId !== state.engineConversationId) {
    update.engineConversationId = meta.conversationId;
  }

  if (meta.itemId !== undefined) {
    const known = state.finalizedItems[meta.itemId];
    const revision = meta.revision ?? 0;
    // A redelivered or older final for an item that already committed.
    if (known !== undefined && revision <= known && finalText.trim()) {
      const byEntry = message.entry_id
        ? findLast(state.entries, (e) => e.id === message.entry_id)
        : -1;
      if (byEntry === -1) return update;
    }
    if (known === undefined || revision > known) {
      update.finalizedItems = withFinalizedItem(state.finalizedItems, meta.itemId, revision);
    }
  }

  if (!finalText.trim() || meta.retracted) return update;

  const entry: TranscriptEntry = {
    id: (message.entry_id as string) || nextId(),
    text: finalText,
    translation: (message.translation as string) ?? null,
    isFinal: true,
    timestamp: message.timestamp as number,
    source: (message.source as TranscriptSource) ?? null,
  };
  if (meta.itemId) entry.itemId = meta.itemId;
  if (meta.sttProvider) entry.sttProvider = meta.sttProvider;
  const fallbackFrom = sttStreams[streamId]?.fallbackFrom;
  if (fallbackFrom) entry.sttFallbackFrom = fallbackFrom;
  if (meta.continuity) entry.continuity = meta.continuity;
  if (meta.conversationId) entry.conversationId = meta.conversationId;
  if (meta.messageId) entry.messageId = meta.messageId;

  // A revisioned final for a known entry replaces it (older revisions are dropped). Without
  // a revision the legacy append behavior applies.
  const revision = revisionOf(message);
  if (revision !== undefined && message.entry_id) {
    const index = findLast(
      state.entries,
      (e) => e.id === entry.id || (Boolean(entry.messageId) && e.messageId === entry.messageId),
    );
    if (index !== -1) {
      const existing = state.entries[index];
      if (existing.revision !== undefined && existing.revision > revision) return update;
      const entries = [...state.entries];
      entries[index] = { ...existing, ...entry, revision };
      return { ...update, entries };
    }
    entry.revision = revision;
  }

  return { ...update, entries: [...state.entries, entry] };
}

export function applyTranslationUpdate(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> {
  const translation = message.translation as string;
  const entryId = message.entry_id as string | undefined;
  const sourceText = message.source_text as string;
  const rawError = isRecord(message.translation_error) ? message.translation_error : null;

  if (!translation && !rawError) return {};

  const newEntries = [...state.entries];
  let foundIndex = -1;

  // 1. Prefer exact entry_id match — precise
  if (entryId) {
    foundIndex = findLast(newEntries, (e) => e.id === entryId);
  }

  // 2. Fall back to exact text match (legacy messages without entry_id)
  if (foundIndex === -1 && sourceText) {
    foundIndex = findLast(newEntries, (e) => e.text === sourceText);
  }

  // 3. Last resort: prefix match on entries that don't yet have a translation
  if (foundIndex === -1 && sourceText) {
    const prefix = sourceText.slice(0, 30);
    foundIndex = findLast(newEntries, (e) => !e.translation && e.text.includes(prefix));
  }

  // If no match at all, discard — do NOT blindly attach to the last entry
  if (foundIndex === -1) return {};

  // A translation computed for an older revision of the text must not overwrite a newer one.
  const revision = revisionOf(message);
  const target = newEntries[foundIndex];
  if (revision !== undefined && target.revision !== undefined && revision < target.revision) {
    return {};
  }

  // The legacy engine only ever sends final translations; is_final: false marks a draft.
  const isFinal = message.is_final !== false;
  if (target.translationFinal && !isFinal) return {};
  if (
    revision !== undefined &&
    target.translationRevision !== undefined &&
    revision < target.translationRevision
  ) {
    return {};
  }

  if (!translation && rawError) {
    const code = nonEmptyString(rawError.code) ?? "failed";
    const errMsg = nonEmptyString(rawError.message) ?? "Translation failed for this segment.";
    newEntries[foundIndex] = {
      ...target,
      translationError: { code, message: errMsg },
    };
    return { entries: newEntries };
  }

  const next: TranscriptEntry = {
    ...target,
    translation,
    translationFinal: isFinal,
    translationError: null,
  };
  if (revision !== undefined) next.translationRevision = revision;
  const provider =
    nonEmptyString(message.translation_provider) ?? nonEmptyString(message.translation_backend);
  if (provider) next.translationProvider = provider;
  const fallback = message.translation_fallback;
  if (fallback === true || isRecord(fallback)) next.translationFallback = true;

  newEntries[foundIndex] = next;
  return { entries: newEntries };
}

// ---------------------------------------------------------------------------
// Assist
// ---------------------------------------------------------------------------

// Map key for the Assist request of one entry: the engine message id when it sent one.
export function assistKeyOf(entry: Pick<TranscriptEntry, "id" | "messageId">): string {
  return entry.messageId ?? entry.id;
}

export function isAssistantConfigured(state: Pick<EngineState, "config" | "services">): boolean {
  return state.config.llmEnabled || state.services.assistant.enabled;
}

const PROVIDER_ERROR_TEXT: Record<string, string> = {
  auth_failed: "The Assistant provider rejected its credentials.",
  rate_limited: "The Assistant provider is rate limiting requests.",
  quota_exceeded: "The Assistant provider quota is exhausted.",
  network_unavailable: "The Assistant provider could not be reached.",
  timeout: "The Assistant provider did not answer in time.",
  connection_closed: "The Assistant provider closed the connection.",
  model_unavailable: "The Assistant model is not available.",
  context_overflow: "The conversation is too long for the Assistant model.",
  invalid_response: "The Assistant returned a response in an invalid format.",
};

export const ASSIST_INTERRUPTED_TEXT = "The Assist request did not complete.";

export const ASSIST_UNAVAILABLE_TEXT: Record<AssistUnavailableReason, string> = {
  no_conversation:
    "Assist needs this message saved to an engine conversation. Live transcripts are not saved yet.",
  disconnected: "Assist needs a connected engine.",
};

function suggestedRepliesOf(message: EngineMessage): SuggestedReply[] {
  const structured = message.suggested_replies;
  const raw = Array.isArray(structured) ? structured : message.options;
  if (!Array.isArray(raw)) return [];
  const replies: SuggestedReply[] = [];
  for (const item of raw) {
    if (!isRecord(item) || typeof item.text !== "string" || !item.text.trim()) continue;
    const reply: SuggestedReply = {
      strategy: typeof item.strategy === "string" ? item.strategy : "",
      text: item.text,
    };
    if (typeof item.reasoning === "string" && item.reasoning) reply.reasoning = item.reasoning;
    replies.push(reply);
  }
  return replies;
}

function clarifyingQuestionsOf(message: EngineMessage): ClarifyingQuestion[] {
  const raw = message.clarifying_questions;
  if (!Array.isArray(raw)) return [];
  const questions: ClarifyingQuestion[] = [];
  for (const item of raw) {
    if (!isRecord(item) || typeof item.question !== "string" || !item.question.trim()) continue;
    const question: ClarifyingQuestion = { question: item.question };
    if (typeof item.purpose === "string" && item.purpose) question.purpose = item.purpose;
    questions.push(question);
  }
  return questions;
}

function assistErrorOf(message: EngineMessage): AssistError | null {
  const text = typeof message.error === "string" ? message.error : null;
  const code = nonEmptyString(message.error_code) ?? null;
  if (text === null && code === null) return null;
  const providerCode = nonEmptyString(message.provider_code) ?? null;
  const isProvider = providerCode !== null || code === "provider_error";
  return {
    kind: isProvider ? "provider" : "structured",
    code,
    providerCode,
    message:
      (providerCode && PROVIDER_ERROR_TEXT[providerCode]) ||
      text ||
      "The Assist request failed.",
    retryable: message.retryable === true,
  };
}

function providerIdOf(message: EngineMessage): string | null {
  const provider = message.provider;
  if (isRecord(provider)) {
    const id = provider.provider_id ?? provider.providerId ?? provider.id;
    if (typeof id === "string" && id) return id;
  }
  return typeof provider === "string" && provider ? provider : null;
}

function isStructuredAssistReply(message: EngineMessage): boolean {
  return (
    message.conversation_id !== undefined ||
    message.assist_id !== undefined ||
    message.suggested_replies !== undefined ||
    message.error_code !== undefined
  );
}

export function applySuggestionResult(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> {
  const entryId = message.entry_id as string;
  const pending = state.assistRequests[entryId];

  if (pending) {
    // The request already ended (bridge failure, session reset): whatever arrives now is not
    // an answer to anything the user is waiting on.
    if (!state.suggestionLoading[entryId]) return {};
    const conversationId = recordId(message.conversation_id);
    if (conversationId && conversationId !== pending.conversationId) return {};

    const outcome: AssistOutcome = {
      requestId: pending.requestId,
      assistId: nonEmptyString(message.assist_id) ?? null,
      conversationId: pending.conversationId,
      suggestedReplies: suggestedRepliesOf(message),
      clarifyingQuestions: clarifyingQuestionsOf(message),
      providerId: providerIdOf(message),
      error: assistErrorOf(message),
    };
    const suggestionLoading = { ...state.suggestionLoading };
    delete suggestionLoading[entryId];
    const assistRequests = { ...state.assistRequests };
    delete assistRequests[entryId];

    const entries = state.entries.map((e) => {
      if (assistKeyOf(e) !== entryId) return e;
      // The legacy mirror keeps replies in the persisted history schema.
      const suggestions: SuggestionResult = outcome.error
        ? { error: outcome.error.message }
        : { options: outcome.suggestedReplies.map((r) => ({ strategy: r.strategy, text: r.text })) };
      return { ...e, assist: outcome, suggestions };
    });
    return { suggestionLoading, assistRequests, entries };
  }

  // A structured reply with no pending request is stale or redelivered.
  if (isStructuredAssistReply(message)) return {};

  // Legacy path (client-context requestSuggestion, e.g. the Ctrl+Shift+S shortcut).
  const newLoading = { ...state.suggestionLoading };
  delete newLoading[entryId];

  // Store result inside the matching TranscriptEntry (persisted with session)
  const newEntries = state.entries.map((e) => {
    if (e.id !== entryId) return e;
    return {
      ...e,
      suggestions: {
        options: (message.options as SuggestionOption[]) ?? undefined,
        error: (message.error as string) ?? undefined,
      } as SuggestionResult,
    };
  });

  return { suggestionLoading: newLoading, entries: newEntries };
}

export function assistUnavailableReason(
  state: Pick<EngineState, "connected" | "socket" | "engineConversationId">,
  entry: Pick<TranscriptEntry, "conversationId">,
): AssistUnavailableReason | null {
  if (!(entry.conversationId ?? state.engineConversationId)) return "no_conversation";
  if (!state.connected || !state.socket) return "disconnected";
  return null;
}

export interface AssistViewInput {
  configured: boolean;
  entry: TranscriptEntry;
  loading: boolean;
  request: AssistRequestState | undefined;
  unavailable: AssistUnavailableReason | null;
}

export interface AssistView {
  phase: AssistPhase;
  suggestedReplies: SuggestedReply[];
  clarifyingQuestions: ClarifyingQuestion[];
  error: AssistError | null;
  unavailable: AssistUnavailableReason | null;
  canRequest: boolean;
}

// The per-message Assist state machine. Assist exists only on finalized, non-empty messages
// when an Assistant is configured; everything else renders nothing.
export function deriveAssistView(input: AssistViewInput): AssistView {
  const { configured, entry, loading, request, unavailable } = input;
  const view: AssistView = {
    phase: "hidden",
    suggestedReplies: [],
    clarifyingQuestions: [],
    error: null,
    unavailable,
    canRequest: false,
  };
  if (!configured || !entry.isFinal || !entry.text.trim()) return view;
  view.canRequest = unavailable === null && !loading;

  if (loading) return { ...view, phase: "loading" };
  if (request) {
    return {
      ...view,
      phase: "error",
      error: {
        kind: "interrupted",
        code: null,
        providerCode: null,
        message: ASSIST_INTERRUPTED_TEXT,
        retryable: true,
      },
    };
  }

  const outcome = entry.assist;
  if (outcome) {
    if (outcome.error) return { ...view, phase: "error", error: outcome.error };
    const empty = outcome.suggestedReplies.length === 0 && outcome.clarifyingQuestions.length === 0;
    return {
      ...view,
      phase: empty ? "empty" : "success",
      suggestedReplies: outcome.suggestedReplies,
      clarifyingQuestions: outcome.clarifyingQuestions,
    };
  }

  const legacy = entry.suggestions;
  if (legacy) {
    if (legacy.error) {
      return {
        ...view,
        phase: "error",
        error: {
          kind: "structured",
          code: null,
          providerCode: null,
          message: legacy.error,
          retryable: true,
        },
      };
    }
    const replies = (legacy.options ?? []).map((o) => ({ strategy: o.strategy ?? "", text: o.text }));
    return { ...view, phase: replies.length ? "success" : "empty", suggestedReplies: replies };
  }

  return { ...view, phase: unavailable ? "unavailable" : "idle" };
}

// ---------------------------------------------------------------------------
// STT provider status
// ---------------------------------------------------------------------------

export function sttProviderLabel(providerId: string | null | undefined): string {
  if (!providerId) return "Unknown";
  return findProvider("stt", providerId)?.displayName ?? providerId;
}

export interface SttNotice {
  fallbacks: { source: string; from: string }[];
  continuityCount: number;
  latestContinuity: ContinuityInterval | null;
  translationFallback: boolean;
  translationErrorCount: number;
}

// Fallback and continuity warnings for the visible conversation. Entries carry them, so they
// survive Stop and History; live stream state adds what has not reached a final yet.
export function deriveSttNotice(
  entries: TranscriptEntry[],
  sttStreams: Record<string, StreamSttStatus>,
  running: boolean,
): SttNotice {
  const fallbacks = new Map<string, string>();
  const intervals = new Map<string, ContinuityInterval>();
  let translationFallback = false;
  let translationErrorCount = 0;
  for (const entry of entries) {
    if (entry.sttFallbackFrom && entry.source) fallbacks.set(entry.source, entry.sttFallbackFrom);
    if (entry.continuity) intervals.set(continuityKey(entry.continuity), entry.continuity);
    if (entry.translationFallback) translationFallback = true;
    if (entry.translationError) translationErrorCount += 1;
  }
  if (running) {
    for (const stream of Object.values(sttStreams)) {
      if (stream.fallbackFrom) fallbacks.set(stream.source, stream.fallbackFrom);
      for (const interval of stream.continuity) intervals.set(continuityKey(interval), interval);
    }
  }
  let latest: ContinuityInterval | null = null;
  for (const interval of intervals.values()) {
    if (!latest || interval.endMs >= latest.endMs) latest = interval;
  }
  return {
    fallbacks: [...fallbacks.entries()]
      .map(([source, from]) => ({ source, from }))
      .sort((a, b) => streamRank(a.source) - streamRank(b.source)),
    continuityCount: intervals.size,
    latestContinuity: latest,
    translationFallback,
    translationErrorCount,
  };
}

export function streamRank(source: string): number {
  const index = STREAM_ORDER.indexOf(source);
  return index === -1 ? STREAM_ORDER.length : index;
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

export interface TranscriptSlice {
  entries: TranscriptEntry[];
  partials: Record<string, PartialTranscript>;
  activeTranslationBackend: string | null;
  // Ephemeral, never persisted. Keyed by entry id (the engine message id for Assist).
  suggestionLoading: Record<string, boolean>;
  assistRequests: Record<string, AssistRequestState>;
  sttStreams: Record<string, StreamSttStatus>;
  // item_id -> committed revision; late partials and redelivered finals are dropped.
  finalizedItems: Record<string, number>;
  // Engine conversation record id, once the engine reports one. Never the local history id.
  engineConversationId: string | null;

  clearTranscript: () => void;
  // Legacy client-context request, kept for the global shortcut until it moves to Assist.
  requestSuggestion: (entryId: string, targetText: string, context: string[]) => void;
  // Manual Assist on one finalized message. Sends ids only; the engine selects the context.
  requestAssist: (entryId: string) => void;
  // Inline text or speaker correction on a finalized message, persisted to SQLite when available.
  updateEntry: (
    entryId: string,
    patch: { text?: string; source?: "mic" | "system" | "both" },
  ) => Promise<boolean>;
}

export const createTranscriptSlice =
  (commands: EngineCommands): EngineSliceCreator<TranscriptSlice> =>
  (set, get, api) => {
    let assistCounter = 0;

    // Session metadata belongs to one Start-to-Stop run.
    api.subscribe((state, prev) => {
      if (state.running && !prev.running) {
        set({
          assistRequests: {},
          sttStreams: {},
          finalizedItems: {},
          engineConversationId: state.activeConversationId ?? null,
        });
      }
    });

    return {
      ...transcriptResetState(),
      activeTranslationBackend: null,

      clearTranscript: () => {
        set({
          ...transcriptResetState(),
          summaryText: "",
          summaryVisible: false,
          askAnswer: "",
          askHistory: [],
          askError: null,
          activeConversationId: null,
          isViewingHistory: false,
          preserveEntriesOnStart: false,
          pendingTitle: "",
        });
      },

      requestSuggestion: async (entryId, targetText, context) => {
        const { socket, connected, config } = get();

        // If we don't have a direct socket connection (like in the overlay window),
        // we relay the request to the main window via Tauri events.
        if (!socket || !connected) {
          if (
            isTauriRuntime() ||
            (typeof window !== "undefined" && (window as unknown as { __TAURI__?: unknown }).__TAURI__)
          ) {
            const { emit } = await import("@tauri-apps/api/event");
            await emit("relay_request_suggestion", { entryId, targetText, context });
          }
          return;
        }

        set((state) => ({
          suggestionLoading: { ...state.suggestionLoading, [entryId]: true },
        }));

        socket.send(commands.requestSuggestion(entryId, targetText, context, config));
      },

      requestAssist: (entryId) => {
        const state = get();
        const entry = state.entries.find((e) => e.id === entryId || (e.messageId && e.messageId === entryId));
        if (!entry || !entry.isFinal || !entry.text.trim()) return;
        if (!isAssistantConfigured(state)) return;
        const key = assistKeyOf(entry);
        if (state.suggestionLoading[key]) return;

        // Overlay window mirrors state without a direct socket; relay to main window first.
        if (!state.socket && state.connected) {
          if (
            isTauriRuntime() ||
            (typeof window !== "undefined" && (window as unknown as { __TAURI__?: unknown }).__TAURI__)
          ) {
            void import("@tauri-apps/api/event").then(({ emit }) => {
              void emit("relay_request_assist", { entryId });
            });
          }
          return;
        }

        if (assistUnavailableReason(state, entry) !== null) return;
        const assistantSvc = state.services?.assistant;
        if (
          isTauriRuntime() &&
          assistantSvc?.enabled &&
          assistantSvc.providerId !== "local-openai-compatible" &&
          !state.consent?.assistant
        ) {
          set({
            appError:
              "Cloud consent is required for AI Assistant. Grant permission in Settings before using Assist.",
          });
          return;
        }

        const conversationId = (entry.conversationId ?? state.engineConversationId) as string;
        assistCounter += 1;
        const request: AssistRequestState = {
          requestId: `assist-${assistCounter}`,
          conversationId,
          messageId: key,
        };
        set((s) => ({
          suggestionLoading: { ...s.suggestionLoading, [key]: true },
          assistRequests: { ...s.assistRequests, [key]: request },
        }));

        // The command type comes from the one wire encoder in engineStore.ts; the body is the
        // id-only repository shape, so no text, context, or llm_config leaves the webview.
        const { type } = JSON.parse(
          commands.requestSuggestion(key, "", [], publicConfigSnapshot(state.config)),
        ) as { type: string };
        const sendAssist = () => {
          get().socket?.send(
            JSON.stringify({ type, conversation_id: conversationId, message_id: key, entry_id: key }),
          );
        };
        if (isTauriRuntime() && state.services?.assistant?.enabled) {
          void state.syncAssistantService().then((ok) => {
            if (!ok) {
              set((s) => {
                const suggestionLoading = { ...s.suggestionLoading };
                delete suggestionLoading[key];
                const assistRequests = { ...s.assistRequests };
                delete assistRequests[key];
                return {
                  suggestionLoading,
                  assistRequests,
                  appError:
                    "Assistant is not configured or credentials are missing. Check Assistant settings.",
                };
              });
              return;
            }
            sendAssist();
          });
        } else {
          sendAssist();
        }
      },

      updateEntry: async (entryId, patch) => {
        const state = get();
        const idx = state.entries.findIndex(
          (e) => e.id === entryId || (e.messageId && e.messageId === entryId),
        );
        if (idx === -1) return false;
        const prevEntry = state.entries[idx];
        const nextText = patch.text !== undefined ? patch.text.trim() : prevEntry.text;
        if (!nextText) return false;
        const nextSource = patch.source !== undefined ? patch.source : prevEntry.source;
        if (nextText === prevEntry.text && nextSource === prevEntry.source) return true;

        const prevEntries = state.entries;
        const updatedEntries = [...prevEntries];
        updatedEntries[idx] = { ...prevEntry, text: nextText, source: nextSource };
        set({ entries: updatedEntries });

        const convId = prevEntry.conversationId ?? state.engineConversationId ?? state.activeConversationId;
        const msgId = prevEntry.messageId ?? prevEntry.id;
        if (isTauriRuntime() && convId && msgId) {
          try {
            const res = await forwardCommand({
              type: "message_update",
              conversation_id: convId,
              message_id: msgId,
              ...(patch.text !== undefined ? { text: nextText } : {}),
              ...(patch.source !== undefined && nextSource ? { source: nextSource } : {}),
            });
            if (patch.text !== undefined) {
              const reply = res.messages.find(
                (item) => typeof (item as Record<string, unknown>).translation === "string",
              ) as Record<string, unknown> | undefined;
              const retranslated =
                typeof reply?.translation === "string" ? reply.translation.trim() : "";
              if (retranslated) {
                set((s) => ({
                  entries: s.entries.map((entry) =>
                    entry.id === prevEntry.id ||
                    (Boolean(prevEntry.messageId) && entry.messageId === prevEntry.messageId)
                      ? { ...entry, translation: retranslated, translationFinal: true }
                      : entry,
                  ),
                }));
              }
            }
          } catch (err) {
            set({
              entries: prevEntries,
              appError: `Update segment: ${describeCommandError(errorCodeOf(err))}`,
            });
            return false;
          }
        }
        get().saveCurrentSession();
        return true;
      },
    };
  };
