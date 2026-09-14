import type { EngineSliceCreator, EngineState } from "./engineStore";
import type { EngineCommands } from "./sessionSlice";
import type {
  EngineMessage,
  PartialTranscript,
  SuggestionOption,
  SuggestionResult,
  TranscriptEntry,
  TranscriptSource,
} from "./types";

// Transcript, async translation, and Assist (suggestion) state.

let idCounter = 0;

function nextId(): string {
  idCounter += 1;
  return `t-${Date.now()}-${idCounter}`;
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

export function transcriptResetState(): Pick<
  TranscriptSlice,
  "entries" | "partials" | "suggestionLoading"
> {
  return { entries: [], partials: {}, suggestionLoading: {} };
}

// Returns null when the anti-flicker heuristic rejects the update, so no state write occurs.
export function applyPartial(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> | null {
  const newPartial = (message.text as string) || "";
  const streamId = (message.source as string) || "default";
  const currentPartial = state.partials[streamId]?.text || "";

  if (
    !(
      newPartial.length >= currentPartial.length ||
      newPartial.length >= currentPartial.length * 0.6 ||
      currentPartial === ""
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
      },
    },
  };
  if (message.translation_backend) {
    update.activeTranslationBackend = message.translation_backend as string;
  }
  return update;
}

export function applyFinal(state: EngineState, message: EngineMessage): Partial<EngineState> {
  const finalText = (message.text as string) || "";
  const streamId = (message.source as string) || "default";

  const update: Partial<EngineState> = {};
  if (message.translation_backend) {
    update.activeTranslationBackend = message.translation_backend as string;
  }

  const newPartials = { ...state.partials };
  delete newPartials[streamId];
  update.partials = newPartials;

  if (!finalText.trim()) return update;

  const entry: TranscriptEntry = {
    id: (message.entry_id as string) || nextId(),
    text: finalText,
    translation: (message.translation as string) ?? null,
    isFinal: true,
    timestamp: message.timestamp as number,
    source: (message.source as TranscriptSource) ?? null,
  };

  // A revisioned final for a known entry replaces it (older revisions are dropped). Without
  // a revision the legacy append behavior applies.
  const revision = revisionOf(message);
  if (revision !== undefined && message.entry_id) {
    const index = findLast(state.entries, (e) => e.id === entry.id);
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

  if (!translation) return {};

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

  newEntries[foundIndex] = { ...target, translation };
  return { entries: newEntries };
}

export function applySuggestionResult(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> {
  const entryId = message.entry_id as string;

  // Remove loading flag
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

export interface TranscriptSlice {
  entries: TranscriptEntry[];
  partials: Record<string, PartialTranscript>;
  activeTranslationBackend: string | null;
  // Ephemeral, never persisted.
  suggestionLoading: Record<string, boolean>;

  clearTranscript: () => void;
  requestSuggestion: (entryId: string, targetText: string, context: string[]) => void;
}

export const createTranscriptSlice =
  (commands: EngineCommands): EngineSliceCreator<TranscriptSlice> =>
  (set, get) => ({
    ...transcriptResetState(),
    activeTranslationBackend: null,

    clearTranscript: () => {
      set({
        ...transcriptResetState(),
        summaryText: "",
        summaryVisible: false,
        activeConversationId: null,
      });
    },

    requestSuggestion: async (entryId, targetText, context) => {
      const { socket, connected, config } = get();

      // If we don't have a direct socket connection (like in the overlay window),
      // we relay the request to the main window via Tauri events.
      if (!socket || !connected) {
        if (typeof window !== "undefined" && window.__TAURI__) {
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
  });
