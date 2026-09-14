import type { EngineSliceCreator, EngineState } from "./engineStore";
import type { EngineCommands } from "./sessionSlice";
import type {
  Conversation,
  EngineMessage,
  SuggestionOption,
  SuggestionResult,
  TranscriptEntry,
} from "./types";

// Local conversation history, meeting summary, and the active-conversation context.

const SOURCES = new Set(["mic", "system", "both"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitizeSuggestions(raw: unknown): SuggestionResult | undefined {
  if (!isRecord(raw)) return undefined;
  const result: SuggestionResult = {};
  if (Array.isArray(raw.options)) {
    result.options = raw.options
      .filter(
        (o): o is SuggestionOption =>
          isRecord(o) && typeof o.strategy === "string" && typeof o.text === "string",
      )
      .map((o) => ({ strategy: o.strategy, text: o.text }));
  }
  if (typeof raw.error === "string") result.error = raw.error;
  return result;
}

function sanitizeEntry(raw: unknown): TranscriptEntry | null {
  if (!isRecord(raw) || typeof raw.id !== "string" || typeof raw.text !== "string") return null;
  const entry: TranscriptEntry = {
    id: raw.id,
    text: raw.text,
    translation: typeof raw.translation === "string" ? raw.translation : null,
    isFinal: raw.isFinal !== false,
    timestamp: typeof raw.timestamp === "number" ? raw.timestamp : 0,
    source: typeof raw.source === "string" && SOURCES.has(raw.source)
      ? (raw.source as TranscriptEntry["source"])
      : null,
  };
  const suggestions = sanitizeSuggestions(raw.suggestions);
  if (suggestions) entry.suggestions = suggestions;
  if (typeof raw.revision === "number") entry.revision = raw.revision;
  return entry;
}

// Approved local history, rebuilt field by field so nothing outside the schema is stored.
export function sanitizeHistory(raw: unknown): Conversation[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const history: Conversation[] = [];
  for (const item of raw) {
    if (!isRecord(item) || typeof item.id !== "string") continue;
    const conversation: Conversation = {
      id: item.id,
      name: typeof item.name === "string" ? item.name : item.id,
      date: typeof item.date === "number" ? item.date : 0,
      entries: Array.isArray(item.entries)
        ? item.entries.map(sanitizeEntry).filter((e): e is TranscriptEntry => e !== null)
        : [],
    };
    if (typeof item.summary === "string") conversation.summary = item.summary;
    history.push(conversation);
  }
  return history;
}

export function applyLlmChunk(state: EngineState, message: EngineMessage): Partial<EngineState> {
  const text = (message.text as string) || "";
  return { summaryText: state.summaryText + text };
}

export interface ConversationSlice {
  history: Conversation[];
  activeConversationId: string | null;

  // summaryText is the live buffer / current session summary
  summaryText: string;
  summaryLoading: boolean;
  summaryVisible: boolean;

  requestSummary: () => void;
  openSummary: () => void;
  closeSummary: () => void;

  renameConversation: (id: string, name: string) => void;
  deleteConversation: (id: string) => void;
  saveCurrentSession: () => void;
  loadConversation: (id: string) => void;
}

export const createConversationSlice =
  (commands: EngineCommands): EngineSliceCreator<ConversationSlice> =>
  (set, get) => ({
    history: [],
    activeConversationId: null,
    summaryText: "",
    summaryLoading: false,
    summaryVisible: false,

    requestSummary: () => {
      const { socket, connected, entries, config } = get();
      if (!socket || !connected || entries.length === 0) return;

      set({ summaryText: "", summaryLoading: true, summaryVisible: true });

      socket.send(commands.requestSummary(entries, config));
    },

    openSummary: () => {
      set({ summaryVisible: true });
    },

    closeSummary: () => {
      set({ summaryVisible: false, summaryLoading: false });
    },

    saveCurrentSession: () => {
      const { entries, summaryText, activeConversationId } = get();
      if (entries.length === 0) return;

      if (activeConversationId) {
        set((state) => ({
          history: state.history.map((c) =>
            c.id === activeConversationId
              ? { ...c, entries: [...entries], summary: summaryText || c.summary }
              : c
          ),
        }));
        return;
      }

      const timestamp = Date.now();
      const dateStr = new Date(timestamp).toLocaleString();

      let name = "Session " + dateStr;
      if (entries[0] && entries[0].text) {
        name = entries[0].text.slice(0, 30) + (entries[0].text.length > 30 ? "..." : "");
      }

      const newId = `conv-${timestamp}`;
      const conversation: Conversation = {
        id: newId,
        name,
        date: timestamp,
        entries: [...entries],
        summary: summaryText || undefined,
      };

      set((state) => ({
        activeConversationId: newId,
        history: [conversation, ...state.history],
      }));
    },

    renameConversation: (id, name) => {
      set((state) => ({
        history: state.history.map((c) => (c.id === id ? { ...c, name } : c)),
      }));
    },

    deleteConversation: (id) => {
      set((state) => {
        const newHistory = state.history.filter((c) => c.id !== id);
        return {
          history: newHistory,
          activeConversationId:
            state.activeConversationId === id ? null : state.activeConversationId,
        };
      });
    },

    loadConversation: (id) => {
      const { history, running } = get();
      if (running) return;

      const target = history.find((c) => c.id === id);
      if (target) {
        set({
          entries: [...target.entries],
          activeView: "transcript",
          partials: {},
          suggestionLoading: {},
          summaryText: target.summary || "",
          summaryVisible: false,
          summaryLoading: false,
          activeConversationId: id,
        });
      }
    },
  });
