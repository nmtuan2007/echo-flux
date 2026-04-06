import { StateCreator } from "zustand";
import { Conversation, TranscriptEntry } from "../types";
import { StoreState } from "../engineStore";

export interface TranscriptSlice {
  entries: TranscriptEntry[];
  partials: Record<string, {
    text: string;
    translation: string | null;
    source: "mic" | "system" | "both" | null;
  }>;
  history: Conversation[];
  activeConversationId: string | null;

  suggestionLoading: Record<string, boolean>;
  summaryText: string;
  summaryLoading: boolean;
  summaryVisible: boolean;

  clearTranscript: () => void;
  openSummary: () => void;
  closeSummary: () => void;

  renameConversation: (id: string, name: string) => void;
  deleteConversation: (id: string) => void;
  saveCurrentSession: () => void;
  loadConversation: (id: string) => void;
}

export const createTranscriptSlice: StateCreator<StoreState, [], [], TranscriptSlice> = (set, get) => ({
  entries: [],
  partials: {},
  history: [],
  activeConversationId: null,

  suggestionLoading: {},
  summaryText: "",
  summaryLoading: false,
  summaryVisible: false,

  clearTranscript: () => set({
    entries: [],
    partials: {},
    suggestionLoading: {},
    summaryText: "",
    summaryVisible: false,
    activeConversationId: null,
  }),

  openSummary: () => set({ summaryVisible: true }),

  closeSummary: () => set({ summaryVisible: false, summaryLoading: false }),

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
        activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
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
