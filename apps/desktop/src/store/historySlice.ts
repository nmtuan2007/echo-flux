import {
  describeCommandError,
  errorCodeOf,
  forwardCommand,
  isTauriRuntime,
} from "../lib/engineBridge";
import type { EngineState } from "./engineStore";
import {
  MAX_TITLE_CHARS,
  type Conversation,
  type ConversationRecord,
  type SuggestionOption,
  type SuggestionResult,
  type TranscriptEntry,
} from "./types";

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
    source:
      typeof raw.source === "string" && SOURCES.has(raw.source)
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

export interface HistorySliceFields {
  history: Conversation[];
  activeConversationId: string | null;
  isViewingHistory: boolean;
  savedConversations: ConversationRecord[];
  selectedConversationDetail: any | null;
  historyLoading: boolean;
  detailLoading: boolean;
  historySearchQuery: string;

  setHistorySearchQuery: (query: string) => void;
  loadHistoryList: (searchQuery?: string) => Promise<void>;
  getConversationDetail: (conversationId: string) => Promise<any | null>;
  clearSelectedConversationDetail: () => void;
  renameConversation: (id: string, name: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  exportConversation: (id: string) => Promise<any | null>;
  importConversation: (exportData: Record<string, unknown> | string) => Promise<boolean>;
  migrateLegacyHistory: () => Promise<void>;
  saveCurrentSession: () => void;
  loadConversation: (id: string, options?: { continueSession?: boolean }) => void;
}

function hydrateDetailEntries(detail: any, id: string): {
  entries: TranscriptEntry[];
  sourceLabels?: { mic: string; system: string };
  summaryText: string;
  title: string;
} {
  const rawMessages = detail.messages || [];
  const assistByMsgId = new Map<string, any>();
  for (const a of detail.assist_results || []) {
    if (a?.target_message_id) {
      assistByMsgId.set(a.target_message_id, a);
    }
  }
  const entries: TranscriptEntry[] = rawMessages.map((m: any, idx: number) => {
    const msgId = m.message_id || `msg-${idx}`;
    const savedAssist = assistByMsgId.get(msgId);
    return {
      id: msgId,
      messageId: m.message_id || undefined,
      conversationId: id,
      text: m.stt_text || m.text || "",
      translation: m.translation_text ?? m.translation ?? null,
      isFinal: true,
      timestamp:
        typeof m.timestamp_ms === "number"
          ? m.timestamp_ms / 1000
          : typeof m.timestamp === "number"
            ? m.timestamp
            : 0,
      startMs: typeof m.start_ms === "number" ? m.start_ms : null,
      endMs: typeof m.end_ms === "number" ? m.end_ms : null,
      source:
        m.source === "mic" || m.source === "system" || m.source === "both" ? m.source : null,
      revision: m.revision ?? 1,
      ...(savedAssist
        ? {
            assist: {
              requestId: `stored-${savedAssist.assist_id || msgId}`,
              assistId: savedAssist.assist_id ?? null,
              conversationId: id,
              suggestedReplies: Array.isArray(savedAssist.suggested_replies)
                ? savedAssist.suggested_replies
                : [],
              clarifyingQuestions: Array.isArray(savedAssist.clarifying_questions)
                ? savedAssist.clarifying_questions
                : [],
              providerId: savedAssist.provider_id ?? null,
              error: null,
            },
          }
        : {}),
    };
  });
  const convLabels = detail.conversation?.source_labels || detail.source_labels;
  return {
    entries,
    summaryText:
      detail.summary?.summary_markdown ||
      (typeof detail.summary === "string" ? detail.summary : ""),
    title: detail.conversation?.title || detail.title || "",
    ...(convLabels?.mic || convLabels?.system
      ? {
          sourceLabels: {
            mic: convLabels.mic || "Me",
            system: convLabels.system || "Others",
          },
        }
      : {}),
  };
}

export function createHistoryActions(
  set: (
    partial: Partial<EngineState> | ((state: EngineState) => Partial<EngineState>),
  ) => void,
  get: () => EngineState,
): HistorySliceFields {
  return {
    history: [],
    activeConversationId: null,
    isViewingHistory: false,
    savedConversations: [],
    selectedConversationDetail: null,
    historyLoading: false,
    detailLoading: false,
    historySearchQuery: "",

    setHistorySearchQuery: (query) => set({ historySearchQuery: query }),

    loadHistoryList: async (searchQuery) => {
      set({ historyLoading: true });
      try {
        const q = (searchQuery ?? get().historySearchQuery).trim();
        const res = await forwardCommand({
          type: "conversation_list",
          ...(q ? { search_query: q } : {}),
        });
        const dataMsg = res.messages.find(
          (m) => (m as any).conversations !== undefined || m.type === "conversation_list_result",
        );
        const convs: ConversationRecord[] = (
          (dataMsg as any)?.conversations ||
          (dataMsg as any)?.data?.conversations ||
          []
        ).map((c: any) => ({
          conversation_id: c.conversation_id || c.conversationId,
          title: c.title || "",
          created_at_ms: c.created_at_ms || c.createdAtMs || 0,
          updated_at_ms: c.updated_at_ms || c.updatedAtMs || 0,
          source_labels: c.source_labels || c.sourceLabels,
          import_id: c.import_id || c.importId,
          message_count: c.message_count ?? c.messageCount ?? 0,
          has_summary: Boolean(c.has_summary ?? c.hasSummary),
        }));
        set({ savedConversations: convs, historyLoading: false });
      } catch (err) {
        const code = errorCodeOf(err);
        set({
          historyLoading: false,
          ...(isTauriRuntime()
            ? { appError: `Failed to load history: ${describeCommandError(code)} (${code})` }
            : {}),
        });
      }
    },

    getConversationDetail: async (conversationId) => {
      set({ detailLoading: true });
      try {
        const res = await forwardCommand({
          type: "conversation_get",
          conversation_id: conversationId,
          include_messages: true,
        });
        const first =
          res.messages.find((m) => (m as any).conversation || (m as any).messages) ||
          res.messages[0];
        set({ selectedConversationDetail: first, detailLoading: false });
        return first;
      } catch (err) {
        const code = errorCodeOf(err);
        set({
          detailLoading: false,
          appError: `Failed to open conversation: ${describeCommandError(code)} (${code})`,
        });
        return null;
      }
    },

    clearSelectedConversationDetail: () => set({ selectedConversationDetail: null }),

    renameConversation: async (id, name) => {
      const trimmed = name.trim().slice(0, MAX_TITLE_CHARS);
      if (!trimmed) return;
      const prevHistory = get().history;
      const prevSaved = get().savedConversations;
      set((state) => ({
        history: state.history.map((c) => (c.id === id ? { ...c, name: trimmed } : c)),
        savedConversations: state.savedConversations.map((c) =>
          c.conversation_id === id ? { ...c, title: trimmed } : c,
        ),
      }));
      if (isTauriRuntime()) {
        try {
          await forwardCommand({
            type: "conversation_rename",
            conversation_id: id,
            title: trimmed,
          });
        } catch (err) {
          const code = errorCodeOf(err);
          set({
            history: prevHistory,
            savedConversations: prevSaved,
            appError: `Rename failed: ${describeCommandError(code)} (${code})`,
          });
        }
      }
    },

    deleteConversation: async (id) => {
      const prevHistory = get().history;
      const prevSaved = get().savedConversations;
      const prevActive = get().activeConversationId;
      const prevDetail = get().selectedConversationDetail;
      set((state) => ({
        history: state.history.filter((c) => c.id !== id),
        savedConversations: state.savedConversations.filter((c) => c.conversation_id !== id),
        activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
        selectedConversationDetail:
          state.selectedConversationDetail?.conversation?.conversation_id === id
            ? null
            : state.selectedConversationDetail,
      }));
      if (isTauriRuntime()) {
        try {
          await forwardCommand({ type: "conversation_delete", conversation_id: id });
        } catch (err) {
          const code = errorCodeOf(err);
          set({
            history: prevHistory,
            savedConversations: prevSaved,
            activeConversationId: prevActive,
            selectedConversationDetail: prevDetail,
            appError: `Delete failed: ${describeCommandError(code)} (${code})`,
          });
        }
      }
    },

    exportConversation: async (id) => {
      try {
        const res = await forwardCommand({ type: "conversation_export", conversation_id: id });
        const first = res.messages.find((m) => (m as any).export !== undefined) || res.messages[0];
        return (first as any)?.export || first;
      } catch (err) {
        const code = errorCodeOf(err);
        set({ appError: `Export failed: ${describeCommandError(code)} (${code})` });
        return null;
      }
    },

    importConversation: async (exportData) => {
      try {
        const payload =
          typeof exportData === "string"
            ? (JSON.parse(exportData) as Record<string, unknown>)
            : exportData;
        const res = await forwardCommand({ type: "conversation_import", export_data: payload });
        await get().loadHistoryList();
        const hasError = res.messages.some((m) => m.type === "error");
        if (hasError) set({ appError: "Failed to import conversation data." });
        return !hasError;
      } catch (err) {
        const code = errorCodeOf(err);
        set({ appError: `Import failed: ${describeCommandError(code)} (${code})` });
        return false;
      }
    },

    migrateLegacyHistory: async () => {
      const { history } = get();
      if (!history || history.length === 0) return;

      for (const item of [...history]) {
        const exportData = {
          conversation_id: item.id,
          title: item.name || `Session ${new Date(item.date).toLocaleString()}`,
          created_at_ms: item.date || Date.now(),
          updated_at_ms: item.date || Date.now(),
          source_labels: { mic: "Me", system: "Others" },
          import_id: `legacy-${item.id}`,
          messages: (item.entries || []).map((e, index) => ({
            message_id: e.id || `msg-${item.id}-${index}`,
            source: e.source === "system" ? "system" : "mic",
            text: e.text || "",
            translation: e.translation || null,
            is_final: e.isFinal !== false,
            revision: e.revision || 1,
            timestamp_ms: (e.timestamp ? e.timestamp * 1000 : item.date) + index * 100,
          })),
          summaries: item.summary
            ? [{ summary_markdown: item.summary, created_at_ms: item.date || Date.now() }]
            : [],
        };

        try {
          const res = await forwardCommand({
            type: "conversation_import",
            export_data: exportData,
          });
          if (res && !res.messages.some((m) => m.type === "error")) {
            set((state) => ({
              history: state.history.filter((c) => c.id !== item.id),
            }));
          }
        } catch {
          // Keep item in state to retry later
        }
      }
    },

    saveCurrentSession: () => {
      const { entries, summaryText, activeConversationId, engineConversationId } = get();
      if (entries.length === 0) return;

      if (isTauriRuntime() && (activeConversationId || engineConversationId)) {
        return;
      }

      if (activeConversationId) {
        set((state) => ({
          history: state.history.map((c) =>
            c.id === activeConversationId
              ? { ...c, entries: [...entries], summary: summaryText || c.summary }
              : c,
          ),
        }));
        return;
      }

      const timestamp = Date.now();
      const dateStr = new Date(timestamp).toLocaleString();
      let name = "Session " + dateStr;
      if (entries[0]?.text) {
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

    loadConversation: (id, options) => {
      const { history, running, selectedConversationDetail } = get();
      if (running) return;
      const continueSession = options?.continueSession === true;

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
          pendingTitle: target.name || "",
          isViewingHistory: !continueSession,
          ...(continueSession ? { preparedConversationId: id, engineConversationId: id } : {}),
        });
        return;
      }

      const detail = selectedConversationDetail;
      const detailId = detail?.conversation?.conversation_id || detail?.conversation_id;
      if (detail && detailId === id) {
        const hydrated = hydrateDetailEntries(detail, id);
        set({
          entries: hydrated.entries,
          activeView: "transcript",
          partials: {},
          suggestionLoading: {},
          assistRequests: {},
          summaryText: hydrated.summaryText,
          summaryVisible: false,
          summaryLoading: false,
          activeConversationId: id,
          engineConversationId: id,
          isViewingHistory: !continueSession,
          preparedConversationId: continueSession ? id : null,
          pendingTitle: hydrated.title,
          ...(hydrated.sourceLabels ? { sourceLabels: hydrated.sourceLabels } : {}),
        });
      }
    },
  };
}
