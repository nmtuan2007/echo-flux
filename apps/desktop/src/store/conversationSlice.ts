import type { EngineSliceCreator, EngineState } from "./engineStore";
import type { EngineCommands } from "./sessionSlice";
import { publicConfigSnapshot } from "./settingsSlice";
import { describeCommandError, forwardCommand, isTauriRuntime } from "../lib/engineBridge";
import type {
  ContextProfile,
  ContextProfileInfo,
  Conversation,
  ConversationRecord,
  EngineMessage,
  ProfileContentFormat,
  SuggestionOption,
  SuggestionResult,
  SummaryWarning,
  TranscriptEntry,
} from "./types";
import {
  MAX_LABEL_CHARS,
  MAX_PROFILES_PER_SESSION,
  MAX_PROFILE_CONTENT_BYTES,
  MAX_PROFILE_NAME_CHARS,
  MAX_TITLE_CHARS,
} from "./types";

// Local conversation history, meeting summary, context profiles, and active-conversation context.

const SOURCES = new Set(["mic", "system", "both"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function utf8ByteLength(str: string): number {
  return new TextEncoder().encode(str).length;
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

export function applyLlmChunk(state: EngineState, message: EngineMessage): Partial<EngineState> {
  const convId = message.conversation_id as string | undefined;
  if (convId && state.activeSummaryConversationId && convId !== state.activeSummaryConversationId) {
    return {};
  }
  const text = (message.text as string) || "";
  if (message.is_error === true) {
    const cleaned = text.replace(/^\s*\*\*Error:\*\*\s*/i, "").trim();
    return { summaryError: cleaned || "Summary generation failed." };
  }
  return { summaryText: state.summaryText + text };
}

export interface ConversationSlice {
  // Legacy history (client-side)
  history: Conversation[];
  activeConversationId: string | null;

  // EF-029 Summary
  summaryText: string;
  summaryLoading: boolean;
  summaryVisible: boolean;
  summaryWarning: SummaryWarning | null;
  summaryError: string | null;
  activeSummaryRequestId: string | null;
  activeSummaryConversationId: string | null;
  summaryStreamedViaEvent: boolean;

  requestSummary: () => void;
  cancelSummary: () => Promise<void>;
  openSummary: () => void;
  closeSummary: () => void;

  // EF-029 SQLite-Backed Conversation History
  savedConversations: ConversationRecord[];
  selectedConversationDetail: any | null;
  historyLoading: boolean;
  historySearchQuery: string;

  setHistorySearchQuery: (query: string) => void;
  loadHistoryList: () => Promise<void>;
  getConversationDetail: (conversationId: string) => Promise<any | null>;
  clearSelectedConversationDetail: () => void;
  renameConversation: (id: string, name: string) => void;
  deleteConversation: (id: string) => void;
  exportConversation: (id: string) => Promise<any | null>;
  importConversation: (exportData: Record<string, unknown> | string) => Promise<boolean>;
  migrateLegacyHistory: () => Promise<void>;

  saveCurrentSession: () => void;
  loadConversation: (id: string) => void;

  // EF-027 Context Profiles & Pre-Session Setup
  profiles: ContextProfileInfo[];
  activeProfile: ContextProfile | null;
  profilesLoading: boolean;
  profilesError: string | null;
  attachedProfileIds: string[];
  sourceLabels: { mic: string; system: string };
  isSetupOpen: boolean;
  pendingTitle: string;

  loadProfiles: () => Promise<void>;
  getProfileDetail: (profileId: string) => Promise<ContextProfile | null>;
  createProfile: (
    name: string,
    content: string,
    contentFormat?: ProfileContentFormat,
  ) => Promise<ContextProfile | null>;
  updateProfile: (
    profileId: string,
    name: string,
    content: string,
    contentFormat?: ProfileContentFormat,
    expectedRevision?: number,
  ) => Promise<ContextProfile | null>;
  deleteProfile: (profileId: string) => Promise<boolean>;
  setAttachedProfileIds: (profileIds: string[]) => void;
  setSourceLabels: (labels: { mic: string; system: string }) => void;
  openSetup: () => void;
  closeSetup: () => void;
  setPendingTitle: (title: string) => void;
  startSessionWithSetup: () => Promise<void>;
}

export const createConversationSlice =
  (commands: EngineCommands): EngineSliceCreator<ConversationSlice> =>
  (set, get) => ({
    history: [],
    activeConversationId: null,

    // Summary state
    summaryText: "",
    summaryLoading: false,
    summaryVisible: false,
    summaryWarning: null,
    summaryError: null,
    activeSummaryRequestId: null,
    activeSummaryConversationId: null,
    summaryStreamedViaEvent: false,

    // History state
    savedConversations: [],
    selectedConversationDetail: null,
    historyLoading: false,
    historySearchQuery: "",

    // Context profiles state
    profiles: [],
    activeProfile: null,
    profilesLoading: false,
    profilesError: null,
    attachedProfileIds: [],
    sourceLabels: { mic: "Me", system: "Others" },
    isSetupOpen: false,
    pendingTitle: "",

    requestSummary: () => {
      const state = get();
      const conversationId = state.engineConversationId || state.activeConversationId;
      const { socket, connected, entries, config } = state;

      if (!socket || !connected || (entries.length === 0 && !conversationId)) return;

      const requestId = `sum-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      set({
        summaryText: "",
        summaryLoading: true,
        summaryVisible: true,
        summaryWarning: null,
        summaryError: null,
        activeSummaryRequestId: requestId,
        activeSummaryConversationId: conversationId || null,
        summaryStreamedViaEvent: false,
      });

      if (conversationId) {
        const { type } = JSON.parse(
          commands.requestSummary([], publicConfigSnapshot(config)),
        ) as { type: string };
        socket.send(JSON.stringify({ type, conversation_id: conversationId }));
      } else {
        socket.send(commands.requestSummary(entries, config));
      }
    },

    cancelSummary: async () => {
      const { activeSummaryRequestId, activeSummaryConversationId } = get();
      if ((activeSummaryRequestId || activeSummaryConversationId) && isTauriRuntime()) {
        try {
          await forwardCommand({
            type: "summary_cancel",
            ...(activeSummaryRequestId ? { request_id: activeSummaryRequestId } : {}),
            ...(activeSummaryConversationId
              ? { conversation_id: activeSummaryConversationId }
              : {}),
          });
        } catch {}
      }
      set({
        summaryLoading: false,
        activeSummaryRequestId: null,
      });
    },

    openSummary: () => {
      set({ summaryVisible: true });
    },

    closeSummary: () => {
      if (get().summaryLoading) {
        void get().cancelSummary();
      }
      set({ summaryVisible: false, summaryLoading: false });
    },

    setHistorySearchQuery: (query: string) => {
      set({ historySearchQuery: query });
    },

    loadHistoryList: async () => {
      set({ historyLoading: true });
      try {
        const res = await forwardCommand({ type: "conversation_list" });
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
      } catch {
        set({ historyLoading: false });
      }
    },

    getConversationDetail: async (conversationId: string) => {
      try {
        const res = await forwardCommand({
          type: "conversation_get",
          conversation_id: conversationId,
          include_messages: true,
        });
        const first =
          res.messages.find((m) => (m as any).conversation || (m as any).messages) ||
          res.messages[0];
        set({ selectedConversationDetail: first });
        return first;
      } catch {
        return null;
      }
    },

    renameConversation: (id: string, name: string) => {
      const trimmed = name.trim().slice(0, MAX_TITLE_CHARS);
      set((state) => ({
        history: state.history.map((c) => (c.id === id ? { ...c, name: trimmed } : c)),
        savedConversations: state.savedConversations.map((c) =>
          c.conversation_id === id ? { ...c, title: trimmed } : c,
        ),
      }));
      if (isTauriRuntime()) {
        void forwardCommand({
          type: "conversation_rename",
          conversation_id: id,
          title: trimmed,
        }).catch(() => {});
      }
    },

    deleteConversation: (id: string) => {
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
        void forwardCommand({
          type: "conversation_delete",
          conversation_id: id,
        }).catch(() => {});
      }
    },

    clearSelectedConversationDetail: () => {
      set({ selectedConversationDetail: null });
    },

    exportConversation: async (id: string) => {
      try {
        const res = await forwardCommand({
          type: "conversation_export",
          conversation_id: id,
        });
        const first = res.messages.find((m) => (m as any).export !== undefined) || res.messages[0];
        return (first as any)?.export || first;
      } catch {
        return null;
      }
    },

    importConversation: async (exportData: Record<string, unknown> | string) => {
      try {
        const payload =
          typeof exportData === "string" ? (JSON.parse(exportData) as Record<string, unknown>) : exportData;
        const res = await forwardCommand({
          type: "conversation_import",
          export_data: payload,
        });
        await get().loadHistoryList();
        return !res.messages.some((m) => m.type === "error");
      } catch {
        return false;
      }
    },

    migrateLegacyHistory: async () => {
      const { history } = get();
      if (!history || history.length === 0) return;

      for (const item of [...history]) {
        const importId = `legacy-${item.id}`;
        const exportData = {
          conversation_id: item.id,
          title: item.name || `Session ${new Date(item.date).toLocaleString()}`,
          created_at_ms: item.date || Date.now(),
          updated_at_ms: item.date || Date.now(),
          source_labels: { mic: "Me", system: "Others" },
          import_id: importId,
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
            ? [
                {
                  summary_markdown: item.summary,
                  created_at_ms: item.date || Date.now(),
                },
              ]
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

      // Desktop SQLite persists conversations directly on the engine; skip duplicating into
      // localStorage legacy history when a SQLite conversation id is active.
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

    loadConversation: (id) => {
      const { history, running, selectedConversationDetail } = get();
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
          pendingTitle: target.name || "",
        });
        return;
      }

      const detail = selectedConversationDetail;
      const detailId = detail?.conversation?.conversation_id || detail?.conversation_id;
      if (detail && detailId === id) {
        const rawMessages = detail.messages || [];
        const assistByMsgId = new Map<string, any>();
        for (const a of detail.assist_results || []) {
          if (a?.target_message_id) {
            assistByMsgId.set(a.target_message_id, a);
          }
        }
        const loadedEntries: TranscriptEntry[] = rawMessages.map((m: any, idx: number) => {
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
            source:
              m.source === "mic" || m.source === "system" || m.source === "both"
                ? m.source
                : null,
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
        set({
          entries: loadedEntries,
          activeView: "transcript",
          partials: {},
          suggestionLoading: {},
          assistRequests: {},
          summaryText: detail.summary?.summary_markdown || (typeof detail.summary === "string" ? detail.summary : ""),
          summaryVisible: false,
          summaryLoading: false,
          activeConversationId: id,
          engineConversationId: id,
          pendingTitle: detail.conversation?.title || detail.title || "",
          ...(convLabels?.mic || convLabels?.system
            ? {
                sourceLabels: {
                  mic: convLabels.mic || "Me",
                  system: convLabels.system || "Others",
                },
              }
            : {}),
        });
      }
    },

    loadProfiles: async () => {
      set({ profilesLoading: true, profilesError: null });
      try {
        const res = await forwardCommand({ type: "profile_list" });
        const dataMsg = res.messages.find(
          (m) => (m as any).profiles !== undefined || m.type === "profile_list_result",
        );
        const list = (dataMsg as any)?.profiles || (dataMsg as any)?.data?.profiles || [];
        const profiles: ContextProfileInfo[] = list.map((p: any) => ({
          profileId: p.profile_id || p.profileId,
          name: p.name || "",
          contentFormat: p.content_format || p.contentFormat || "markdown",
          revision: p.revision ?? 1,
          contentBytes: p.content_bytes || p.contentBytes || 0,
          updatedAtMs: p.updated_at_ms || p.updatedAtMs || Date.now(),
        }));
        set({ profiles, profilesLoading: false });
      } catch (err: any) {
        set({
          profilesLoading: false,
          profilesError: describeCommandError(err?.code || "bridge_failed"),
        });
      }
    },

    getProfileDetail: async (profileId: string) => {
      try {
        const res = await forwardCommand({ type: "profile_get", profile_id: profileId });
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        if (!p) return null;
        const profile: ContextProfile = {
          profileId: p.profile_id || p.profileId,
          name: p.name,
          content: p.content || "",
          contentFormat: p.content_format || p.contentFormat || "markdown",
          revision: p.revision ?? 1,
          createdAtMs: p.created_at_ms || p.createdAtMs || Date.now(),
          updatedAtMs: p.updated_at_ms || p.updatedAtMs || Date.now(),
        };
        set({ activeProfile: profile });
        return profile;
      } catch {
        return null;
      }
    },

    createProfile: async (name: string, content: string, contentFormat = "markdown") => {
      const trimmedName = name.trim().slice(0, MAX_PROFILE_NAME_CHARS);
      if (!trimmedName) return null;
      if (utf8ByteLength(content) > MAX_PROFILE_CONTENT_BYTES) {
        set({ profilesError: "Profile content exceeds 64 KiB limit" });
        return null;
      }
      try {
        const res = await forwardCommand({
          type: "profile_create",
          name: trimmedName,
          content,
          content_format: contentFormat,
        });
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        await get().loadProfiles();
        if (p) {
          return {
            profileId: p.profile_id || p.profileId,
            name: p.name,
            content: p.content || content,
            contentFormat: p.content_format || p.contentFormat || contentFormat,
            revision: p.revision ?? 1,
            createdAtMs: p.created_at_ms || Date.now(),
            updatedAtMs: p.updated_at_ms || Date.now(),
          };
        }
        return null;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return null;
      }
    },

    updateProfile: async (
      profileId: string,
      name: string,
      content: string,
      contentFormat = "markdown",
      expectedRevision?: number,
    ) => {
      const trimmedName = name.trim().slice(0, MAX_PROFILE_NAME_CHARS);
      if (utf8ByteLength(content) > MAX_PROFILE_CONTENT_BYTES) {
        set({ profilesError: "Profile content exceeds 64 KiB limit" });
        return null;
      }
      try {
        const payload: Record<string, unknown> = {
          type: "profile_update",
          profile_id: profileId,
          name: trimmedName,
          content,
          content_format: contentFormat,
        };
        if (expectedRevision !== undefined) {
          payload.expected_revision = expectedRevision;
        }
        const res = await forwardCommand(payload as any);
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        await get().loadProfiles();
        if (p) {
          return {
            profileId: p.profile_id || p.profileId,
            name: p.name,
            content: p.content || content,
            contentFormat: p.content_format || p.contentFormat || contentFormat,
            revision: p.revision ?? (expectedRevision ? expectedRevision + 1 : 1),
            createdAtMs: p.created_at_ms || Date.now(),
            updatedAtMs: p.updated_at_ms || Date.now(),
          };
        }
        return null;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return null;
      }
    },

    deleteProfile: async (profileId: string) => {
      try {
        await forwardCommand({ type: "profile_delete", profile_id: profileId });
        set((state) => ({
          profiles: state.profiles.filter((p) => p.profileId !== profileId),
          attachedProfileIds: state.attachedProfileIds.filter((id) => id !== profileId),
          activeProfile: state.activeProfile?.profileId === profileId ? null : state.activeProfile,
        }));
        return true;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return false;
      }
    },

    setAttachedProfileIds: (attachedProfileIds) => {
      set({ attachedProfileIds: attachedProfileIds.slice(0, MAX_PROFILES_PER_SESSION) });
    },

    setSourceLabels: (sourceLabels) =>
      set({
        sourceLabels: {
          mic: sourceLabels.mic.slice(0, MAX_LABEL_CHARS),
          system: sourceLabels.system.slice(0, MAX_LABEL_CHARS),
        },
      }),

    openSetup: () => {
      set({ isSetupOpen: true });
    },

    closeSetup: () => set({ isSetupOpen: false }),

    setPendingTitle: (title) => set({ pendingTitle: title.slice(0, MAX_TITLE_CHARS) }),

    startSessionWithSetup: async () => {
      const { pendingTitle, sourceLabels, attachedProfileIds } = get();
      const title = pendingTitle.trim() || "Session " + new Date().toLocaleString();
      try {
        const convRes = await forwardCommand({
          type: "conversation_create",
          title,
          source_labels: sourceLabels,
        });
        const convData = convRes.messages.find(
          (m) => (m as any).conversation_id || (m as any).conversation,
        );
        const convId =
          (convData as any)?.conversation_id || (convData as any)?.conversation?.conversation_id;
        if (convId && attachedProfileIds.length > 0) {
          await forwardCommand({
            type: "profile_attach",
            conversation_id: convId,
            profile_ids: attachedProfileIds,
          });
        }
        if (convId) {
          set({
            activeConversationId: convId,
            engineConversationId: convId,
            isSetupOpen: false,
          });
        } else {
          set({ isSetupOpen: false });
        }
      } catch (err: any) {
        set({
          isSetupOpen: false,
          ...(isTauriRuntime()
            ? { appError: `Session setup failed: ${describeCommandError(err?.code || "bridge_failed")}` }
            : {}),
        });
      }
      get().startPipeline();
    },
  });
