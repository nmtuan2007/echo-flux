import type { EngineSliceCreator, EngineState } from "./engineStore";
import type { EngineCommands } from "./sessionSlice";
import { publicConfigSnapshot } from "./settingsSlice";
import {
  describeCommandError,
  errorCodeOf,
  forwardCommand,
  isTauriRuntime,
  testConnectionService,
  type StartCredentialRef,
} from "../lib/engineBridge";
import { LEGACY_CREDENTIALS } from "../lib/secureStore";
import { createHistoryActions, sanitizeHistory, type HistorySliceFields } from "./historySlice";
import { createProfileActions, utf8ByteLength, type ProfilesSliceFields } from "./profilesSlice";
import type { AskTurn, EngineMessage, SummaryWarning } from "./types";

export { sanitizeHistory, utf8ByteLength };

export function applyLlmChunk(state: EngineState, message: EngineMessage): Partial<EngineState> {
  const convId = message.conversation_id as string | undefined;
  if (convId && state.activeSummaryConversationId && convId !== state.activeSummaryConversationId) {
    return {};
  }
  const text = (message.text as string) || "";
  if (message.is_error === true) {
    if (message.error_code === "cancelled" && state.summaryCancelledByUser) {
      return {};
    }
    const cleaned = text.replace(/^\s*\*\*Error:\*\*\s*/i, "").trim();
    return { summaryError: cleaned || "Summary generation failed." };
  }
  return { summaryText: state.summaryText + text };
}

export function describeAssistantSetupError(code: string | null | undefined): string {
  switch (code) {
    case "missing_credential":
      return "Assistant API key is missing. Save your key in Settings → Assistant.";
    case "invalid_endpoint":
    case "missing_endpoint":
      return "Assistant API URL is missing or invalid. Check Settings → Assistant.";
    case "missing_model":
      return "Assistant model name is required. Check Settings → Assistant.";
    case "unsupported_protocol":
      return "Selected Assistant provider protocol is unsupported.";
    case "assistant_start_failed":
      return "Assistant failed to initialize. Check endpoint, model, and key in Settings.";
    default:
      return "Assistant is not configured or credentials are missing. Check Assistant settings.";
  }
}

export interface ConversationSlice extends HistorySliceFields, ProfilesSliceFields {
  // Summary & Quick Ask
  summaryText: string;
  summaryLoading: boolean;
  summaryVisible: boolean;
  summaryWarning: SummaryWarning | null;
  summaryError: string | null;
  summaryCancelledByUser: boolean;
  activeSummaryRequestId: string | null;
  activeSummaryConversationId: string | null;
  summaryStreamedViaEvent: boolean;
  askAnswer: string;
  askHistory: AskTurn[];
  askLoading: boolean;
  askError: string | null;
  lastAssistantErrorCode: string | null;

  syncAssistantService: () => Promise<boolean>;
  requestSummary: () => void;
  cancelSummary: () => Promise<void>;
  openSummary: () => void;
  closeSummary: () => void;
  askMeeting: (question: string, targetConversationId?: string) => Promise<void>;
  clearAsk: (targetConversationId?: string) => void;
}

export const createConversationSlice =
  (commands: EngineCommands): EngineSliceCreator<ConversationSlice> =>
  (set, get) => {
    let lastSyncedAssistantKey: string | null = null;

    return {
      // Summary state
      summaryText: "",
      summaryLoading: false,
      summaryVisible: false,
      summaryWarning: null,
      summaryError: null,
      summaryCancelledByUser: false,
      activeSummaryRequestId: null,
      activeSummaryConversationId: null,
      summaryStreamedViaEvent: false,
      askAnswer: "",
      askHistory: [],
      askLoading: false,
      askError: null,
      lastAssistantErrorCode: null,

      syncAssistantService: async () => {
        const { connected, services, secretStatus, engineGeneration } = get();
        if (!connected || !isTauriRuntime()) return false;
        const assistantCfg = services?.assistant;
        if (!assistantCfg?.enabled) {
          lastSyncedAssistantKey = null;
          set({ lastAssistantErrorCode: "assistant_disabled" });
          return false;
        }
        const syncKey = JSON.stringify({
          gen: engineGeneration,
          service: testConnectionService(assistantCfg),
          legacyKey: secretStatus.llmApiKey,
        });
        if (syncKey === lastSyncedAssistantKey) return true;

        const credentials: Record<string, StartCredentialRef> = {};
        for (const [slot, ref] of Object.entries(assistantCfg.credentialRefs || {})) {
          if (ref && ref.configId) {
            credentials[`assistant.${slot}`] = {
              capability: "assistant",
              configId: ref.configId,
              slot: ref.slot || slot,
            };
          }
        }
        if (!credentials["assistant.api_key"] && secretStatus.llmApiKey) {
          credentials["assistant.api_key"] = { ...LEGACY_CREDENTIALS.llmApiKey };
        }

        try {
          const res = await forwardCommand(
            {
              type: "configure_assistant",
              service: testConnectionService(assistantCfg),
            },
            credentials,
          );
          const reply = res.messages.find(
            (m) => (m as { configured?: unknown }).configured !== undefined,
          ) as { configured?: unknown; error_code?: unknown } | undefined;
          if (reply?.configured === true) {
            lastSyncedAssistantKey = syncKey;
            set({ lastAssistantErrorCode: null });
            return true;
          }
          lastSyncedAssistantKey = null;
          const code =
            typeof reply?.error_code === "string" ? reply.error_code : "assistant_start_failed";
          set({ lastAssistantErrorCode: code });
          return false;
        } catch (err) {
          lastSyncedAssistantKey = null;
          set({ lastAssistantErrorCode: errorCodeOf(err) });
          return false;
        }
      },

      requestSummary: () => {
        const state = get();
        const conversationId = state.engineConversationId || state.activeConversationId;
        const { socket, connected, entries, config, services, consent } = state;

        if (!socket || !connected || (entries.length === 0 && !conversationId)) return;

        const assistantSvc = services?.assistant;
        if (
          isTauriRuntime() &&
          assistantSvc?.enabled &&
          assistantSvc.providerId !== "local-openai-compatible" &&
          !consent?.summary
        ) {
          set({
            appError:
              "Cloud consent is required for Meeting Summary. Grant permission in Settings before summarizing.",
          });
          return;
        }

        const requestId = `sum-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        set({
          summaryText: "",
          summaryLoading: true,
          summaryVisible: true,
          summaryWarning: null,
          summaryError: null,
          summaryCancelledByUser: false,
          activeSummaryRequestId: requestId,
          activeSummaryConversationId: conversationId || null,
          summaryStreamedViaEvent: false,
        });

        const sendSummary = () => {
          const current = get();
          if (!current.socket || !current.connected) return;
          if (conversationId) {
            const { type } = JSON.parse(
              commands.requestSummary([], publicConfigSnapshot(current.config)),
            ) as { type: string };
            current.socket.send(JSON.stringify({ type, conversation_id: conversationId }));
          } else {
            current.socket.send(commands.requestSummary(entries, config));
          }
        };

        if (conversationId && isTauriRuntime() && state.services?.assistant?.enabled) {
          void get()
            .syncAssistantService()
            .then((ok) => {
              if (!ok) {
                set({
                  summaryLoading: false,
                  summaryError: describeAssistantSetupError(get().lastAssistantErrorCode),
                });
                return;
              }
              sendSummary();
            });
        } else {
          sendSummary();
        }
      },

      cancelSummary: async () => {
        const { activeSummaryRequestId, activeSummaryConversationId } = get();
        set({
          summaryLoading: false,
          summaryError: null,
          summaryCancelledByUser: true,
          activeSummaryRequestId: null,
        });
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
      },

      openSummary: () => set({ summaryVisible: true }),

      closeSummary: () => {
        // Hiding the modal preserves an in-flight summary; users cancel explicitly via Stop.
        set({ summaryVisible: false });
      },

      clearAsk: (targetConversationId) => {
        const state = get();
        const convId =
          targetConversationId || state.engineConversationId || state.activeConversationId;
        set({ askAnswer: "", askHistory: [], askError: null, askLoading: false });
        if (convId && isTauriRuntime()) {
          void forwardCommand({
            type: "assistant_ask_clear",
            conversation_id: convId,
          }).catch(() => {});
        }
        const detail = state.selectedConversationDetail;
        const detailId = detail?.conversation?.conversation_id || detail?.conversation_id;
        if (detail && detailId === convId) {
          set({ selectedConversationDetail: { ...detail, ask_turns: [] } });
        }
      },

      askMeeting: async (question, targetConversationId) => {
        const trimmed = question.trim();
        if (!trimmed) return;
        const state = get();
        const conversationId =
          targetConversationId || state.engineConversationId || state.activeConversationId;
        if (!conversationId || !state.connected || !isTauriRuntime()) {
          set({ askError: "Start or load a saved conversation with a connected engine to ask AI." });
          return;
        }
        const assistantSvc = state.services?.assistant;
        if (
          assistantSvc?.enabled &&
          assistantSvc.providerId !== "local-openai-compatible" &&
          !state.consent?.assistant
        ) {
          set({
            askError: "Cloud consent is required for AI Assistant. Grant permission in Settings.",
          });
          return;
        }

        set({ askLoading: true, askError: null });
        if (assistantSvc?.enabled) {
          const ok = await get().syncAssistantService();
          if (!ok) {
            set({
              askLoading: false,
              askError: describeAssistantSetupError(get().lastAssistantErrorCode),
            });
            return;
          }
        }
        try {
          const res = await forwardCommand({
            type: "assistant_ask",
            conversation_id: conversationId,
            question: trimmed,
          });
          const reply = res.messages.find(
            (item) => typeof (item as Record<string, unknown>).answer_markdown === "string",
          ) as Record<string, unknown> | undefined;
          const answer =
            typeof reply?.answer_markdown === "string" ? reply.answer_markdown.trim() : "";
          if (!answer) {
            set({ askLoading: false, askError: "Assistant returned an empty answer." });
            return;
          }
          const askId =
            typeof reply?.ask_id === "string" && reply.ask_id
              ? reply.ask_id
              : `${Date.now()}-${get().askHistory.length}`;
          set((prev) => {
            const newTurn = {
              id: askId,
              question: trimmed,
              answer,
              createdAtMs: Date.now(),
            };
            const detail = prev.selectedConversationDetail;
            const detailId = detail?.conversation?.conversation_id || detail?.conversation_id;
            const updatedDetail =
              detail && detailId === conversationId
                ? {
                    ...detail,
                    ask_turns: [
                      ...(Array.isArray(detail.ask_turns) ? detail.ask_turns : []),
                      {
                        ask_id: askId,
                        question: trimmed,
                        answer_markdown: answer,
                        created_at_ms: newTurn.createdAtMs,
                      },
                    ],
                  }
                : detail;
            return {
              askAnswer: answer,
              askHistory: [...prev.askHistory, newTurn],
              askLoading: false,
              askError: null,
              selectedConversationDetail: updatedDetail,
            };
          });
        } catch (err) {
          const code = errorCodeOf(err);
          set({
            askLoading: false,
            askError: `Quick Ask failed: ${describeCommandError(code)} (${code})`,
          });
        }
      },

      ...createHistoryActions(set, get),
      ...createProfileActions(set, get),
    };
  };
