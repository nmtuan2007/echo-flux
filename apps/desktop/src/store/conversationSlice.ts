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

  syncAssistantService: () => Promise<boolean>;
  requestSummary: () => void;
  cancelSummary: () => Promise<void>;
  openSummary: () => void;
  closeSummary: () => void;
  askMeeting: (question: string) => Promise<void>;
  clearAsk: () => void;
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

      syncAssistantService: async () => {
        const { connected, services, secretStatus, engineGeneration } = get();
        if (!connected || !isTauriRuntime()) return false;
        const assistantCfg = services?.assistant;
        if (!assistantCfg?.enabled) {
          lastSyncedAssistantKey = null;
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
          const configured = res.messages.some(
            (m) => (m as { configured?: unknown }).configured === true,
          );
          if (configured) {
            lastSyncedAssistantKey = syncKey;
            return true;
          }
          lastSyncedAssistantKey = null;
          return false;
        } catch {
          lastSyncedAssistantKey = null;
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
                  summaryError:
                    "Assistant is not configured or credentials are missing. Check Assistant settings.",
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

      clearAsk: () => set({ askAnswer: "", askHistory: [], askError: null, askLoading: false }),

      askMeeting: async (question) => {
        const trimmed = question.trim();
        if (!trimmed) return;
        const state = get();
        const conversationId = state.engineConversationId || state.activeConversationId;
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
              askError: "Assistant is not configured or credentials are missing.",
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
          set((prev) => ({
            askAnswer: answer,
            askHistory: [
              ...prev.askHistory,
              {
                id: `${Date.now()}-${prev.askHistory.length}`,
                question: trimmed,
                answer,
              },
            ],
            askLoading: false,
            askError: null,
          }));
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
