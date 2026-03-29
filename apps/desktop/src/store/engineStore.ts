import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export interface SuggestionOption {
  strategy: string;
  text: string;
}

export interface SuggestionResult {
  options?: SuggestionOption[];
  error?: string;
}

export interface TranscriptEntry {
  id: string;
  text: string;
  translation: string | null;
  isFinal: boolean;
  timestamp: number;
  source?: "mic" | "system" | "both" | null;
  suggestions?: SuggestionResult; // persisted with the entry
}

export interface Conversation {
  id: string;
  name: string;
  date: number;
  entries: TranscriptEntry[];
  summary?: string; // persisted meeting summary
}

export interface AudioDeviceInfo {
  id: string;
  name: string;
}

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
  llmApiKey: string;
  llmModel: string;
  stealthMode: boolean;
}

export type AppView = "transcript" | "settings" | "history";

interface EngineState {
  // Connection
  connected: boolean;
  socket: WebSocket | null;

  // Pipeline
  running: boolean;
  isToggling: boolean;

  // Transcript
  entries: TranscriptEntry[];
  partials: Record<string, {
    text: string;
    translation: string | null;
    source: "mic" | "system" | "both" | null;
  }>;

  // Translation status
  activeTranslationBackend: string | null;

  // Downloads
  downloading: boolean;
  downloadModel: string;
  downloadPercent: number;

  // History
  history: Conversation[];
  activeConversationId: string | null;

  // UI
  activeView: AppView;
  theme: "dark" | "light";

  // Config
  config: EngineConfig;

  // Audio device lists
  availableMics: AudioDeviceInfo[];
  availableSpeakers: AudioDeviceInfo[];

  // LLM — suggestion loading (ephemeral, not persisted)
  suggestionLoading: Record<string, boolean>;

  // LLM — summary (summaryText is the live buffer / current session summary)
  summaryText: string;
  summaryLoading: boolean;
  summaryVisible: boolean;

  // Actions
  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  togglePipeline: () => void;
  clearTranscript: () => void;
  setActiveView: (view: AppView) => void;
  setTheme: (theme: "dark" | "light") => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
  requestDeviceList: () => void;

  // LLM Actions
  requestSuggestion: (entryId: string, targetText: string, context: string[]) => void;
  requestSummary: () => void;
  openSummary: () => void;
  closeSummary: () => void;

  // History Actions
  renameConversation: (id: string, name: string) => void;
  deleteConversation: (id: string) => void;
  saveCurrentSession: () => void;
  loadConversation: (id: string) => void;
}

const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "en",
  device: "auto",
  translationEnabled: true,
  translationBackend: "online",
  translationModel: "",
  sourceLang: "en",
  targetLang: "vi",
  vadEnabled: true,
  audioSource: "microphone",
  micDeviceId: "",
  speakerDeviceId: "",
  llmEnabled: false,
  llmProviderUrl: "https://openrouter.ai/api/v1",
  llmApiKey: "",
  llmModel: "openai/gpt-4o-mini",
  stealthMode: false,
};

let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let idCounter = 0;

function nextId(): string {
  idCounter += 1;
  return `t-${Date.now()}-${idCounter}`;
}

export const useEngineStore = create<EngineState>()(
  persist(
    (set, get) => ({
      connected: false,
      socket: null,
      running: false,
      isToggling: false,
      entries: [],
      partials: {},
      activeTranslationBackend: null,
      downloading: false,
      downloadModel: "",
      downloadPercent: 0,
      history: [],
      activeView: "transcript",
      theme: window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark",
      activeConversationId: null,
      config: { ...DEFAULT_CONFIG },
      availableMics: [],
      availableSpeakers: [],
      suggestionLoading: {},
      summaryText: "",
      summaryLoading: false,
      summaryVisible: false,

      connect: () => {
        const { config, socket } = get();
        if (socket && socket.readyState === WebSocket.OPEN) return;

        const url = `ws://${config.host}:${config.port}`;
        const ws = new WebSocket(url);

        ws.onopen = () => {
          set({ connected: true, socket: ws });
          if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
          }
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            handleMessage(message, set, get);
          } catch {
            // Ignore malformed messages
          }
        };

        ws.onclose = () => {
          set({
            connected: false,
            socket: null,
            running: false,
            isToggling: false,
            activeTranslationBackend: null,
            downloading: false,
            downloadModel: "",
            downloadPercent: 0,
          });
          reconnectTimer = setTimeout(() => {
            get().connect();
          }, 3000);
        };

        ws.onerror = () => {
          ws.close();
        };

        set({ socket: ws });
      },

      disconnect: () => {
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        const { socket } = get();
        if (socket) {
          socket.close();
        }
        set({
          connected: false,
          socket: null,
          running: false,
          isToggling: false,
          activeTranslationBackend: null,
          downloading: false,
          downloadModel: "",
          downloadPercent: 0,
        });
      },

      startPipeline: () => {
        const { socket, connected, config } = get();
        if (!socket || !connected) return;

        set({ isToggling: true, activeView: "transcript" });

        socket.send(
          JSON.stringify({
            type: "start",
            config: {
              "asr.model_size": config.modelSize,
              "asr.language": config.language,
              "asr.device": config.device,
              "translation.enabled": config.translationEnabled,
              "translation.backend": config.translationBackend,
              "translation.model": config.translationModel || undefined,
              "translation.source_lang": config.sourceLang,
              "translation.target_lang": config.targetLang,
              "vad.enabled": config.vadEnabled,
              "audio.source": config.audioSource,
              "audio.mic_device_id": config.micDeviceId || undefined,
              "audio.speaker_device_id": config.speakerDeviceId || undefined,
              "llm.enabled": config.llmEnabled,
              "llm.provider_url": config.llmProviderUrl || undefined,
              "llm.api_key": config.llmApiKey || undefined,
              "llm.model": config.llmModel || undefined,
            },
          }),
        );
      },

      stopPipeline: () => {
        const { socket, connected } = get();
        if (!socket || !connected) return;

        set({ isToggling: true });
        socket.send(JSON.stringify({ type: "stop" }));
      },

      togglePipeline: () => {
        const { running } = get();
        if (running) {
          get().stopPipeline();
        } else {
          get().startPipeline();
        }
      },

      clearTranscript: () => {
        set({
          entries: [],
          partials: {},
          suggestionLoading: {},
          summaryText: "",
          summaryVisible: false,
          activeConversationId: null,
        });
      },

      setActiveView: (view) => {
        set({ activeView: view });
      },

      setTheme: (theme) => {
        set({ theme });
      },

      updateConfig: (partial) => {
        set((state) => ({
          config: { ...state.config, ...partial },
        }));
      },

      requestDeviceList: () => {
        const { socket, connected } = get();
        if (!socket || !connected) return;
        socket.send(JSON.stringify({ type: "list_devices" }));
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

        socket.send(JSON.stringify({
          type: "request_suggestion",
          entry_id: entryId,
          target_text: targetText,
          context,
          llm_config: {
            api_key: config.llmApiKey,
            model: config.llmModel,
            provider_url: config.llmProviderUrl || undefined,
          },
        }));
      },

      requestSummary: () => {
        const { socket, connected, entries, config } = get();
        if (!socket || !connected || entries.length === 0) return;

        set({ summaryText: "", summaryLoading: true, summaryVisible: true });

        socket.send(JSON.stringify({
          type: "request_summary",
          entries: entries.map((e) => ({ text: e.text, source: e.source ?? "speaker" })),
          llm_config: {
            api_key: config.llmApiKey,
            model: config.llmModel,
            provider_url: config.llmProviderUrl || undefined,
          },
        }));
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
    }),
    {
      name: "echoflux-storage",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        config: state.config,
        history: state.history,
        theme: state.theme,
      }),
    },
  ),
);

function handleMessage(
  message: Record<string, unknown>,
  set: (partial: Partial<EngineState> | ((state: EngineState) => Partial<EngineState>)) => void,
  get: () => EngineState,
) {
  switch (message.type) {
    case "partial": {
      const newPartial = (message.text as string) || "";
      const streamId = (message.source as string) || "default";
      const currentPartial = get().partials[streamId]?.text || "";

      const extra: Partial<EngineState> = {};
      if (message.translation_backend) {
        extra.activeTranslationBackend = message.translation_backend as string;
      }

      if (
        newPartial.length >= currentPartial.length ||
        newPartial.length >= currentPartial.length * 0.6 ||
        currentPartial === ""
      ) {
        set((state) => ({
          partials: {
            ...state.partials,
            [streamId]: {
              text: newPartial,
              translation: (message.translation as string) ?? null,
              source: (message.source as TranscriptEntry["source"]) ?? null,
            },
          },
          ...extra,
        }));
      }
      break;
    }

    case "final": {
      const finalText = (message.text as string) || "";
      const streamId = (message.source as string) || "default";
      if (!finalText.trim()) break;

      const update: Partial<EngineState> = {};
      if (message.translation_backend) {
        update.activeTranslationBackend = message.translation_backend as string;
      }

      set((state) => {
        const newPartials = { ...state.partials };
        delete newPartials[streamId];

        return {
          ...update,
          entries: [
            ...state.entries,
            {
              id: (message.entry_id as string) || nextId(),
              text: finalText,
              translation: (message.translation as string) ?? null,
              isFinal: true,
              timestamp: message.timestamp as number,
              source: (message.source as TranscriptEntry["source"]) ?? null,
            },
          ],
          partials: newPartials,
        };
      });
      break;
    }

    case "translation_update": {
      const translation = message.translation as string;
      const entryId = message.entry_id as string | undefined;
      const sourceText = message.source_text as string;

      if (!translation) break;

      // ES2021-compatible reverse linear search helper
      const findLast = (arr: TranscriptEntry[], pred: (e: TranscriptEntry) => boolean) => {
        for (let i = arr.length - 1; i >= 0; i--) {
          if (pred(arr[i])) return i;
        }
        return -1;
      };

      set((state) => {
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

        if (foundIndex !== -1) {
          newEntries[foundIndex] = { ...newEntries[foundIndex], translation };
          return { entries: newEntries };
        }

        // If no match at all, discard — do NOT blindly attach to the last entry
        return {};
      });
      break;
    }

    case "download_progress": {
      if (message.percent === 100) {
        set({ downloading: false, downloadModel: "", downloadPercent: 0 });
      } else {
        set({
          downloading: true,
          downloadModel: (message.model as string) || "AI Model",
          downloadPercent: (message.percent as number) || 0,
        });
      }
      break;
    }

    case "status":
      if (message.status === "started") {
        set({
          running: true,
          isToggling: false,
          entries: [],
          partials: {},
          suggestionLoading: {},
          summaryText: "",
          summaryVisible: false,
          activeTranslationBackend: null,
          downloading: false,
          downloadModel: "",
          downloadPercent: 0,
          activeConversationId: null,
        });
      } else if (message.status === "stopped") {
        get().saveCurrentSession();
        set({
          running: false,
          isToggling: false,
          partials: {},
          downloading: false,
          downloadModel: "",
          downloadPercent: 0,
        });
      }
      break;

    case "error":
      console.error("Engine error:", message.message);
      set({ isToggling: false });
      break;

    case "devices_list":
      set({
        availableMics: (message.microphones as AudioDeviceInfo[]) ?? [],
        availableSpeakers: (message.speakers as AudioDeviceInfo[]) ?? [],
      });
      break;

    case "suggestion_result": {
      const entryId = message.entry_id as string;
      if (!entryId) break;

      set((state) => {
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
      });
      get().saveCurrentSession();
      break;
    }

    case "llm_chunk": {
      const text = (message.text as string) || "";
      set((state) => ({ summaryText: state.summaryText + text }));
      break;
    }

    case "llm_done": {
      set({ summaryLoading: false });
      get().saveCurrentSession();
      break;
    }
  }
}
