import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export interface TranscriptEntry {
  id: string;
  text: string;
  translation: string | null;
  isFinal: boolean;
  timestamp: number;
}

export interface Conversation {
  id: string;
  name: string;
  date: number;
  entries: TranscriptEntry[];
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
  partialText: string;
  partialTranslation: string | null;

  // Translation status
  activeTranslationBackend: string | null;

  // History
  history: Conversation[];

  // UI
  activeView: AppView;

  // Config
  config: EngineConfig;

  // Actions
  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  clearTranscript: () => void;
  setActiveView: (view: AppView) => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;

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
      partialText: "",
      partialTranslation: null,
      activeTranslationBackend: null,
      history: [],
      activeView: "transcript",
      config: { ...DEFAULT_CONFIG },

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

      clearTranscript: () => {
        set({ entries: [], partialText: "", partialTranslation: null });
      },

      setActiveView: (view) => {
        set({ activeView: view });
      },

      updateConfig: (partial) => {
        set((state) => ({
          config: { ...state.config, ...partial },
        }));
      },

      saveCurrentSession: () => {
        const { entries } = get();
        if (entries.length === 0) return;

        const timestamp = Date.now();
        const dateStr = new Date(timestamp).toLocaleString();

        let name = "Session " + dateStr;
        if (entries[0] && entries[0].text) {
          name = entries[0].text.slice(0, 30) + (entries[0].text.length > 30 ? "..." : "");
        }

        const conversation: Conversation = {
          id: `conv-${timestamp}`,
          name: name,
          date: timestamp,
          entries: [...entries],
        };

        set((state) => ({
          history: [conversation, ...state.history],
        }));
      },

      renameConversation: (id, name) => {
        set((state) => ({
          history: state.history.map((c) => (c.id === id ? { ...c, name } : c)),
        }));
      },

      deleteConversation: (id) => {
        set((state) => ({
          history: state.history.filter((c) => c.id !== id),
        }));
      },

      loadConversation: (id) => {
        const { history, running } = get();
        if (running) return;

        const target = history.find((c) => c.id === id);
        if (target) {
          set({
            entries: [...target.entries],
            activeView: "transcript",
            partialText: "",
            partialTranslation: null,
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
      const currentPartial = get().partialText;

      // Update active backend if present
      const extra: Partial<EngineState> = {};
      if (message.translation_backend) {
        extra.activeTranslationBackend = message.translation_backend as string;
      }

      // Simple heuristic to prevent flickering: update only if longer or significant change
      if (
        newPartial.length >= currentPartial.length ||
        newPartial.length >= currentPartial.length * 0.6 ||
        currentPartial === ""
      ) {
        set({
          partialText: newPartial,
          partialTranslation: (message.translation as string) ?? null,
          ...extra,
        });
      }
      break;
    }

    case "final": {
      const finalText = (message.text as string) || "";
      if (!finalText.trim()) break;

      const update: Partial<EngineState> = {};
      if (message.translation_backend) {
        update.activeTranslationBackend = message.translation_backend as string;
      }

      set((state) => ({
        ...update,
        entries: [
          ...state.entries,
          {
            id: (message.entry_id as string) || nextId(),
            text: finalText,
            translation: (message.translation as string) ?? null,
            isFinal: true,
            timestamp: message.timestamp as number,
          },
        ],
        partialText: "",
        partialTranslation: null,
      }));
      break;
    }

    case "translation_update": {
      // Async translation update for a final entry
      const translation = message.translation as string;
      const sourceText = message.source_text as string;

      if (!translation) break;

      set((state) => {
        // Strategy: Find the most recent entry with matching source text that has no translation
        // or just update the last one if source text matches approximately.
        // For robustness, we search from end to start.

        const newEntries = [...state.entries];
        let foundIndex = -1;

        for (let i = newEntries.length - 1; i >= 0; i--) {
          if (
            newEntries[i].text === sourceText ||
            (sourceText && newEntries[i].text.includes(sourceText))
          ) {
            foundIndex = i;
            break;
          }
        }

        if (foundIndex !== -1) {
          newEntries[foundIndex] = {
            ...newEntries[foundIndex],
            translation: translation,
          };
          return { entries: newEntries };
        }

        // If not found (e.g. accumulator merged multiple entries),
        // fallback: Update the very last entry if it has no translation
        const lastIdx = newEntries.length - 1;
        if (lastIdx >= 0 && !newEntries[lastIdx].translation) {
          newEntries[lastIdx] = {
            ...newEntries[lastIdx],
            translation: translation,
          };
          return { entries: newEntries };
        }

        return {};
      });
      break;
    }

    case "status":
      if (message.status === "started") {
        set({
          running: true,
          isToggling: false,
          entries: [],
          partialText: "",
          partialTranslation: null,
          activeTranslationBackend: null,
        });
      } else if (message.status === "stopped") {
        get().saveCurrentSession();
        set({
          running: false,
          isToggling: false,
          partialText: "",
          partialTranslation: null,
        });
      }
      break;

    case "error":
      console.error("Engine error:", message.message);
      set({ isToggling: false });
      break;
  }
}
