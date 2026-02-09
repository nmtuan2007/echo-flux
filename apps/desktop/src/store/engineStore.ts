import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

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
  // Translation config
  translationEnabled: boolean;
  translationBackend: "marian" | "online";
  sourceLang: string;
  targetLang: string;
  // VAD config
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
  translationBackend: "marian",
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
          set({ connected: false, socket: null, running: false, isToggling: false });
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
        set({ connected: false, socket: null, running: false, isToggling: false });
      },

      startPipeline: () => {
        const { socket, connected, config } = get();
        if (!socket || !connected) return;

        // Ensure we are on the transcript view when starting
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
        
        // Generate a name from the first entry, or default
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
        if (running) {
          console.warn("Cannot load conversation while engine is running");
          return;
        }

        const target = history.find((c) => c.id === id);
        if (target) {
          set({
            entries: [...target.entries],
            activeView: "transcript",
            partialText: "",
            partialTranslation: null
          });
        }
      }
    }),
    {
      name: "echoflux-storage",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        config: state.config,
        history: state.history,
      }),
    }
  )
);

function handleMessage(
  message: Record<string, unknown>,
  set: (partial: Partial<EngineState> | ((state: EngineState) => Partial<EngineState>)) => void,
  get: () => EngineState,
) {
  switch (message.type) {
    case "partial":
      set({
        partialText: message.text as string,
        partialTranslation: (message.translation as string) ?? null,
      });
      break;

    case "final":
      set((state) => ({
        entries: [
          ...state.entries,
          {
            id: nextId(),
            text: message.text as string,
            translation: (message.translation as string) ?? null,
            isFinal: true,
            timestamp: message.timestamp as number,
          },
        ],
        partialText: "",
        partialTranslation: null,
      }));
      break;

    case "status":
      if (message.status === "started") {
        set({ running: true, isToggling: false });
      } else if (message.status === "stopped") {
        // Auto-save on stop
        get().saveCurrentSession();
        set({ running: false, isToggling: false });
      }
      break;

    case "error":
      console.error("Engine error:", message.message);
      set({ isToggling: false });
      break;
  }
}
