import { create } from "zustand";

export interface TranscriptEntry {
  id: string;
  text: string;
  translation: string | null;
  isFinal: boolean;
  timestamp: number;
}

export interface EngineConfig {
  host: string;
  port: number;
  modelSize: string;
  language: string;
  device: string;
  translationEnabled: boolean;
  sourceLang: string;
  targetLang: string;
  vadEnabled: boolean;
}

interface EngineState {
  // Connection
  connected: boolean;
  socket: WebSocket | null;

  // Pipeline
  running: boolean;

  // Transcript
  entries: TranscriptEntry[];
  partialText: string;
  partialTranslation: string | null;

  // UI
  settingsOpen: boolean;

  // Config
  config: EngineConfig;

  // Actions
  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  clearTranscript: () => void;
  toggleSettings: () => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
}

const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "en",
  device: "auto",
  translationEnabled: false,
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

export const useEngineStore = create<EngineState>((set, get) => ({
  connected: false,
  socket: null,
  running: false,
  entries: [],
  partialText: "",
  partialTranslation: null,
  settingsOpen: false,
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
      set({ connected: false, socket: null, running: false });
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
    set({ connected: false, socket: null, running: false });
  },

  startPipeline: () => {
    const { socket, connected, config } = get();
    if (!socket || !connected) return;

    socket.send(
      JSON.stringify({
        type: "start",
        config: {
          "asr.model_size": config.modelSize,
          "asr.language": config.language,
          "asr.device": config.device,
          "translation.enabled": config.translationEnabled,
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

    socket.send(JSON.stringify({ type: "stop" }));
  },

  clearTranscript: () => {
    set({ entries: [], partialText: "", partialTranslation: null });
  },

  toggleSettings: () => {
    set((state) => ({ settingsOpen: !state.settingsOpen }));
  },

  updateConfig: (partial) => {
    set((state) => ({
      config: { ...state.config, ...partial },
    }));
  },
}));

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
        set({ running: true });
      } else if (message.status === "stopped") {
        set({ running: false });
      }
      break;

    case "error":
      console.error("Engine error:", message.message);
      break;
  }
}
