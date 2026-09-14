import type { EngineSliceCreator, EngineState } from "./engineStore";
import type {
  AudioDeviceInfo,
  EngineConfig,
  EngineMessage,
  HubSearchResult,
  ModelItem,
  ModelType,
  TranscriptEntry,
} from "./types";

// Connection lifecycle, pipeline control, devices/models, and the transient error toast.
// Wire encoding and event routing are injected from engineStore.ts, which keeps the whole
// message contract in the one file tests/test_ws_contract.py scans.

export interface EngineCommands {
  start: (config: EngineConfig) => string;
  stop: () => string;
  listDevices: () => string;
  requestModelsList: () => string;
  downloadModel: (modelId: string, modelType: ModelType, hfToken: string) => string;
  deleteModel: (modelId: string, modelType: ModelType) => string;
  searchHub: (query: string, task: ModelType, hfToken: string) => string;
  requestSuggestion: (
    entryId: string,
    targetText: string,
    context: string[],
    config: EngineConfig,
  ) => string;
  requestSummary: (entries: TranscriptEntry[], config: EngineConfig) => string;
}

export interface SessionDeps {
  commands: EngineCommands;
  routeMessage: (message: EngineMessage) => void;
  reconnectDelayMs: number;
}

export const RECONNECT_DELAY_MS = 3000;

// Fresh objects on every call so no reset shares a reference with live state.
export function connectionResetState(): Partial<EngineState> {
  return {
    connected: false,
    socket: null,
    running: false,
    isToggling: false,
    activeTranslationBackend: null,
    downloading: false,
    downloadModel: "",
    downloadPercent: 0,
    modelsList: { asr: [], translation: [] },
  };
}

export function sessionStartedState(): Partial<EngineState> {
  return {
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
  };
}

export function sessionStoppedState(): Partial<EngineState> {
  return {
    running: false,
    isToggling: false,
    partials: {},
    downloading: false,
    downloadModel: "",
    downloadPercent: 0,
  };
}

export function downloadIdleState(): Partial<EngineState> {
  return { downloading: false, downloadModel: "", downloadPercent: 0 };
}

export function applyDownloadProgress(
  state: EngineState,
  message: EngineMessage,
): Partial<EngineState> {
  return {
    downloading: true,
    downloadModel: state.downloadModel || (message.model as string) || "AI Model",
    downloadPercent: (message.percent as number) || 0,
  };
}

export function applyModelsList(message: EngineMessage): Partial<EngineState> {
  return {
    modelsList: {
      asr: (message.asr as ModelItem[]) ?? [],
      translation: (message.translation as ModelItem[]) ?? [],
    },
  };
}

export function applyHubSearchResults(message: EngineMessage): Partial<EngineState> {
  return {
    hubSearchResults: (message.results as HubSearchResult[]) ?? [],
    searchingHub: false,
  };
}

export function applyDevicesList(message: EngineMessage): Partial<EngineState> {
  return {
    availableMics: (message.microphones as AudioDeviceInfo[]) ?? [],
    availableSpeakers: (message.speakers as AudioDeviceInfo[]) ?? [],
  };
}

export interface SessionSlice {
  // Connection
  connected: boolean;
  socket: WebSocket | null;

  // Pipeline
  running: boolean;
  isToggling: boolean;

  // Downloads & Hub
  downloading: boolean;
  downloadModel: string;
  downloadPercent: number;
  hubSearchResults: HubSearchResult[];
  searchingHub: boolean;
  modelsList: { asr: ModelItem[]; translation: ModelItem[] };

  // Audio device lists
  availableMics: AudioDeviceInfo[];
  availableSpeakers: AudioDeviceInfo[];

  // Global Error Toast
  appError: string | null;

  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  togglePipeline: () => void;
  requestDeviceList: () => void;
  clearAppError: () => void;
  requestModelsList: () => void;
  downloadModelById: (id: string, type: ModelType) => void;
  deleteModelById: (id: string, type: ModelType) => void;
  searchHub: (query: string, task: ModelType) => void;
}

export const createSessionSlice =
  (deps: SessionDeps): EngineSliceCreator<SessionSlice> =>
  (set, get) => {
    // Per store instance, so separate stores (and tests) never share a pending reconnect.
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const cancelReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const connectedSocket = (): WebSocket | null => {
      const { socket, connected } = get();
      return socket && connected ? socket : null;
    };

    return {
      connected: false,
      socket: null,
      running: false,
      isToggling: false,
      downloading: false,
      downloadModel: "",
      downloadPercent: 0,
      hubSearchResults: [],
      searchingHub: false,
      modelsList: { asr: [], translation: [] },
      availableMics: [],
      availableSpeakers: [],
      appError: null,

      connect: () => {
        const { config, socket } = get();
        if (socket && socket.readyState === WebSocket.OPEN) return;

        const url = `ws://${config.host}:${config.port}`;
        const ws = new WebSocket(url);
        // Handlers of a socket that has since been replaced or disconnected are ignored;
        // otherwise a late close would wipe the live connection or re-arm reconnect.
        const isCurrent = () => get().socket === ws;

        ws.onopen = () => {
          if (!isCurrent()) {
            ws.close();
            return;
          }
          set({ connected: true, socket: ws });
          cancelReconnect();
        };

        ws.onmessage = (event) => {
          if (!isCurrent()) return;
          let message: unknown;
          try {
            message = JSON.parse(event.data);
          } catch {
            return; // Ignore malformed messages
          }
          if (message && typeof message === "object") {
            deps.routeMessage(message as EngineMessage);
          }
        };

        ws.onclose = () => {
          if (!isCurrent()) return;
          set(connectionResetState());
          cancelReconnect();
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            get().connect();
          }, deps.reconnectDelayMs);
        };

        ws.onerror = () => {
          ws.close();
        };

        set({ socket: ws });
      },

      disconnect: () => {
        cancelReconnect();
        const { socket } = get();
        // Reset before closing so the socket's own close handler sees it as stale.
        set(connectionResetState());
        if (socket) {
          socket.close();
        }
      },

      startPipeline: () => {
        const { config } = get();
        const socket = connectedSocket();
        if (!socket) return;

        set({ isToggling: true, activeView: "transcript" });
        socket.send(deps.commands.start(config));
      },

      stopPipeline: () => {
        const socket = connectedSocket();
        if (!socket) return;

        set({ isToggling: true });
        socket.send(deps.commands.stop());
      },

      togglePipeline: () => {
        const { running } = get();
        if (running) {
          get().stopPipeline();
        } else {
          get().startPipeline();
        }
      },

      requestDeviceList: () => {
        connectedSocket()?.send(deps.commands.listDevices());
      },

      clearAppError: () => {
        set({ appError: null });
      },

      requestModelsList: () => {
        connectedSocket()?.send(deps.commands.requestModelsList());
      },

      downloadModelById: (id, type) => {
        const { downloading, config } = get();
        const socket = connectedSocket();
        if (!socket || downloading) return;
        set({ downloading: true, downloadModel: id, downloadPercent: 0 });
        socket.send(deps.commands.downloadModel(id, type, config.hfToken));
      },

      deleteModelById: (id, type) => {
        connectedSocket()?.send(deps.commands.deleteModel(id, type));
      },

      searchHub: (query, task) => {
        const { config } = get();
        const socket = connectedSocket();
        if (!socket) return;
        set({ searchingHub: true, hubSearchResults: [] });
        socket.send(deps.commands.searchHub(query, task, config.hfToken));
      },
    };
  };
