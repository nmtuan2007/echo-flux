import { StateCreator } from "zustand";
import { AudioDeviceInfo, HubSearchResult, ModelItem } from "../types";
import { StoreState } from "../engineStore";

export interface WSSlice {
  connected: boolean;
  socket: WebSocket | null;
  running: boolean;
  isToggling: boolean;

  activeTranslationBackend: string | null;

  downloading: boolean;
  downloadModel: string;
  downloadPercent: number;

  hubSearchResults: HubSearchResult[];
  searchingHub: boolean;

  modelsList: { asr: ModelItem[]; translation: ModelItem[] };

  availableMics: AudioDeviceInfo[];
  availableSpeakers: AudioDeviceInfo[];

  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  togglePipeline: () => void;

  requestDeviceList: () => void;
  requestModelsList: () => void;
  downloadModelById: (id: string, type: "asr" | "translation") => void;
  deleteModelById: (id: string, type: "asr" | "translation") => void;
  searchHub: (query: string, task: "asr" | "translation") => void;

  requestSuggestion: (entryId: string, targetText: string, context: string[]) => void;
  requestSummary: () => void;
}

export const createWSSlice = (handleMessage: any): StateCreator<StoreState, [], [], WSSlice> => {
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let isConnecting = false;
  let connectId = 0;

  return (set, get) => ({
    connected: false,
    socket: null,
    running: false,
    isToggling: false,
    activeTranslationBackend: null,
    downloading: false,
    downloadModel: "",
    downloadPercent: 0,
    hubSearchResults: [],
    searchingHub: false,
    modelsList: { asr: [], translation: [] },
    availableMics: [],
    availableSpeakers: [],

    connect: async () => {
      const { config, socket } = get();
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
      if (isConnecting) return;

      isConnecting = true;
      const myId = ++connectId;
      let token = "";
      try {
        if (typeof window !== "undefined" && window.__TAURI__) {
          const { readTextFile } = await import("@tauri-apps/api/fs");
          const { homeDir, join } = await import("@tauri-apps/api/path");
          const home = await homeDir();
          const tokenPath = await join(home, ".echoflux", "ws_token.txt");
          token = await readTextFile(tokenPath);
          token = token.trim();
        }
      } catch (err) {
        console.warn("Could not read auth token:", err);
      }

      if (myId !== connectId) {
        // A disconnect or another connect happened while we were awaiting. Abort.
        return;
      }

      const url = `ws://${config.host}:${config.port}`;
      const ws = new WebSocket(url);

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "auth", token }));
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
          // Ignore
        }
      };

      ws.onclose = () => {
        isConnecting = false;
        set({
          connected: false,
          socket: null,
          running: false,
          isToggling: false,
          activeTranslationBackend: null,
          downloading: false,
          downloadModel: "",
          downloadPercent: 0,
          modelsList: { asr: [], translation: [] },
        });
        reconnectTimer = setTimeout(() => get().connect(), 3000);
      };

      ws.onerror = () => {
        isConnecting = false;
        ws.close();
      };
      set({ socket: ws });
      isConnecting = false;
    },

    disconnect: () => {
      connectId++; // Abort any pending connect awaits
      isConnecting = false;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      const { socket } = get();
      if (socket) socket.close();
      set({
        connected: false,
        socket: null,
        running: false,
        isToggling: false,
        activeTranslationBackend: null,
        downloading: false,
        downloadModel: "",
        downloadPercent: 0,
        modelsList: { asr: [], translation: [] },
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
        })
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
      if (running) get().stopPipeline();
      else get().startPipeline();
    },

    requestDeviceList: () => {
      const { socket, connected } = get();
      if (!socket || !connected) return;
      socket.send(JSON.stringify({ type: "list_devices" }));
    },

    requestModelsList: () => {
      const { socket, connected } = get();
      if (!socket || !connected) return;
      socket.send(JSON.stringify({ type: "request_models_list" }));
    },

    downloadModelById: (id, type) => {
      const { socket, connected, downloading, config } = get();
      if (!socket || !connected || downloading) return;
      set({ downloading: true, downloadModel: id, downloadPercent: 0 });
      socket.send(JSON.stringify({ type: "download_model", model_id: id, model_type: type, hf_token: config.hfToken }));
    },

    deleteModelById: (id, type) => {
      const { socket, connected } = get();
      if (!socket || !connected) return;
      socket.send(JSON.stringify({ type: "delete_model", model_id: id, model_type: type }));
    },

    searchHub: (query, task) => {
      const { socket, connected, config } = get();
      if (!socket || !connected) return;
      set({ searchingHub: true, hubSearchResults: [] });
      socket.send(JSON.stringify({ type: "search_hub", query, task, hf_token: config.hfToken }));
    },

    requestSuggestion: async (entryId, targetText, context) => {
      const { socket, connected, config } = get();
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
  });
};
