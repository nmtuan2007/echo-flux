import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { ConfigSlice, createConfigSlice } from "./slices/configSlice";
import { TranscriptSlice, createTranscriptSlice } from "./slices/transcriptSlice";
import { UISlice, createUISlice } from "./slices/uiSlice";
import { WSSlice, createWSSlice } from "./slices/wsSlice";
import { TranscriptEntry, SuggestionOption, SuggestionResult, AudioDeviceInfo, ModelItem, HubSearchResult } from "./types";
import { Store } from "tauri-plugin-store-api";
import { StateStorage } from "zustand/middleware";

export * from "./types";

const isTauri = typeof window !== "undefined" && window.__TAURI__;
let tauriStore: Store | null = null;
if (isTauri) {
  tauriStore = new Store(".echoflux-settings.dat");
}

const customStorage: StateStorage = {
  getItem: async (name: string) => {
    if (tauriStore) {
      return (await tauriStore.get<string>(name)) || null;
    }
    return localStorage.getItem(name);
  },
  setItem: async (name: string, value: string) => {
    if (tauriStore) {
      await tauriStore.set(name, value);
      await tauriStore.save();
    } else {
      localStorage.setItem(name, value);
    }
  },
  removeItem: async (name: string) => {
    if (tauriStore) {
      await tauriStore.delete(name);
      await tauriStore.save();
    } else {
      localStorage.removeItem(name);
    }
  },
};

let idCounter = 0;
function nextId(): string {
  idCounter += 1;
  return `t-${Date.now()}-${idCounter}`;
}

export type StoreState = ConfigSlice & TranscriptSlice & UISlice & WSSlice;

export const useEngineStore = create<StoreState>()(
  persist(
    (...a) => ({
      ...createConfigSlice(...a),
      ...createTranscriptSlice(...a),
      ...createUISlice(...a),
      ...createWSSlice(handleMessage)(...a),
    }),
    {
      name: "echoflux-storage",
      storage: createJSONStorage(() => customStorage),
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
  set: (partial: Partial<StoreState> | ((state: StoreState) => Partial<StoreState>)) => void,
  get: () => StoreState,
) {
  switch (message.type) {
    case "partial": {
      const newPartial = (message.text as string) || "";
      const streamId = (message.source as string) || "default";
      const currentPartial = get().partials[streamId]?.text || "";

      const extra: Partial<StoreState> = {};
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

      const update: Partial<StoreState> = {};
      if (message.translation_backend) {
        update.activeTranslationBackend = message.translation_backend as string;
      }

      set((state) => {
        const newPartials = { ...state.partials };
        delete newPartials[streamId];

        if (!finalText.trim()) {
          return {
            ...update,
            partials: newPartials,
          };
        }

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
        const { connected, socket } = get();
        if (connected && socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "request_models_list" }));
        }
      } else {
        set((state) => ({
          downloading: true,
          downloadModel: state.downloadModel || (message.model as string) || "AI Model",
          downloadPercent: (message.percent as number) || 0,
        }));
      }
      break;
    }

    case "models_list": {
      set({ 
        modelsList: { 
          asr: (message.asr as ModelItem[]) ?? [], 
          translation: (message.translation as ModelItem[]) ?? [] 
        } 
      });
      break;
    }

    case "hub_search_results": {
      set({
        hubSearchResults: (message.results as HubSearchResult[]) ?? [],
        searchingHub: false
      });
      break;
    }

    case "model_action_result": {
      set({ downloading: false, downloadModel: "", downloadPercent: 0 });
      if (message.success) {
        get().requestModelsList();
      } else {
        console.error("Model action failed:", message.error);
        set({ appError: `Model Error: ${message.error}` });
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
      set({ isToggling: false, appError: (message.message as string) });
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
