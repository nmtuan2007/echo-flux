import { create, type StateCreator } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";
import {
  applyLlmChunk,
  createConversationSlice,
  sanitizeHistory,
  type ConversationSlice,
} from "./conversationSlice";
import {
  applyDevicesList,
  createSessionSlice,
  RECONNECT_DELAY_MS,
  sessionStartedState,
  sessionStoppedState,
  type EngineCommands,
  type SessionSlice,
} from "./sessionSlice";
import {
  createSettingsSlice,
  sanitizeConfig,
  sanitizeServices,
  sanitizeTheme,
  type PersistedConfig,
  type SettingsSlice,
} from "./settingsSlice";
import {
  applyFinal,
  applyPartial,
  applySuggestionResult,
  applyTranslationUpdate,
  createTranscriptSlice,
  type TranscriptSlice,
} from "./transcriptSlice";
import type { Conversation, EngineConfig, EngineMessage, ServicesConfig, Theme } from "./types";

// Composition root for the UI store. Components keep importing `useEngineStore` and the
// types below from this module.
//
// The full engine<->UI wire surface (every command encoded and every event routed) stays in
// this file: tests/test_ws_contract.py reads only engineStore.ts.

export type {
  AppView,
  AudioDeviceInfo,
  Conversation,
  EngineConfig,
  SuggestionOption,
  SuggestionResult,
  TranscriptEntry,
} from "./types";

export type EngineState = SettingsSlice & SessionSlice & TranscriptSlice & ConversationSlice;

export type EngineSliceCreator<T> = StateCreator<
  EngineState,
  [["zustand/persist", unknown]],
  [],
  T
>;

type StoreSet = Parameters<EngineSliceCreator<EngineState>>[0];
type StoreGet = () => EngineState;

// ---------------------------------------------------------------------------
// UI -> engine commands
// ---------------------------------------------------------------------------

export const engineCommands: EngineCommands = {
  start: (config) =>
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

  stop: () => JSON.stringify({ type: "stop" }),

  listDevices: () => JSON.stringify({ type: "list_devices" }),

  requestSuggestion: (entryId, targetText, context, config) =>
    JSON.stringify({
      type: "request_suggestion",
      entry_id: entryId,
      target_text: targetText,
      context,
      llm_config: {
        api_key: config.llmApiKey,
        model: config.llmModel,
        provider_url: config.llmProviderUrl || undefined,
      },
    }),

  requestSummary: (entries, config) =>
    JSON.stringify({
      type: "request_summary",
      entries: entries.map((e) => ({ text: e.text, source: e.source ?? "speaker" })),
      llm_config: {
        api_key: config.llmApiKey,
        model: config.llmModel,
        provider_url: config.llmProviderUrl || undefined,
      },
    }),

  testConnection: () =>
    JSON.stringify({
      type: "test_connection",
    }),
};

// ---------------------------------------------------------------------------
// Engine -> UI events
// ---------------------------------------------------------------------------

export function routeEngineMessage(message: EngineMessage, set: StoreSet, get: StoreGet): void {
  switch (message.type) {
    case "partial": {
      const update = applyPartial(get(), message);
      if (update) set(update);
      break;
    }

    case "final":
      set((state) => applyFinal(state, message));
      break;

    case "translation_update":
      set((state) => applyTranslationUpdate(state, message));
      break;

    case "status":
      if (message.status === "started") {
        set((state) =>
          sessionStartedState(
            state.activeConversationId || state.engineConversationId,
            state.preserveEntriesOnStart,
          ),
        );
      } else if (message.status === "stopped") {
        get().saveCurrentSession();
        set(sessionStoppedState());
      }
      break;

    case "error":
      console.error("Engine error:", message.message);
      set({ isToggling: false, appError: message.message as string });
      break;

    case "audio_level": {
      const src = message.source;
      const lvl = message.level;
      if ((src === "mic" || src === "system") && typeof lvl === "number" && Number.isFinite(lvl)) {
        set((state) => ({
          audioLevels: { ...state.audioLevels, [src]: lvl },
        }));
      }
      break;
    }

    case "devices_list":
      set(applyDevicesList(message));
      break;

    case "suggestion_result": {
      if (!message.entry_id) break;
      set((state) => applySuggestionResult(state, message));
      get().saveCurrentSession();
      break;
    }

    case "llm_chunk":
      set((state) => applyLlmChunk(state, message));
      break;

    case "llm_done": {
      const warningRaw = message.warning as Record<string, unknown> | null | undefined;
      const warning =
        warningRaw && typeof warningRaw === "object"
          ? {
              reason: String(warningRaw.reason || "context_limit"),
              omitted_message_count: Number(warningRaw.omitted_message_count || 0),
              included_message_count:
                typeof message.included_message_count === "number"
                  ? message.included_message_count
                  : undefined,
              first_omitted_timestamp_ms:
                typeof warningRaw.first_omitted_timestamp_ms === "number"
                  ? warningRaw.first_omitted_timestamp_ms
                  : undefined,
              last_omitted_timestamp_ms:
                typeof warningRaw.last_omitted_timestamp_ms === "number"
                  ? warningRaw.last_omitted_timestamp_ms
                  : undefined,
            }
          : null;
      const summaryError =
        typeof message.error === "string" && message.error ? message.error : undefined;
      set((state) => ({
        summaryLoading: false,
        summaryWarning: warning ?? state.summaryWarning,
        ...(summaryError ? { summaryError } : {}),
      }));
      get().saveCurrentSession();
      break;
    }

    case "test_connection_result":
      break;
  }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

// Versioned key. The pre-slice blob under the legacy key is read, never rewritten: it is the
// only copy of the user's legacy LLM key until the vault migration task consumes it.
export const PERSIST_STORAGE_NAME = "echoflux-storage-v1";
export const LEGACY_PERSIST_STORAGE_NAME = "echoflux-storage";
export const PERSIST_VERSION = 1;

// The complete persisted whitelist. Sockets, loading maps, tokens, raw keys, and transient
// errors are absent by construction.
export interface PersistedEngineState {
  theme: Theme;
  config: PersistedConfig;
  services: ServicesConfig;
  history: Conversation[];
}

export function selectPersistedState(state: EngineState): PersistedEngineState {
  return {
    theme: state.theme,
    config: sanitizeConfig(state.config) as PersistedConfig,
    services: sanitizeServices(state.services),
    history: sanitizeHistory(state.history) ?? [],
  };
}

export function sanitizePersistedState(raw: unknown): Partial<PersistedEngineState> {
  if (typeof raw !== "object" || raw === null) return {};
  const source = raw as Record<string, unknown>;
  const out: Partial<PersistedEngineState> = {};
  const theme = sanitizeTheme(source.theme);
  if (theme) out.theme = theme;
  if (source.config !== undefined) out.config = sanitizeConfig(source.config) as PersistedConfig;
  if (source.services !== undefined) out.services = sanitizeServices(source.services);
  const history = sanitizeHistory(source.history);
  if (history) out.history = history;
  return out;
}

// Hook for future schema changes. Version 0 is the pre-slice blob; secrets are dropped from
// what gets rewritten but are not migrated anywhere yet.
export function migratePersistedState(
  persisted: unknown,
  _fromVersion: number,
): Partial<PersistedEngineState> {
  return sanitizePersistedState(persisted);
}

type LegacySecrets = Partial<Pick<EngineConfig, "llmApiKey" | "hfToken">>;

// Read-only compatibility: the legacy runtime still needs these in memory.
function readLegacySecrets(storage: StateStorage): LegacySecrets {
  const raw = storage.getItem(LEGACY_PERSIST_STORAGE_NAME);
  if (typeof raw !== "string") return {};
  try {
    const config = JSON.parse(raw)?.state?.config;
    const secrets: LegacySecrets = {};
    if (typeof config?.llmApiKey === "string" && config.llmApiKey) {
      secrets.llmApiKey = config.llmApiKey;
    }
    if (typeof config?.hfToken === "string" && config.hfToken) {
      secrets.hfToken = config.hfToken;
    }
    return secrets;
  } catch {
    return {};
  }
}

export interface CreateEngineStoreOptions {
  // Defaults to window.localStorage.
  storage?: StateStorage;
  reconnectDelayMs?: number;
}

export function createEngineStore(options: CreateEngineStoreOptions = {}) {
  const resolveStorage = (): StateStorage => options.storage ?? window.localStorage;

  // Falls back to the legacy blob only when no versioned state exists yet, which routes the
  // first load through migrate().
  const versionedStorage = (): StateStorage => {
    const base = resolveStorage();
    return {
      getItem: (name) => {
        const value = base.getItem(name);
        return value === null && name === PERSIST_STORAGE_NAME
          ? base.getItem(LEGACY_PERSIST_STORAGE_NAME)
          : value;
      },
      setItem: (name, value) => base.setItem(name, value),
      removeItem: (name) => base.removeItem(name),
    };
  };

  return create<EngineState>()(
    persist(
      (set, get, api) => {
        const commands = engineCommands;
        return {
          ...createSettingsSlice(set, get, api),
          ...createSessionSlice({
            commands,
            routeMessage: (message) => routeEngineMessage(message, set, get),
            reconnectDelayMs: options.reconnectDelayMs ?? RECONNECT_DELAY_MS,
          })(set, get, api),
          ...createTranscriptSlice(commands)(set, get, api),
          ...createConversationSlice(commands)(set, get, api),
        };
      },
      {
        name: PERSIST_STORAGE_NAME,
        version: PERSIST_VERSION,
        storage: createJSONStorage<PersistedEngineState>(versionedStorage),
        partialize: selectPersistedState,
        migrate: (persisted, version) =>
          migratePersistedState(persisted, version) as PersistedEngineState,
        merge: (persisted, current) => {
          const safe = sanitizePersistedState(persisted);
          let secrets: LegacySecrets = {};
          try {
            secrets = readLegacySecrets(resolveStorage());
          } catch {
            secrets = {};
          }
          return {
            ...current,
            theme: safe.theme ?? current.theme,
            history: safe.history ?? current.history,
            services: safe.services ? { ...current.services, ...safe.services } : current.services,
            config: { ...current.config, ...safe.config, ...secrets },
          };
        },
      },
    ),
  );
}

export const useEngineStore = createEngineStore();
