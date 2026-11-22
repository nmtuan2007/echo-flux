import type { StateStorage } from "zustand/middleware";
import {
  BridgeFailure,
  CLOSE_AUTH_FAILED,
  describeCommandError,
  describeTransportError,
  errorCodeOf,
  eventAuthFrame,
  eventSocketUrl,
  forwardCommand,
  forwardStart,
  forwardStop,
  isForwardedCommandType,
  isTauriRuntime,
  parseEventPlaneFrame,
  startOwnedEngine,
  type ControlForwardResult,
  type EngineConnectionInfo,
  type ForwardedCommand,
  type ForwardedCommandType,
  type StartCredentialRef,
} from "../lib/engineBridge";
import { LEGACY_CREDENTIALS, LEGACY_SECRET_FIELDS } from "../lib/secureStore";
import type { EngineSliceCreator, EngineState } from "./engineStore";
import { findProvider, isSecretKeyName, publicConfigSnapshot } from "./settingsSlice";
import type {
  AudioDeviceInfo,
  EngineConfig,
  EngineMessage,
  ProviderCapability,
  TranscriptEntry,
} from "./types";

// Connection lifecycle, pipeline control, devices/models, and the transient error toast.
// Wire encoding and event routing are injected from engineStore.ts, which keeps the whole
// message contract in the one file tests/test_ws_contract.py scans.
//
// Two transports:
//   legacy - unauthenticated socket to a separately started engine (plain browser, tests).
//   secure - entered by bootstrapEngine() inside Tauri. Rust owns the engine process and the
//            control token; React opens a receive-only, role-authenticated event socket and
//            sends Start/Stop and every other command through Tauri invoke. No command is
//            ever written to that socket.

export interface EngineCommands {
  start: (config: EngineConfig) => string;
  stop: () => string;
  listDevices: () => string;
  requestSuggestion: (
    entryId: string,
    targetText: string,
    context: string[],
    config: EngineConfig,
  ) => string;
  requestSummary: (entries: TranscriptEntry[], config: EngineConfig) => string;
  testConnection?: () => string;
}

export interface SessionDeps {
  commands: EngineCommands;
  routeMessage: (message: EngineMessage) => void;
  reconnectDelayMs: number;
}

export type EngineTransport = "legacy" | "secure";

export const RECONNECT_DELAY_MS = 3000;

// Fresh objects on every call so no reset shares a reference with live state.
export function connectionResetState(): Partial<EngineState> {
  return {
    connected: false,
    socket: null,
    running: false,
    isToggling: false,
    preparedConversationId: null,
    activeTranslationBackend: null,
    downloading: false,
    downloadModel: "",
    downloadPercent: 0,
    audioLevels: {},
  };
}

export function sessionStartedState(existingConversationId?: string | null): Partial<EngineState> {
  return {
    running: true,
    isToggling: false,
    preparedConversationId: null,
    entries: [],
    partials: {},
    suggestionLoading: {},
    summaryText: "",
    summaryVisible: false,
    activeTranslationBackend: null,
    downloading: false,
    downloadModel: "",
    downloadPercent: 0,
    audioLevels: {},
    isViewingHistory: false,
    activeConversationId: existingConversationId ?? null,
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
    audioLevels: {},
  };
}



export function applyDevicesList(message: EngineMessage): Partial<EngineState> {
  return {
    availableMics: (message.microphones as AudioDeviceInfo[]) ?? [],
    availableSpeakers: (message.speakers as AudioDeviceInfo[]) ?? [],
  };
}

// The dotted `start` config minus every secret-named key. Reuses the single encoder in
// engineStore.ts so the key mapping lives in one place; secrets are blanked before encoding
// and any secret-named leaf is dropped again afterwards, because Rust rejects the request if
// one slips through.
export function publicStartConfig(
  commands: EngineCommands,
  config: EngineConfig,
): Record<string, unknown> {
  const encoded = JSON.parse(commands.start(publicConfigSnapshot(config))) as {
    config?: Record<string, unknown>;
  };
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(encoded.config ?? {})) {
    const leaf = key.slice(key.lastIndexOf(".") + 1);
    if (value !== undefined && value !== null && !isSecretKeyName(leaf)) out[key] = value;
  }
  return out;
}

const BLOCKED_NOTICE = "This action is not available on the secure engine connection";

// Legacy payloads carry raw keys (hf_token, llm_config.api_key). Rust refuses any command that
// still has one, so they are dropped at every depth before a command leaves the webview.
export function stripCommandSecrets(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripCommandSecrets);
  if (typeof value !== "object" || value === null) return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (key.toLowerCase() === "hf_token" || isSecretKeyName(key)) continue;
    out[key] = stripCommandSecrets(child);
  }
  return out;
}

// Reply type that completes each forwarded command; download_model completes on the event
// socket instead.
const TERMINAL_REPLY: Record<ForwardedCommandType, string | null> = {
  list_devices: "devices_list",
  request_models_list: "models_list",
  download_model: null,
  delete_model: "model_action_result",
  search_hub: "hub_search_results",
  request_suggestion: "suggestion_result",
  request_assist: null,
  configure_assistant: null,
  request_summary: "llm_done",
  summary_cancel: null,
  conversation_create: null,
  conversation_start: null,
  conversation_list: null,
  conversation_get: null,
  conversation_rename: null,
  conversation_delete: null,
  conversation_export: null,
  conversation_import: null,
  message_update: null,
  profile_create: null,
  profile_list: null,
  profile_get: null,
  profile_update: null,
  profile_delete: null,
  profile_attach: null,
  pack_status: null,
  pack_install: null,
  pack_cancel: null,
  pack_retry: null,
  pack_check_readiness: null,
  get_provider_manifests: null,
  audio_probe: null,
  test_connection: "test_connection_result",
  assistant_ask: null,
};

const COMMAND_LABEL: Record<ForwardedCommandType, string> = {
  list_devices: "Device list",
  request_models_list: "Model list",
  download_model: "Model download",
  delete_model: "Model delete",
  search_hub: "Hub search",
  request_suggestion: "Suggestion",
  request_assist: "Assist",
  configure_assistant: "Configure assistant",
  request_summary: "Summary",
  summary_cancel: "Cancel summary",
  conversation_create: "Create conversation",
  conversation_start: "Start conversation",
  conversation_list: "List conversations",
  conversation_get: "Get conversation",
  conversation_rename: "Rename conversation",
  conversation_delete: "Delete conversation",
  conversation_export: "Export conversation",
  conversation_import: "Import conversation",
  message_update: "Update segment",
  profile_create: "Create profile",
  profile_list: "List profiles",
  profile_get: "Get profile",
  profile_update: "Update profile",
  profile_delete: "Delete profile",
  profile_attach: "Attach profile",
  pack_status: "Pack status",
  pack_install: "Install pack",
  pack_cancel: "Cancel pack",
  pack_retry: "Retry pack",
  pack_check_readiness: "Check pack readiness",
  get_provider_manifests: "Get provider manifests",
  audio_probe: "Audio probe",
  test_connection: "Test connection",
  assistant_ask: "Ask Assistant",
};

// What components and the other slices see as `socket` in secure mode. Legacy call sites
// keep calling send(); nothing reaches the wire, because the event role may not send commands.
// Each payload is handed to `onCommand`, which forwards it through Tauri.
function receiveOnlySocket(ws: WebSocket, onCommand: (data: string) => void): WebSocket {
  const facade = {
    get readyState() {
      return ws.readyState;
    },
    send: (data: unknown) => {
      if (typeof data === "string") onCommand(data);
    },
    close: (code?: number, reason?: string) => ws.close(code, reason),
  };
  return facade as unknown as WebSocket;
}

export interface BootstrapOptions {
  // Storage holding the legacy blob to scrub after migration; defaults to localStorage.
  storage?: StateStorage;
}

export interface SessionSlice {
  // Connection
  connected: boolean;
  socket: WebSocket | null;
  transport: EngineTransport;
  // Bumped whenever the owned engine run changes or the connection is torn down; timers and
  // handlers from an older generation are ignored. Never carries a token.
  engineGeneration: number;

  // Pipeline
  running: boolean;
  isToggling: boolean;
  preparedConversationId: string | null;

  // Downloads
  downloading: boolean;
  downloadModel: string;
  downloadPercent: number;

  // Audio device lists, live levels & inline signal probe
  availableMics: AudioDeviceInfo[];
  availableSpeakers: AudioDeviceInfo[];
  audioLevels: Record<string, number>;
  audioProbeLoading: boolean;
  audioProbeResult: Record<string, { ok: boolean; level: number; error?: string }> | null;

  // Global Error Toast
  appError: string | null;

  // Secure desktop start-up: migrate legacy secrets, read credential status, start the owned
  // engine, and open the event socket. Outside Tauri it is the legacy connect().
  bootstrapEngine: (options?: BootstrapOptions) => Promise<void>;
  connect: () => void;
  disconnect: () => void;
  startPipeline: () => void;
  stopPipeline: () => void;
  togglePipeline: () => void;
  requestDeviceList: () => void;
  probeAudioInput: () => Promise<void>;
  clearAppError: () => void;
}

export const createSessionSlice =
  (deps: SessionDeps): EngineSliceCreator<SessionSlice> =>
  (set, get) => {
    // Per store instance, so separate stores (and tests) never share a pending reconnect.
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    // Secure transport only. The event token stays in this closure, never in store state.
    let generation = 0;
    let owned: EngineConnectionInfo | null = null;
    let connectInFlight: number | null = null;

    const cancelReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleSecureReconnect = (gen: number) => {
      cancelReconnect();
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (gen !== generation) return;
        get().connect();
      }, deps.reconnectDelayMs);
    };

    const connectedSocket = (): WebSocket | null => {
      const { socket, connected } = get();
      return socket && connected ? socket : null;
    };

    // Unwinds the loading state the calling action set before it sent the command.
    const clearCommandLoading = (type: ForwardedCommandType, entryId: unknown) => {
      switch (type) {
        case "request_suggestion":
          set((state) => {
            if (typeof entryId !== "string" || !(entryId in state.suggestionLoading)) return {};
            const suggestionLoading = { ...state.suggestionLoading };
            delete suggestionLoading[entryId];
            return { suggestionLoading };
          });
          return;
        case "request_summary":
          set({ summaryLoading: false });
          return;
        default:
          return;
      }
    };

    const forwardSocketCommand = async (data: string) => {
      let parsed: Record<string, unknown>;
      try {
        const value = JSON.parse(data);
        if (typeof value !== "object" || value === null || Array.isArray(value)) return;
        parsed = value as Record<string, unknown>;
      } catch {
        return;
      }
      const type = parsed.type;
      if (!isForwardedCommandType(type)) {
        set({ appError: BLOCKED_NOTICE });
        return;
      }
      const entryId = parsed.entry_id;
      const gen = generation;

      // A secret typed in Settings but not yet committed goes to the vault first, so the
      // reference below can cover it.
      if (LEGACY_SECRET_FIELDS.some((field) => get().config[field])) {
        await get().commitLegacySecrets();
      }
      if (gen !== generation) return;

      const command = { ...(stripCommandSecrets(parsed) as ForwardedCommand), type };
      const credentials: Record<string, StartCredentialRef> = {};
      if ((type === "download_model" || type === "search_hub") && get().secretStatus.hfToken) {
        credentials.hf_token = { ...LEGACY_CREDENTIALS.hfToken };
      }

      let result: ControlForwardResult;
      try {
        result = await forwardCommand(command, credentials);
      } catch (error) {
        if (gen !== generation) return;
        clearCommandLoading(type, entryId);
        if (type === "request_summary" && get().summaryCancelledByUser) return;
        const code = errorCodeOf(error);
        set({ appError: `${COMMAND_LABEL[type]}: ${describeCommandError(code)} (${code})` });
        return;
      }
      // Replies for an engine run that has since been replaced describe nothing current.
      if (gen !== generation) return;
      const skipStreamedChunks = type === "request_summary" && get().summaryStreamedViaEvent;
      for (const message of result.messages) {
        if (skipStreamedChunks && message.type === "llm_chunk" && message.is_error !== true) {
          continue;
        }
        deps.routeMessage(message);
      }

      const terminal = TERMINAL_REPLY[type];
      const completed =
        terminal === null
          ? !result.messages.some((m) => m.type === "error")
          : result.messages.some((m) => m.type === terminal);
      if (!completed) clearCommandLoading(type, entryId);
    };

    const connectLegacy = () => {
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
    };

    const connectSecure = async () => {
      const { socket } = get();
      if (
        socket &&
        (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
      if (connectInFlight === generation) return;

      const requested = generation;
      connectInFlight = requested;
      let info: EngineConnectionInfo;
      try {
        info = await startOwnedEngine();
      } catch (error) {
        if (connectInFlight === requested) connectInFlight = null;
        if (requested !== generation) return;
        set({
          appError:
            error instanceof BridgeFailure
              ? error.message
              : "The engine could not be started (bridge_failed)",
        });
        scheduleSecureReconnect(requested);
        return;
      }
      if (connectInFlight === requested) connectInFlight = null;
      if (requested !== generation || get().transport !== "secure") return;

      if (!owned || owned.port !== info.port || owned.eventToken !== info.eventToken) {
        // A different engine run: everything scheduled against the previous one is void.
        generation += 1;
        cancelReconnect();
        owned = info;
        const previous = get().socket;
        set({ ...connectionResetState(), engineGeneration: generation });
        previous?.close();
      }

      const gen = generation;
      const ws = new WebSocket(eventSocketUrl(info));
      const facade = receiveOnlySocket(ws, (data) => void forwardSocketCommand(data));
      let authenticated = false;
      const isCurrent = () => gen === generation && get().socket === facade;

      ws.onopen = () => {
        if (!isCurrent()) {
          ws.close();
          return;
        }
        // The only frame React ever writes to the engine socket.
        ws.send(eventAuthFrame(info.eventToken));
      };

      ws.onmessage = (event) => {
        if (!isCurrent()) return;
        const frame = parseEventPlaneFrame(event.data);
        if (!frame) return;
        if (!authenticated) {
          // Nothing but the handshake reply is processed before auth_ok.
          if (frame.op === "auth_ok") {
            authenticated = true;
            cancelReconnect();
            set({ connected: true });
            if (get().history.length > 0) {
              void get().migrateLegacyHistory();
            }
          } else if (frame.op === "error") {
            set({ appError: describeTransportError(frame.code) });
          }
          return;
        }
        if (frame.op === "event") {
          if (frame.event.type === "llm_chunk" && frame.event.is_error !== true) {
            set({ summaryStreamedViaEvent: true });
          }
          deps.routeMessage(frame.event);
        } else if (frame.op === "error") {
          set({ appError: describeTransportError(frame.code) });
        }
      };

      ws.onclose = (event?: CloseEvent) => {
        if (!isCurrent()) return;
        set(connectionResetState());
        // A rejected token means the run changed under us; take whatever Rust reports next.
        if (event?.code === CLOSE_AUTH_FAILED) owned = null;
        scheduleSecureReconnect(gen);
      };

      ws.onerror = () => {
        ws.close();
      };

      set({ socket: facade, engineGeneration: gen });
    };

    // Routes the legacy replies a forwarded command returns: in authenticated mode the engine
    // answers start/stop to the control connection only, never on the event plane.
    const runControl = async (gen: number, action: () => Promise<ControlForwardResult>) => {
      try {
        const result = await action();
        if (gen === generation) {
          for (const message of result.messages) deps.routeMessage(message);
        }
      } catch (error) {
        if (gen === generation) {
          set({
            appError:
              error instanceof BridgeFailure
                ? error.message
                : "The engine control request failed (bridge_failed)",
          });
        }
      } finally {
        if (get().isToggling) set({ isToggling: false });
      }
    };

    const startSecure = async () => {
      const { connected, isToggling, services: currentServices, consent } = get();
      if (!connected || isToggling) return;

      for (const cap of ["stt", "translation"] as const) {
        const svc = currentServices?.[cap];
        const active = cap === "stt" || Boolean(svc?.enabled);
        const manifest = svc ? findProvider(cap, svc.providerId) : undefined;
        if (active && manifest && manifest.protocolFamily !== "local" && !consent?.[cap]) {
          set({
            appError: `Cloud consent is required for ${cap === "stt" ? "Speech-to-Text" : "Translation"} (${manifest.displayName}). Grant permission in Settings before starting.`,
          });
          return;
        }
      }

      const gen = generation;
      const {
        activeConversationId: prevActiveId,
        engineConversationId: prevEngineId,
        preparedConversationId,
        isViewingHistory,
      } = get();
      const hasPrepared = Boolean(
        preparedConversationId &&
          (prevActiveId === preparedConversationId || prevEngineId === preparedConversationId),
      );
      set({
        isToggling: true,
        activeView: "transcript",
        preparedConversationId: null,
        ...(!hasPrepared
          ? {
              activeConversationId: null,
              engineConversationId: null,
              isViewingHistory: false,
              ...(isViewingHistory ? { pendingTitle: "" } : {}),
            }
          : {}),
      });

      if (LEGACY_SECRET_FIELDS.some((field) => get().config[field])) {
        await get().commitLegacySecrets();
      }
      const {
        config,
        secretStatus,
        services,
        activeConversationId,
        engineConversationId,
        pendingTitle,
        sourceLabels,
        attachedProfileIds,
      } = get();
      const credentials: Record<string, StartCredentialRef> = {};
      if (secretStatus.llmApiKey) {
        credentials["llm.api_key"] = { ...LEGACY_CREDENTIALS.llmApiKey };
      }
      if (services) {
        for (const [serviceName, serviceConfig] of Object.entries(services)) {
          if (!serviceConfig?.credentialRefs) continue;
          for (const [slot, ref] of Object.entries(serviceConfig.credentialRefs)) {
            if (ref && ref.configId && ref.configId !== "legacy-llm") {
              credentials[`${serviceName}.${slot}`] = {
                capability: serviceName as ProviderCapability,
                configId: ref.configId,
                slot: ref.slot || slot,
              };
            }
          }
        }
      }
      const publicConfig = publicStartConfig(deps.commands, config);
      if (services) {
        publicConfig.services = services;
      }
      const convId = activeConversationId || engineConversationId;
      if (convId) {
        publicConfig.conversation_id = convId;
      }
      if (pendingTitle.trim()) {
        publicConfig.conversation_title = pendingTitle.trim();
      }
      if (sourceLabels) {
        publicConfig.source_labels = {
          mic: sourceLabels.mic || "Me",
          system: sourceLabels.system || "Others",
        };
      }
      if (attachedProfileIds.length > 0) {
        publicConfig.attached_profile_ids = attachedProfileIds;
      }
      await runControl(gen, () => forwardStart(publicConfig, credentials));
    };

    return {
      connected: false,
      socket: null,
      transport: "legacy",
      engineGeneration: 0,
      running: false,
      isToggling: false,
      preparedConversationId: null,
      downloading: false,
      downloadModel: "",
      downloadPercent: 0,
      availableMics: [],
      availableSpeakers: [],
      audioLevels: {},
      audioProbeLoading: false,
      audioProbeResult: null,
      appError: null,

      bootstrapEngine: async (options = {}) => {
        if (!isTauriRuntime()) {
          get().connect();
          return;
        }
        set({ transport: "secure" });
        const gen = generation;
        await get().commitLegacySecrets(options.storage);
        await get().refreshCredentialStatus();
        // Torn down (or re-bootstrapped) while the vault calls ran.
        if (gen !== generation) return;
        get().connect();
      },

      connect: () => {
        if (get().transport === "secure") {
          void connectSecure();
          return;
        }
        connectLegacy();
      },

      disconnect: () => {
        cancelReconnect();
        const { socket, transport } = get();
        if (transport === "secure") {
          generation += 1;
          connectInFlight = null;
          // Reset before closing so the socket's own close handler sees it as stale.
          set({ ...connectionResetState(), engineGeneration: generation });
        } else {
          set(connectionResetState());
        }
        if (socket) {
          socket.close();
        }
      },

      startPipeline: () => {
        if (get().transport === "secure") {
          void startSecure();
          return;
        }
        const {
          config,
          isViewingHistory,
          activeConversationId,
          engineConversationId,
          preparedConversationId,
        } = get();
        const socket = connectedSocket();
        if (!socket) return;

        const hasPrepared = Boolean(
          preparedConversationId &&
            (activeConversationId === preparedConversationId ||
              engineConversationId === preparedConversationId),
        );
        set({
          isToggling: true,
          activeView: "transcript",
          preparedConversationId: null,
          ...(!hasPrepared
            ? {
                activeConversationId: null,
                engineConversationId: null,
                isViewingHistory: false,
                ...(isViewingHistory ? { pendingTitle: "" } : {}),
              }
            : {}),
        });
        socket.send(deps.commands.start(config));
      },

      stopPipeline: () => {
        if (get().transport === "secure") {
          const { connected, isToggling } = get();
          if (!connected || isToggling) return;
          set({ isToggling: true });
          void runControl(generation, forwardStop);
          return;
        }
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

      probeAudioInput: async () => {
        const { connected, running, audioProbeLoading, config } = get();
        if (!connected || running || audioProbeLoading) return;
        set({ audioProbeLoading: true });
        try {
          const res = await forwardCommand({
            type: "audio_probe",
            audio_source: config.audioSource,
            mic_device_id: config.micDeviceId || null,
            speaker_device_id: config.speakerDeviceId || null,
            duration_ms: 350,
          });
          const dataMsg = res.messages.find((m) => (m as any).streams !== undefined) as any;
          const streams = dataMsg?.streams || null;
          set({ audioProbeResult: streams, audioProbeLoading: false });
          if (!streams) {
            set({ appError: "Audio check did not return stream metrics." });
          }
        } catch (error) {
          const code = errorCodeOf(error);
          set({
            audioProbeLoading: false,
            appError: `Audio check failed: ${describeCommandError(code)} (${code})`,
          });
        }
      },

      clearAppError: () => {
        set({ appError: null });
      },
    };
  };
