import { useEffect } from "react";
import { emit, listen } from "@tauri-apps/api/event";
import { useEngineStore } from "../store/engineStore";
import { publicConfigSnapshot } from "../store/settingsSlice";

export function useOverlaySync(isOverlay: boolean): void {
  const bootstrapEngine = useEngineStore((s) => s.bootstrapEngine);
  const disconnect = useEngineStore((s) => s.disconnect);

  useEffect(() => {
    let unlistenSync: (() => void) | undefined;
    let unlistenAudioSync: (() => void) | undefined;
    let unlistenReq: (() => void) | undefined;
    let unlistenClosed: (() => void) | undefined;
    let unlistenRelay: (() => void) | undefined;
    let unlistenAssistRelay: (() => void) | undefined;
    let unlistenToggleRelay: (() => void) | undefined;
    let unlistenTransRelay: (() => void) | undefined;
    let unlistenTargetLangRelay: (() => void) | undefined;
    let unlistenOverlayPrefs: (() => void) | undefined;
    let unlistenOverlayLock: (() => void) | undefined;
    let unsubStore: (() => void) | undefined;
    let overlayActive = false;

    let lastSyncedServices: unknown = null;
    let lastSyncedConfig: unknown = null;
    let lastSyncedLabels: unknown = null;
    let lastSyncedEntries: unknown = null;
    const getSyncPayload = (full: boolean) => {
      const state = useEngineStore.getState();
      const payload: Record<string, unknown> = {
        partials: state.partials,
        running: state.running,
        connected: state.connected,
        isToggling: state.isToggling,
        appError: state.appError,
        suggestionLoading: state.suggestionLoading,
        engineConversationId: state.engineConversationId,
        sttStreams: state.sttStreams,
        shortcutsEnabled: state.shortcutsEnabled,
        overlayFontSize: state.overlayFontSize,
        overlayHighOpacity: state.overlayHighOpacity,
      };
      if (full) {
        payload.audioLevels = state.audioLevels;
      }
      if (full || state.entries !== lastSyncedEntries) {
        lastSyncedEntries = state.entries;
        // Overlay only renders recent captions; slicing keeps Tauri IPC bounded during long sessions.
        payload.entries = state.entries.slice(-30);
      }
      if (full || state.sourceLabels !== lastSyncedLabels) {
        lastSyncedLabels = state.sourceLabels;
        payload.sourceLabels = state.sourceLabels;
      }
      if (full || state.services !== lastSyncedServices) {
        lastSyncedServices = state.services;
        payload.services = state.services;
      }
      if (full || state.config !== lastSyncedConfig) {
        lastSyncedConfig = state.config;
        payload.config = publicConfigSnapshot(state.config);
      }
      return payload;
    };

    let syncTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleSync = () => {
      if (!overlayActive || syncTimer) return;
      syncTimer = setTimeout(() => {
        syncTimer = null;
        if (!overlayActive) return;
        void emit("sync_state", getSyncPayload(false));
      }, 60);
    };

    let audioSyncTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleAudioSync = () => {
      if (!overlayActive || audioSyncTimer) return;
      audioSyncTimer = setTimeout(() => {
        audioSyncTimer = null;
        if (!overlayActive) return;
        void emit("sync_audio_levels", useEngineStore.getState().audioLevels);
      }, 100);
    };

    if (isOverlay) {
      // OVERLAY WINDOW: Mirror state from Main Window
      listen("sync_state", (event: any) => {
        useEngineStore.setState(event.payload);
      }).then((f) => (unlistenSync = f));

      listen("sync_audio_levels", (event: any) => {
        useEngineStore.setState({ audioLevels: event.payload });
      }).then((f) => (unlistenAudioSync = f));

      void emit("request_sync_state");
    } else {
      // MAIN WINDOW: start the owned engine (Tauri) or connect (browser), then broadcast state
      void bootstrapEngine();

      listen("request_sync_state", () => {
        overlayActive = true;
        void emit("sync_state", getSyncPayload(true));
      }).then((f) => (unlistenReq = f));

      listen("overlay_closed", () => {
        overlayActive = false;
        useEngineStore.getState().setOverlayLocked(false);
      }).then((f) => (unlistenClosed = f));

      listen("relay_set_overlay_prefs", (event: any) => {
        const payload = event.payload || {};
        useEngineStore.getState().setOverlayPrefs({
          ...(payload.fontSize ? { fontSize: payload.fontSize } : {}),
          ...(typeof payload.highOpacity === "boolean"
            ? { highOpacity: payload.highOpacity }
            : {}),
        });
      }).then((f) => (unlistenOverlayPrefs = f));

      listen("overlay_lock_changed", (event: any) => {
        useEngineStore.getState().setOverlayLocked(Boolean(event.payload?.locked));
      }).then((f) => (unlistenOverlayLock = f));

      listen("relay_request_suggestion", (event: any) => {
        const { entryId } = event.payload || {};
        if (entryId) {
          useEngineStore.getState().requestAssist(entryId);
        }
      }).then((f) => (unlistenRelay = f));

      listen("relay_request_assist", (event: any) => {
        const { entryId } = event.payload || {};
        if (entryId) {
          useEngineStore.getState().requestAssist(entryId);
        }
      }).then((f) => (unlistenAssistRelay = f));

      listen("relay_toggle_pipeline", () => {
        useEngineStore.getState().togglePipeline();
      }).then((f) => (unlistenToggleRelay = f));

      listen("relay_toggle_translation", () => {
        const s = useEngineStore.getState();
        const next = !(s.services?.translation?.enabled ?? s.config.translationEnabled);
        void s.configureLiveTranslation({ enabled: next });
      }).then((f) => (unlistenTransRelay = f));

      listen("relay_set_target_lang", (event: any) => {
        const targetLang = event.payload?.targetLang;
        if (typeof targetLang === "string" && targetLang) {
          void useEngineStore.getState().configureLiveTranslation({ targetLang });
        }
      }).then((f) => (unlistenTargetLangRelay = f));

      unsubStore = useEngineStore.subscribe((state, prev) => {
        if (!overlayActive) return;
        if (state.audioLevels !== prev.audioLevels) {
          scheduleAudioSync();
        }
        if (
          state.entries !== prev.entries ||
          state.partials !== prev.partials ||
          state.running !== prev.running ||
          state.connected !== prev.connected ||
          state.isToggling !== prev.isToggling ||
          state.appError !== prev.appError ||
          state.suggestionLoading !== prev.suggestionLoading ||
          state.engineConversationId !== prev.engineConversationId ||
          state.sourceLabels !== prev.sourceLabels ||
          state.sttStreams !== prev.sttStreams ||
          state.services !== prev.services ||
          state.config !== prev.config ||
          state.shortcutsEnabled !== prev.shortcutsEnabled ||
          state.overlayFontSize !== prev.overlayFontSize ||
          state.overlayHighOpacity !== prev.overlayHighOpacity
        ) {
          scheduleSync();
        }
      });
    }

    return () => {
      if (syncTimer) clearTimeout(syncTimer);
      if (audioSyncTimer) clearTimeout(audioSyncTimer);
      if (unlistenSync) unlistenSync();
      if (unlistenAudioSync) unlistenAudioSync();
      if (unlistenReq) unlistenReq();
      if (unlistenClosed) unlistenClosed();
      if (unlistenRelay) unlistenRelay();
      if (unlistenAssistRelay) unlistenAssistRelay();
      if (unlistenToggleRelay) unlistenToggleRelay();
      if (unlistenTransRelay) unlistenTransRelay();
      if (unlistenTargetLangRelay) unlistenTargetLangRelay();
      if (unlistenOverlayPrefs) unlistenOverlayPrefs();
      if (unlistenOverlayLock) unlistenOverlayLock();
      if (unsubStore) unsubStore();
      if (!isOverlay) {
        disconnect();
      } else {
        void emit("overlay_closed");
      }
    };
  }, [bootstrapEngine, disconnect, isOverlay]);
}
