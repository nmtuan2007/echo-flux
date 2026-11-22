import { useEffect } from "react";
import { Header } from "./components/Header";
import { HistoryPanel } from "./components/HistoryPanel";
import { OverlayPanel } from "./components/OverlayPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import { SummaryModal } from "./components/SummaryModal";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { LiveSessionBar } from "./components/LiveSessionBar";
import { ContextProfilesPanel } from "./components/ContextProfilesPanel";
import { ConversationSetupPanel } from "./components/ConversationSetupPanel";
import { OnboardingDialog } from "./components/OnboardingDialog";
import { useEngineStore } from "./store/engineStore";
import { publicConfigSnapshot } from "./store/settingsSlice";

import { register, unregisterAll } from "@tauri-apps/api/globalShortcut";
import { invoke } from "@tauri-apps/api/tauri";
import { appWindow } from "@tauri-apps/api/window";
import { emit, listen } from "@tauri-apps/api/event";

function ErrorToast() {
  const { appError, clearAppError } = useEngineStore();

  useEffect(() => {
    if (appError) {
      const t = setTimeout(() => clearAppError(), 5000);
      return () => clearTimeout(t);
    }
  }, [appError, clearAppError]);

  if (!appError) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: "20px",
        left: "50%",
        transform: "translateX(-50%)",
        backgroundColor: "var(--danger)",
        color: "#fff",
        padding: "12px 20px",
        borderRadius: "8px",
        boxShadow: "0 4px 12px rgba(220, 38, 38, 0.4)",
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        gap: "12px",
        fontSize: "14px",
        fontWeight: 500,
        maxWidth: "80%",
        animation: "slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translate(-50%, 20px); }
          to { opacity: 1; transform: translate(-50%, 0); }
        }
      `}</style>
      <span>⚠ {appError}</span>
      <button
        onClick={clearAppError}
        style={{
          background: "transparent",
          border: "none",
          color: "#fff",
          cursor: "pointer",
          fontSize: "18px",
          padding: 0,
          opacity: 0.8,
        }}
      >
        ×
      </button>
    </div>
  );
}

export default function App() {
  const {
    activeView,
    bootstrapEngine,
    disconnect,
    theme,
    shortcutsEnabled,
    onboardingCompleted,
    isSetupOpen,
    closeSetup,
    startSessionWithSetup,
  } = useEngineStore();
  const isOverlay = appWindow.label === "overlay";

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    let unlistenSync: (() => void) | undefined;
    let unlistenReq: (() => void) | undefined;
    let unlistenClosed: (() => void) | undefined;
    let unlistenRelay: (() => void) | undefined;
    let unlistenAssistRelay: (() => void) | undefined;
    let unlistenToggleRelay: (() => void) | undefined;
    let unsubStore: (() => void) | undefined;
    let overlayActive = false;

    let lastSyncedServices: unknown = null;
    let lastSyncedConfig: unknown = null;
    let lastSyncedLabels: unknown = null;
    let lastSyncedEntries: unknown = null;
    let lastSyncedAssistRequests: unknown = null;

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
        audioLevels: state.audioLevels,
      };
      if (full || state.entries !== lastSyncedEntries) {
        lastSyncedEntries = state.entries;
        // Overlay only renders recent captions; slicing keeps Tauri IPC bounded during long sessions.
        payload.entries = state.entries.slice(-20);
      }
      if (full || state.assistRequests !== lastSyncedAssistRequests) {
        lastSyncedAssistRequests = state.assistRequests;
        payload.assistRequests = state.assistRequests;
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

    if (isOverlay) {
      // OVERLAY WINDOW: Mirror state from Main Window
      listen("sync_state", (event: any) => {
        useEngineStore.setState(event.payload);
      }).then((f) => (unlistenSync = f));

      emit("request_sync_state");
    } else {
      // MAIN WINDOW: start the owned engine (Tauri) or connect (browser), then broadcast state
      void bootstrapEngine();

      listen("request_sync_state", () => {
        overlayActive = true;
        emit("sync_state", getSyncPayload(true));
      }).then((f) => (unlistenReq = f));

      listen("overlay_closed", () => {
        overlayActive = false;
      }).then((f) => (unlistenClosed = f));

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

      unsubStore = useEngineStore.subscribe((state, prev) => {
        if (!overlayActive) return;
        if (
          state.entries !== prev.entries ||
          state.partials !== prev.partials ||
          state.running !== prev.running ||
          state.connected !== prev.connected ||
          state.isToggling !== prev.isToggling ||
          state.appError !== prev.appError ||
          state.suggestionLoading !== prev.suggestionLoading ||
          state.assistRequests !== prev.assistRequests ||
          state.engineConversationId !== prev.engineConversationId ||
          state.sourceLabels !== prev.sourceLabels ||
          state.sttStreams !== prev.sttStreams ||
          state.services !== prev.services ||
          state.config !== prev.config ||
          state.audioLevels !== prev.audioLevels
        ) {
          scheduleSync();
        }
      });
    }

    return () => {
      if (syncTimer) clearTimeout(syncTimer);
      if (unlistenSync) unlistenSync();
      if (unlistenReq) unlistenReq();
      if (unlistenClosed) unlistenClosed();
      if (unlistenRelay) unlistenRelay();
      if (unlistenAssistRelay) unlistenAssistRelay();
      if (unlistenToggleRelay) unlistenToggleRelay();
      if (unsubStore) unsubStore();
      if (!isOverlay) {
        disconnect();
      } else {
        void emit("overlay_closed");
      }
    };
  }, [bootstrapEngine, disconnect, isOverlay]);

  useEffect(() => {
    if (isOverlay) return;
    let cancelled = false;
    const syncShortcuts = async () => {
      try {
        await unregisterAll();
        if (!shortcutsEnabled || cancelled) return;
        await register("CommandOrControl+Shift+O", async () => {
          await invoke("toggle_overlay_window");
        });
        await register("CommandOrControl+Shift+M", () => {
          useEngineStore.getState().togglePipeline();
        });
        await register("CommandOrControl+Shift+S", () => {
          const state = useEngineStore.getState();
          const finalEntries = state.entries.filter((e) => e.isFinal && e.text.trim());
          const target =
            [...finalEntries].reverse().find((e) => e.source !== "mic") ||
            [...finalEntries].reverse()[0];
          if (target) {
            state.requestAssist(target.id);
          }
        });
        await register("CommandOrControl+Shift+U", () => {
          void emit("overlay_unlock");
        });
      } catch (e) {
        console.error("Failed to setup global shortcuts:", e);
      }
    };
    void syncShortcuts();
    return () => {
      cancelled = true;
      unregisterAll().catch(() => {});
    };
  }, [isOverlay, shortcutsEnabled]);

  if (isOverlay) {
    return <OverlayPanel />;
  }

  return (
    <div className="app">
      <Header />
      <main className="app-content" style={{ position: "relative" }}>
        {activeView === "settings" && <SettingsPanel />}
        {activeView === "history" && <HistoryPanel />}
        {activeView === "profiles" && <ContextProfilesPanel />}
        {activeView === "transcript" && (
          <div className="transcript-workspace">
            <LiveSessionBar />
            <TranscriptPanel />
          </div>
        )}
        {/* Modals and Overlays */}
        <SummaryModal />
        <ConversationSetupPanel
          isOpen={isSetupOpen}
          onClose={closeSetup}
          onStart={startSessionWithSetup}
        />
        {!onboardingCompleted && <OnboardingDialog />}
        <ErrorToast />
      </main>
      <StatusBar />
    </div>
  );
}
