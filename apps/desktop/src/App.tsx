import { useEffect } from "react";
import { Header } from "./components/Header";
import { HistoryPanel } from "./components/HistoryPanel";
import { ModelManagerPanel } from "./components/ModelManagerPanel";
import { OverlayPanel } from "./components/OverlayPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import { SummaryModal } from "./components/SummaryModal";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { useEngineStore } from "./store/engineStore";

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
    <div style={{
      position: 'absolute',
      bottom: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: 'var(--danger)',
      color: '#fff',
      padding: '12px 20px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(220, 38, 38, 0.4)',
      zIndex: 9999,
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      fontSize: '14px',
      fontWeight: 500,
      maxWidth: '80%',
      animation: 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)'
    }}>
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translate(-50%, 20px); }
          to { opacity: 1; transform: translate(-50%, 0); }
        }
      `}</style>
      <span>⚠ {appError}</span>
      <button 
        onClick={clearAppError} 
        style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px', padding: 0, opacity: 0.8 }}
      >
        ×
      </button>
    </div>
  );
}

export default function App() {
  const { activeView, connect, disconnect, theme } = useEngineStore();
  const isOverlay = appWindow.label === "overlay";

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    let unlistenSync: (() => void) | undefined;
    let unlistenReq: (() => void) | undefined;
    let unlistenRelay: (() => void) | undefined;
    let unsubStore: (() => void) | undefined;

    if (isOverlay) {
      // OVERLAY WINDOW: Mirror state from Main Window
      listen("sync_state", (event: any) => {
        useEngineStore.setState(event.payload);
      }).then(f => unlistenSync = f);

      emit("request_sync_state");
    } else {
      // MAIN WINDOW: Connect to server and broadcast state
      connect();

      listen("request_sync_state", () => {
        const state = useEngineStore.getState();
        emit("sync_state", {
          entries: state.entries,
          partials: state.partials,
          running: state.running,
          connected: state.connected,
          suggestionLoading: state.suggestionLoading,
          config: state.config,
        });
      }).then(f => unlistenReq = f);

      listen("relay_request_suggestion", (event: any) => {
        const { entryId, targetText, context } = event.payload;
        useEngineStore.getState().requestSuggestion(entryId, targetText, context);
      }).then(f => unlistenRelay = f);

      unsubStore = useEngineStore.subscribe((state) => {
        emit("sync_state", { // Broadcast state continuously
          entries: state.entries,
          partials: state.partials,
          running: state.running,
          connected: state.connected,
          suggestionLoading: state.suggestionLoading,
          config: state.config, // Ensure config is synced to pick up settings like stealthMode
        });
      });

      const setupShortcuts = async () => {
        try {
          await unregisterAll();
          await register("CommandOrControl+Shift+O", async () => {
            await invoke("toggle_overlay_window");
          });
          await register("CommandOrControl+Shift+M", () => {
            useEngineStore.getState().togglePipeline();
          });
          await register("CommandOrControl+Shift+S", () => {
            const state = useEngineStore.getState();
            // Find the last spoken entry from actual speakers, not from the user
            const lastSpoken = [...state.entries].reverse().find((e) => e.source !== "mic");
            if (lastSpoken && !state.suggestionLoading[lastSpoken.id]) {
              state.requestSuggestion(
                lastSpoken.id,
                lastSpoken.text,
                state.entries.map((e) => e.text),
              );
            }
          });
        } catch (e) {
          console.error("Failed to setup global shortcuts:", e);
        }
      };
      setupShortcuts();
    }

    return () => {
      if (unlistenSync) unlistenSync();
      if (unlistenReq) unlistenReq();
      if (unlistenRelay) unlistenRelay();
      if (unsubStore) unsubStore();
      if (!isOverlay) {
        unregisterAll().catch(console.error);
        disconnect();
      }
    };
  }, [connect, disconnect, isOverlay]);

  if (isOverlay) {
    return <OverlayPanel />;
  }

  return (
    <div className="app">
      <Header />
      <main className="app-content" style={{ position: "relative" }}>
        {activeView === "settings" && <SettingsPanel />}
        {activeView === "model_manager" && <ModelManagerPanel />}
        {activeView === "history" && <HistoryPanel />}
        {activeView === "transcript" && <TranscriptPanel />}
        {/* Summary modal overlays any active view */}
        <SummaryModal />
        <ErrorToast />
      </main>
      <StatusBar />
    </div>
  );
}
