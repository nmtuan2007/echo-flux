import { useEffect, useState } from "react";
import { appWindow } from "@tauri-apps/api/window";
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
import { useGlobalShortcuts } from "./hooks/useGlobalShortcuts";
import { useOverlaySync } from "./hooks/useOverlaySync";
import { useEngineStore } from "./store/engineStore";

function ErrorToast() {
  const { appError, clearAppError } = useEngineStore();
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (!appError || paused) return;
    const t = setTimeout(() => clearAppError(), 9000);
    return () => clearTimeout(t);
  }, [appError, paused, clearAppError]);

  if (!appError) return null;

  return (
    <div
      role="alert"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
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
        type="button"
        aria-label="Dismiss notification"
        onClick={clearAppError}
        style={{
          background: "transparent",
          border: "none",
          color: "#fff",
          cursor: "pointer",
          fontSize: "18px",
          padding: 0,
          opacity: 0.85,
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

  useOverlaySync(isOverlay);
  useGlobalShortcuts(isOverlay, shortcutsEnabled);

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
