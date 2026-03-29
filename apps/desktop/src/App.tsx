import { useEffect } from "react";
import { Header } from "./components/Header";
import { SettingsPanel } from "./components/SettingsPanel";
import { HistoryPanel } from "./components/HistoryPanel";
import { StatusBar } from "./components/StatusBar";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { SummaryModal } from "./components/SummaryModal";
import { useEngineStore } from "./store/engineStore";

export default function App() {
  const { activeView, connect, disconnect, theme } = useEngineStore();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <div className="app">
      <Header />
      <main className="app-content" style={{ position: "relative" }}>
        {activeView === "settings" && <SettingsPanel />}
        {activeView === "history" && <HistoryPanel />}
        {activeView === "transcript" && <TranscriptPanel />}
        {/* Summary modal overlays any active view */}
        <SummaryModal />
      </main>
      <StatusBar />
    </div>
  );
}
