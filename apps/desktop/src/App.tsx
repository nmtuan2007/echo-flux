import { useEffect } from "react";
import { Header } from "./components/Header";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { useEngineStore } from "./store/engineStore";

export default function App() {
  const { settingsOpen, connect, disconnect } = useEngineStore();

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <div className="app">
      <Header />
      <main className="app-content">{settingsOpen ? <SettingsPanel /> : <TranscriptPanel />}</main>
      <StatusBar />
    </div>
  );
}
