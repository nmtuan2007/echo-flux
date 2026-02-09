import { useEngineStore } from "../store/engineStore";

export function Header() {
  const { 
    running, 
    connected, 
    isToggling, 
    activeView,
    startPipeline, 
    stopPipeline, 
    clearTranscript, 
    setActiveView 
  } = useEngineStore();

  const handleStartStop = () => {
    if (isToggling) return;
    if (running) {
      stopPipeline();
    } else {
      startPipeline();
    }
  };

  const toggleView = (view: "history" | "settings") => {
    if (activeView === view) {
      setActiveView("transcript");
    } else {
      setActiveView(view);
    }
  };

  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title" onClick={() => setActiveView("transcript")} style={{cursor: 'pointer'}}>
          EchoFlux
        </h1>
      </div>

      <div className="header-center">
        <button 
          className={`btn ${running ? "btn-stop" : "btn-start"}`} 
          onClick={handleStartStop} 
          disabled={!connected || isToggling}
        >
          {isToggling ? (
            <>
              <Spinner />
              {running ? "Stopping..." : "Starting..."}
            </>
          ) : (
            running ? "Stop" : "Start"
          )}
        </button>
        
        <button 
          className="btn btn-secondary" 
          onClick={clearTranscript} 
          disabled={isToggling || activeView !== "transcript"}
          title="Clear current transcript"
        >
          Clear
        </button>
      </div>

      <div className="header-right">
        <button 
          className={`btn btn-icon ${activeView === "history" ? "active" : ""}`}
          onClick={() => toggleView("history")} 
          title="History"
        >
          <HistoryIcon />
        </button>
        <button 
          className={`btn btn-icon ${activeView === "settings" ? "active" : ""}`}
          onClick={() => toggleView("settings")} 
          title="Settings"
        >
          <SettingsIcon />
        </button>
      </div>
    </header>
  );
}

function SettingsIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <polyline points="12 6 12 12 16 14"></polyline>
    </svg>
  );
}

function Spinner() {
  return (
    <svg 
      className="spinner" 
      width="14" 
      height="14" 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="3" 
      strokeLinecap="round" 
      strokeLinejoin="round"
      style={{ marginRight: "6px" }}
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}
