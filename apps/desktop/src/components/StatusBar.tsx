import { useEngineStore } from "../store/engineStore";

export function StatusBar() {
  const { connected, running, entries, config } = useEngineStore();

  return (
    <footer className="status-bar">
      <div className="status-bar-left">
        <span className={`status-dot ${connected ? "status-connected" : "status-disconnected"}`} />
        <span className="status-label">{connected ? "Connected" : "Disconnected"}</span>
        {running && (
          <>
            <span className="status-separator">|</span>
            <span className="status-recording">
              <PulsingDot />
              Listening
            </span>
          </>
        )}
      </div>

      <div className="status-bar-center">
        {running && (
          <span className="status-info">
            {config.modelSize} · {config.language}
            {config.translationEnabled && ` → ${config.targetLang}`}
          </span>
        )}
      </div>

      <div className="status-bar-right">
        <span className="status-count">
          {entries.length} segment{entries.length !== 1 ? "s" : ""}
        </span>
      </div>
    </footer>
  );
}

function PulsingDot() {
  return <span className="pulsing-dot" />;
}
