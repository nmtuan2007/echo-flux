import { useEffect } from "react";
import { useEngineStore, ModelItem } from "../store/engineStore";

export function ModelManagerPanel() {
  const {
    connected,
    requestModelsList,
    modelsList,
    downloading,
    downloadModel,
    downloadPercent,
    downloadModelById,
    deleteModelById,
    config,
    updateConfig,
  } = useEngineStore();

  useEffect(() => {
    if (connected) {
      requestModelsList();
    }
  }, [connected, requestModelsList]);

  if (!connected) {
    return (
      <div className="settings-panel" style={{ padding: "24px" }}>
        <h2 style={{ marginBottom: 16 }}>Model Manager</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          Please start the pipeline (Connect) to view and manage models.
        </p>
      </div>
    );
  }

  const renderModelRow = (item: ModelItem, type: "asr" | "translation") => {
    // Determine if this exact model is currently downloading
    const isDownloadingThis = downloading && downloadModel === item.id;
      
    const isActive = type === "asr" 
      ? config.modelSize === item.id 
      : config.translationModel === item.id;

    return (
      <div
        key={item.id}
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "16px",
          background: "var(--bg-secondary)",
          borderRadius: "8px",
          marginBottom: "8px",
          border: isDownloadingThis ? "1px solid var(--accent)" : "1px solid transparent",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <div style={{ fontWeight: 600, fontSize: "15px" }}>{item.name}</div>
          <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Size: {item.size_mb} MB &nbsp;|&nbsp; Status:{" "}
            <span style={{ color: item.is_downloaded ? "var(--success, #4ade80)" : "inherit" }}>
              {item.is_downloaded ? "Downloaded" : "Not Downloaded"}
            </span>
          </div>
        </div>
        <div style={{ minWidth: "140px", display: "flex", justifyContent: "flex-end" }}>
          {isDownloadingThis ? (
            <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 6 }}>
              <div
                style={{
                  width: "100%",
                  height: "10px",
                  background: "var(--bg-tertiary)",
                  borderRadius: "5px",
                  overflow: "hidden",
                  border: "1px solid var(--border)",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${Math.max(0, Math.min(100, downloadPercent))}%`,
                    background: "linear-gradient(90deg, var(--accent), #60a5fa)",
                    boxShadow: "0 0 8px rgba(96, 165, 250, 0.5)",
                    transition: "width 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                  }}
                />
              </div>
              <div style={{ fontSize: "12px", textAlign: "right", color: "var(--text-secondary)" }}>
                {downloadPercent > 0 ? `${Math.round(downloadPercent)}%` : "Processing..."}
              </div>
            </div>
          ) : item.is_downloaded ? (
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <button
                className="btn"
                style={{ 
                  backgroundColor: isActive ? "var(--success)" : "var(--bg-tertiary)",
                  color: isActive ? "#ffffff" : "var(--text-primary)",
                }}
                onClick={() => {
                  if (type === "asr") updateConfig({ modelSize: item.id });
                  else updateConfig({ translationModel: item.id });
                }}
              >
                {isActive ? "✓ Selected" : "Select"}
              </button>
              
              <button
                className="btn btn-secondary"
                style={{ color: "var(--danger)" }}
                onClick={() => deleteModelById(item.id, type)}
              >
                🗑 Delete
              </button>
            </div>
          ) : (
            <button
              className="btn btn-secondary"
              style={{ color: "var(--accent)" }}
              onClick={() => downloadModelById(item.id, type)}
              disabled={downloading}
            >
              ⤓ Download
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="settings-panel" style={{ overflowY: "auto", height: "100%", padding: "24px" }}>
      <h2 style={{ marginBottom: "24px", fontSize: "1.75rem", fontWeight: 700 }}>Model Manager</h2>

      <div className="settings-section" style={{ marginBottom: "40px" }}>
        <h3 style={{ marginBottom: "16px", paddingBottom: "12px", borderBottom: "1px solid var(--border-color)", fontWeight: 600 }}>
          Speech Recognition (ASR)
        </h3>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {modelsList.asr.map((item) => renderModelRow(item, "asr"))}
          {modelsList.asr.length === 0 && (
            <div style={{ color: "var(--text-secondary)", fontStyle: "italic", padding: "12px 0" }}>Loading models...</div>
          )}
        </div>
      </div>

      <div className="settings-section">
        <h3 style={{ marginBottom: "16px", paddingBottom: "12px", borderBottom: "1px solid var(--border-color)", fontWeight: 600 }}>
          Translation (MarianMT)
        </h3>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {modelsList.translation.map((item) => renderModelRow(item, "translation"))}
          {modelsList.translation.length === 0 && (
            <div style={{ color: "var(--text-secondary)", fontStyle: "italic", padding: "12px 0" }}>Loading models...</div>
          )}
        </div>
      </div>
    </div>
  );
}
