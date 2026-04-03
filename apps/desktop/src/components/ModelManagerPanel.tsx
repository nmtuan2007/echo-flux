import { useEffect, useState } from "react";
import { useEngineStore, ModelItem, HubSearchResult } from "../store/engineStore";

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
    hubSearchResults,
    searchingHub,
    searchHub,
  } = useEngineStore();

  const [activeTab, setActiveTab] = useState<"installed" | "discover">("installed");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTask, setSearchTask] = useState<"asr" | "translation">("asr");

  useEffect(() => {
    if (connected) {
      requestModelsList();
    }
  }, [connected, requestModelsList]);

  if (!connected) {
    return (
      <div className="settings-panel" style={{ padding: "24px" }}>
        <h2 style={{ marginBottom: 16 }}>Universal Model Hub</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          Please start the pipeline (Connect) to view and manage models.
        </p>
      </div>
    );
  }

  const renderBadge = (text: string, color: string) => (
    <span style={{
      background: color,
      color: "#fff",
      padding: "2px 6px",
      borderRadius: "4px",
      fontSize: "11px",
      fontWeight: 'bold',
      marginLeft: "8px"
    }}>
      {text}
    </span>
  );

  const renderModelRow = (item: ModelItem, type: "asr" | "translation") => {
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
          border: isDownloadingThis ? "1px solid var(--accent)" : isActive ? "1px solid var(--success)" : "1px solid transparent",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <div style={{ fontWeight: 600, fontSize: "15px", display: "flex", alignItems: "center" }}>
            {item.name}
            {item.runtime && renderBadge(item.runtime.toUpperCase(), "var(--accent)")}
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Status:{" "}
            <span style={{ color: item.is_downloaded ? "var(--success, #4ade80)" : "inherit" }}>
              {item.is_downloaded ? "Installed" : "Not Downloaded"}
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
                🗑
              </button>
            </div>
          ) : null}
        </div>
      </div>
    );
  };

  const renderHubResultRow = (item: HubSearchResult) => {
    const isDownloadingThis = downloading && downloadModel === item.id;
    // Check if installed
    const isAlreadyInstalled = modelsList[item.task as "asr" | "translation"]?.some(m => m.id === item.id);

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
          border: "1px solid transparent",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <div style={{ fontWeight: 600, fontSize: "15px", display: "flex", alignItems: "center" }}>
            {item.id}
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Downloads: {item.downloads.toLocaleString()} &nbsp;|&nbsp; Task: {item.task.toUpperCase()}
          </div>
        </div>
        <div style={{ minWidth: "140px", display: "flex", justifyContent: "flex-end" }}>
          {isDownloadingThis ? (
            <div style={{ color: "var(--accent)", fontSize: "14px", fontWeight: "bold" }}>
              Downloading...
            </div>
          ) : isAlreadyInstalled ? (
             <div style={{ color: "var(--success)", fontSize: "14px", fontWeight: "bold" }}>
               Installed
             </div>
          ) : (
            <button
              className="btn btn-secondary"
              style={{ color: "var(--accent)" }}
              onClick={() => downloadModelById(item.id, item.task as "asr" | "translation")}
              disabled={downloading}
            >
              ⤓ Download
            </button>
          )}
        </div>
      </div>
    );
  };

  const handleSearch = () => {
    searchHub(searchQuery.trim(), searchTask);
  };

  useEffect(() => {
    if (activeTab === "discover") {
      searchHub(searchQuery.trim(), searchTask);
    }
  }, [activeTab, searchTask]);

  return (
    <div className="settings-panel" style={{ overflowY: "auto", height: "100%", padding: "24px", display: "flex", flexDirection: "column" }}>
      <h2 style={{ marginBottom: "20px", fontSize: "1.75rem", fontWeight: 700 }}>Universal Model Hub</h2>

      <div style={{ display: "flex", gap: "10px", marginBottom: "24px", borderBottom: "1px solid var(--border)", paddingBottom: "10px" }}>
        <button 
          className="btn" 
          style={{ 
            background: activeTab === "installed" ? "var(--bg-hover)" : "transparent",
            color: activeTab === "installed" ? "var(--text-primary)" : "var(--text-secondary)"
          }}
          onClick={() => setActiveTab("installed")}
        >
          Installed Models
        </button>
        <button 
          className="btn" 
          style={{ 
            background: activeTab === "discover" ? "var(--bg-hover)" : "transparent",
            color: activeTab === "discover" ? "var(--text-primary)" : "var(--text-secondary)"
          }}
          onClick={() => setActiveTab("discover")}
        >
          Discover (Hugging Face)
        </button>
      </div>

      {activeTab === "installed" && (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 32 }}>
          <div>
            <h3 style={{ marginBottom: "16px", paddingBottom: "8px", borderBottom: "1px dashed var(--border)", fontWeight: 600 }}>
              Speech Recognition (ASR)
            </h3>
            <div style={{ display: "flex", flexDirection: "column" }}>
              {modelsList.asr.map((item) => renderModelRow(item, "asr"))}
              {modelsList.asr.length === 0 && (
                <div style={{ color: "var(--text-secondary)", fontStyle: "italic", padding: "12px 0" }}>No ASR models installed.</div>
              )}
            </div>
          </div>

          <div>
            <h3 style={{ marginBottom: "16px", paddingBottom: "8px", borderBottom: "1px dashed var(--border)", fontWeight: 600 }}>
              Translation Models
            </h3>
            <div style={{ display: "flex", flexDirection: "column" }}>
              {modelsList.translation.map((item) => renderModelRow(item, "translation"))}
              {modelsList.translation.length === 0 && (
                <div style={{ color: "var(--text-secondary)", fontStyle: "italic", padding: "12px 0" }}>No Translation models installed.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === "discover" && (
        <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
          <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
            <select 
              className="select" 
              value={searchTask} 
              onChange={(e) => setSearchTask(e.target.value as "asr" | "translation")}
              style={{ width: "150px" }}
            >
              <option value="asr">ASR (Speech)</option>
              <option value="translation">Translation</option>
            </select>
            <input 
              type="text" 
              className="input" 
              placeholder="e.g., openai/whisper-tiny or facebook/seamless-m4t"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSearch(); }}
              style={{ flex: 1 }}
            />
            <button className="btn" style={{ background: "var(--accent)", color: "#fff" }} onClick={handleSearch}>
              Search
            </button>
          </div>

          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
            {searchingHub && <div style={{ color: "var(--text-secondary)", padding: 20 }}>Searching Hugging Face Hub...</div>}
            
            {!searchingHub && hubSearchResults.map(res => renderHubResultRow(res))}
            
            {!searchingHub && hubSearchResults.length === 0 && (
              <div style={{ color: "var(--text-secondary)", padding: 20 }}>
                No results found. Try a different term or check your connection.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
