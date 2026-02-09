import { useState } from "react";
import { useEngineStore, Conversation } from "../store/engineStore";

export function HistoryPanel() {
  const { history, deleteConversation, loadConversation, renameConversation, running } =
    useEngineStore();

  return (
    <div className="history-panel">
      <h2 className="settings-title">Conversation History</h2>
      {running && <p className="settings-warning">Stop recording to load past conversations.</p>}

      {history.length === 0 ? (
        <div className="history-empty">No saved conversations.</div>
      ) : (
        <div className="history-list">
          {history.map((conv) => (
            <HistoryItem
              key={conv.id}
              conversation={conv}
              onLoad={() => !running && loadConversation(conv.id)}
              onDelete={() => deleteConversation(conv.id)}
              onRename={(name) => renameConversation(conv.id, name)}
              disabled={running}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryItem({
  conversation,
  onLoad,
  onDelete,
  onRename,
  disabled,
}: {
  conversation: Conversation;
  onLoad: () => void;
  onDelete: () => void;
  onRename: (name: string) => void;
  disabled: boolean;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempName, setTempName] = useState(conversation.name);

  const handleSave = () => {
    if (tempName.trim()) {
      onRename(tempName.trim());
    } else {
      setTempName(conversation.name); // Revert if empty
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setTempName(conversation.name);
      setIsEditing(false);
    }
  };

  const dateStr = new Date(conversation.date).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });

  return (
    <div className={`history-item ${disabled ? "disabled" : ""}`}>
      <div className="history-item-content">
        {isEditing ? (
          <input
            className="history-input"
            value={tempName}
            onChange={(e) => setTempName(e.target.value)}
            onBlur={handleSave}
            onKeyDown={handleKeyDown}
            autoFocus
          />
        ) : (
          <div className="history-name" onClick={() => !disabled && onLoad()} title="Click to load">
            {conversation.name}
          </div>
        )}
        <div className="history-meta">
          <span className="history-date">{dateStr}</span>
          <span className="history-count">{conversation.entries.length} entries</span>
        </div>
      </div>

      <div className="history-actions">
        {isEditing ? (
          <button className="btn-icon-sm success" onClick={handleSave} title="Save">
            ✓
          </button>
        ) : (
          <button className="btn-icon-sm" onClick={() => setIsEditing(true)} title="Rename">
            ✎
          </button>
        )}
        <button
          className="btn-icon-sm danger"
          onClick={(e) => {
            e.stopPropagation();
            if (confirm("Delete this conversation?")) onDelete();
          }}
          title="Delete"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

// Add these styles to global.css later, or keep them inline if necessary.
// For now, assume standard classes.
