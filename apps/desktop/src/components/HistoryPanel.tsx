import { useEffect, useRef, useState } from "react";
import { useEngineStore, Conversation } from "../store/engineStore";

interface DisplayItem {
  id: string;
  title: string;
  date: number;
  messageCount: number;
  hasSummary: boolean;
  isLegacy: boolean;
  legacyData?: Conversation;
}

export function HistoryPanel() {
  const {
    history,
    savedConversations,
    historyLoading,
    historySearchQuery,
    selectedConversationDetail,
    running,
    loadHistoryList,
    getConversationDetail,
    deleteConversation,
    renameConversation,
    exportConversation,
    importConversation,
    setHistorySearchQuery,
    clearSelectedConversationDetail,
    loadConversation,
  } = useEngineStore();

  const [deleteTarget, setDeleteTarget] = useState<DisplayItem | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadHistoryList().catch(() => {});
  }, [loadHistoryList]);

  // Combine savedConversations and legacy history, de-duplicating by ID
  const displayItems: DisplayItem[] = [];
  const seenIds = new Set<string>();

  for (const s of savedConversations) {
    if (!s.conversation_id || seenIds.has(s.conversation_id)) continue;
    seenIds.add(s.conversation_id);
    displayItems.push({
      id: s.conversation_id,
      title: s.title || "Untitled Conversation",
      date: s.created_at_ms || s.updated_at_ms || Date.now(),
      messageCount: s.message_count ?? 0,
      hasSummary: Boolean(s.has_summary),
      isLegacy: false,
    });
  }

  for (const h of history) {
    if (!h.id || seenIds.has(h.id)) continue;
    seenIds.add(h.id);
    displayItems.push({
      id: h.id,
      title: h.name || "Untitled Conversation",
      date: typeof h.date === "number" ? h.date : new Date(h.date).getTime() || Date.now(),
      messageCount: h.entries?.length ?? 0,
      hasSummary: Boolean(h.summary),
      isLegacy: true,
      legacyData: h,
    });
  }

  // Filter items by search query
  const query = historySearchQuery.trim().toLowerCase();
  const filteredItems = displayItems.filter((item) =>
    query ? item.title.toLowerCase().includes(query) : true,
  );

  // Sorting: most recent first
  filteredItems.sort((a, b) => b.date - a.date);

  const handleStartRename = (item: DisplayItem) => {
    setEditingId(item.id);
    setEditingTitle(item.title);
  };

  const handleSaveRename = async (id: string) => {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      await renameConversation(id, trimmed);
    }
    setEditingId(null);
  };

  const handleExport = async (item: DisplayItem) => {
    let jsonString: string | null = null;
    if (item.isLegacy && item.legacyData) {
      jsonString = JSON.stringify(item.legacyData, null, 2);
    } else {
      jsonString = await exportConversation(item.id);
    }
    if (!jsonString) return;

    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `echoflux-conversation-${item.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      await importConversation(text);
      await loadHistoryList();
    } catch (err) {
      console.error("Failed to import conversation file:", err);
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSelectConversation = async (item: DisplayItem) => {
    if (item.isLegacy && item.legacyData) {
      // Create mock detail from legacy
      const detail = {
        conversation: {
          conversation_id: item.legacyData.id,
          title: item.legacyData.name,
          created_at_ms: typeof item.legacyData.date === "number" ? item.legacyData.date : Date.now(),
        },
        messages: item.legacyData.entries.map((e, idx) => ({
          message_id: e.id || `msg-${idx}`,
          source: e.source,
          speaker_label: ((e as any).speaker as string) || (e.source === "mic" ? "Me" : "Others"),
          stt_text: e.text,
          translation_text: e.translation,
          timestamp_ms: e.timestamp,
        })),
        summary: item.legacyData.summary
          ? { summary_markdown: item.legacyData.summary, created_at_ms: Date.now() }
          : null,
      };
      useEngineStore.setState({ selectedConversationDetail: detail });
    } else {
      await getConversationDetail(item.id);
    }
  };

  return (
    <div className="history-panel">
      {/* Detail View */}
      {selectedConversationDetail ? (
        <ConversationDetailView
          detail={selectedConversationDetail}
          onBack={clearSelectedConversationDetail}
          onLoadToActive={() => {
            const convId =
              selectedConversationDetail.conversation?.conversation_id ||
              selectedConversationDetail.conversation_id;
            if (convId && !running) {
              loadConversation(convId);
              clearSelectedConversationDetail();
            }
          }}
          disabled={running}
        />
      ) : (
        <>
          <div className="history-header-row">
            <h2 className="settings-title">Conversation History</h2>
            <div className="history-toolbar">
              <input
                type="text"
                className="history-search-input"
                placeholder="Search conversations..."
                value={historySearchQuery}
                onChange={(e) => setHistorySearchQuery(e.target.value)}
                aria-label="Search conversations"
              />
              <button
                className="btn btn-secondary history-import-btn"
                onClick={() => fileInputRef.current?.click()}
                title="Import conversation JSON"
              >
                📥 Import
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={handleImportFile}
              />
            </div>
          </div>

          {running && (
            <p className="settings-warning">Stop recording to load past conversations into the live view.</p>
          )}

          {historyLoading ? (
            <div className="history-loading">Loading conversations...</div>
          ) : filteredItems.length === 0 ? (
            <div className="history-empty">
              {query ? `No conversations match "${historySearchQuery}".` : "No saved conversations."}
            </div>
          ) : (
            <div className="history-list">
              {filteredItems.map((item) => (
                <div key={item.id} className="history-item">
                  <div className="history-item-content">
                    {editingId === item.id ? (
                      <div className="history-edit-row">
                        <input
                          className="history-input"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleSaveRename(item.id);
                            if (e.key === "Escape") setEditingId(null);
                          }}
                          autoFocus
                        />
                        <button
                          className="btn-icon-sm success"
                          onClick={() => handleSaveRename(item.id)}
                          title="Save"
                        >
                          ✓
                        </button>
                        <button
                          className="btn-icon-sm"
                          onClick={() => setEditingId(null)}
                          title="Cancel"
                        >
                          ✕
                        </button>
                      </div>
                    ) : (
                      <div
                        className="history-name"
                        onClick={() => handleSelectConversation(item)}
                        title="Click to view details"
                      >
                        {item.title}
                      </div>
                    )}
                    <div className="history-meta">
                      <span className="history-date">
                        {new Date(item.date).toLocaleString(undefined, {
                          dateStyle: "medium",
                          timeStyle: "short",
                        })}
                      </span>
                      <span className="history-count">
                        {item.messageCount} {item.messageCount === 1 ? "entry" : "entries"}
                      </span>
                      {item.hasSummary && <span className="history-summary-tag">📝 Summary</span>}
                    </div>
                  </div>

                  <div className="history-actions">
                    <button
                      className="btn-icon-sm"
                      onClick={() => handleSelectConversation(item)}
                      title="View Details"
                      aria-label="View Details"
                    >
                      👁
                    </button>
                    <button
                      className="btn-icon-sm history-export-btn"
                      onClick={() => handleExport(item)}
                      title="Export JSON"
                      aria-label="Export JSON"
                    >
                      📤
                    </button>
                    {editingId !== item.id && (
                      <button
                        className="btn-icon-sm"
                        onClick={() => handleStartRename(item)}
                        title="Rename"
                        aria-label="Rename"
                      >
                        ✎
                      </button>
                    )}
                    <button
                      className="btn-icon-sm danger"
                      onClick={() => setDeleteTarget(item)}
                      title="Delete conversation"
                      aria-label="Delete conversation"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div
          className="history-modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={() => setDeleteTarget(null)}
        >
          <div className="history-modal-content" onClick={(e) => e.stopPropagation()}>
            <h3 className="history-modal-title">Delete Conversation</h3>
            <p className="history-modal-body">
              Are you sure you want to permanently delete <strong>{deleteTarget.title}</strong>? All
              transcript messages, translations, and summaries will be removed.
            </p>
            <div className="history-modal-actions">
              <button
                className="btn btn-secondary history-delete-cancel-btn"
                onClick={() => setDeleteTarget(null)}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger history-delete-confirm-btn"
                onClick={async () => {
                  const targetId = deleteTarget.id;
                  setDeleteTarget(null);
                  await deleteConversation(targetId);
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Archived Conversation Detail Viewer
 */
function ConversationDetailView({
  detail,
  onBack,
  onLoadToActive,
  disabled,
}: {
  detail: any;
  onBack: () => void;
  onLoadToActive: () => void;
  disabled: boolean;
}) {
  const conv = detail.conversation || detail;
  const messages = detail.messages || [];
  const summary = detail.summary;
  const profiles = detail.profiles || [];

  return (
    <div className="history-detail-view">
      <div className="history-detail-header">
        <button className="btn btn-secondary history-detail-back-btn" onClick={onBack}>
          ← Back to History
        </button>
        <div className="history-detail-title-group">
          <h2 className="history-detail-title">{conv.title || "Archived Conversation"}</h2>
          <span className="history-detail-date">
            {new Date(conv.created_at_ms || conv.date || Date.now()).toLocaleString()}
          </span>
        </div>
        <button
          className="btn btn-primary"
          onClick={onLoadToActive}
          disabled={disabled}
          title={disabled ? "Stop recording to load" : "Load into active transcript"}
        >
          Load Into Session
        </button>
      </div>

      {profiles.length > 0 && (
        <div className="history-detail-profiles">
          <strong>Attached Profiles:</strong>{" "}
          {profiles.map((p: any) => p.name || p.profile_id).join(", ")}
        </div>
      )}

      {summary && (
        <div className="history-detail-summary-card">
          <h3 className="history-detail-section-title">Meeting Summary</h3>
          <div className="history-detail-summary-text">
            {summary.summary_markdown || summary}
          </div>
        </div>
      )}

      <div className="history-detail-messages-section">
        <h3 className="history-detail-section-title">
          Messages ({messages.length})
        </h3>
        {messages.length === 0 ? (
          <div className="history-empty">No messages in this conversation.</div>
        ) : (
          <div className="history-detail-messages-list">
            {messages.map((m: any, idx: number) => {
              const speaker = m.speaker_label || (m.source === "mic" ? "Me" : "Others");
              const text = m.stt_text || m.text || "";
              const translation = m.translation_text || m.translation || null;
              return (
                <div key={m.message_id || idx} className="history-detail-message">
                  <div className="history-detail-message-meta">
                    <span className="history-detail-speaker">{speaker}</span>
                    {m.timestamp_ms && (
                      <span className="history-detail-time">
                        {new Date(m.timestamp_ms).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div className="history-detail-message-text">{text}</div>
                  {translation && (
                    <div className="history-detail-message-trans">🌐 {translation}</div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
