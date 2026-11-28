import { useEffect, useRef, useState } from "react";
import { isTauriRuntime } from "../lib/engineBridge";
import { useEngineStore, Conversation } from "../store/engineStore";
import { ConversationDetailView } from "./ConversationDetailView";

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
    detailLoading,
    historySearchQuery,
    selectedConversationDetail,
    running,
    entries,
    isViewingHistory,
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
  const [pendingOverwrite, setPendingOverwrite] = useState<{
    mode: "load" | "continue";
    convId: string;
  } | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isTauriRuntime()) {
      loadHistoryList().catch(() => {});
      return;
    }
    const t = setTimeout(() => {
      loadHistoryList(historySearchQuery).catch(() => {});
    }, 200);
    return () => clearTimeout(t);
  }, [historySearchQuery, loadHistoryList]);

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

  // Filter items by search query (in Tauri mode, SQLite already searched title + message text)
  const query = historySearchQuery.trim().toLowerCase();
  const filteredItems = isTauriRuntime()
    ? displayItems
    : displayItems.filter((item) =>
        query
          ? item.title.toLowerCase().includes(query) ||
            Boolean(
              item.legacyData?.entries?.some(
                (e) =>
                  e.text.toLowerCase().includes(query) ||
                  Boolean(e.translation && e.translation.toLowerCase().includes(query)),
              ),
            )
          : true,
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
      const exported = await exportConversation(item.id);
      if (typeof exported === "string") {
        jsonString = exported;
      } else if (exported && typeof exported === "object") {
        jsonString = JSON.stringify(exported, null, 2);
      }
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
      {detailLoading ? (
        <div className="history-loading">Loading conversation details...</div>
      ) : selectedConversationDetail ? (
        <ConversationDetailView
          detail={selectedConversationDetail}
          onBack={clearSelectedConversationDetail}
          onLoadToActive={() => {
            const convId =
              selectedConversationDetail.conversation?.conversation_id ||
              selectedConversationDetail.conversation_id;
            if (!convId || running) return;
            if (entries.length > 0 && !isViewingHistory) {
              setPendingOverwrite({ mode: "load", convId });
              return;
            }
            loadConversation(convId);
            clearSelectedConversationDetail();
          }}
          onContinueSession={() => {
            const convId =
              selectedConversationDetail.conversation?.conversation_id ||
              selectedConversationDetail.conversation_id;
            if (!convId || running) return;
            if (entries.length > 0 && !isViewingHistory) {
              setPendingOverwrite({ mode: "continue", convId });
              return;
            }
            loadConversation(convId, { continueSession: true });
            clearSelectedConversationDetail();
          }}
          onExportBackup={async () => {
            const convId =
              selectedConversationDetail.conversation?.conversation_id ||
              selectedConversationDetail.conversation_id;
            if (!convId) return;
            const legacyMatch = history.find((h) => h.id === convId);
            await handleExport({
              id: convId,
              title: selectedConversationDetail.conversation?.title || convId,
              date: Date.now(),
              messageCount: 0,
              hasSummary: false,
              isLegacy: Boolean(legacyMatch),
              legacyData: legacyMatch,
            });
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
                placeholder="Search titles or transcript..."
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
                      title="Export Full Backup (.json)"
                      aria-label="Export Full Backup (.json)"
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

      {/* Overwrite Active Transcript Confirmation Modal */}
      {pendingOverwrite && (
        <div
          className="history-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Replace live transcript view"
          onClick={() => setPendingOverwrite(null)}
        >
          <div className="history-modal-content" onClick={(e) => e.stopPropagation()}>
            <h3 className="history-modal-title">Replace Active Session View?</h3>
            <p className="history-modal-body">
              Your live workspace currently has {entries.length} transcript item
              {entries.length === 1 ? "" : "s"}. Loading this conversation will replace the current
              transcript panel view.
            </p>
            <div className="history-modal-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setPendingOverwrite(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  const target = pendingOverwrite;
                  setPendingOverwrite(null);
                  loadConversation(target.convId, {
                    continueSession: target.mode === "continue",
                  });
                  clearSelectedConversationDetail();
                }}
              >
                {pendingOverwrite.mode === "continue" ? "Continue Session" : "Load Session"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
