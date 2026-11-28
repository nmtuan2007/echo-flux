import { useMemo, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { isAssistantConfigured } from "../store/transcriptSlice";
import type { ExportEntry } from "../utils/export";
import { ExportMenu } from "./ExportMenu";
import { MarkdownRenderer } from "./MarkdownRenderer";

export interface ConversationDetailViewProps {
  detail: any;
  onBack: () => void;
  onLoadToActive: () => void;
  onContinueSession: () => void;
  onExportBackup: () => void;
  disabled: boolean;
}

/**
 * Archived Conversation Detail Viewer
 */
export function ConversationDetailView({
  detail,
  onBack,
  onLoadToActive,
  onContinueSession,
  onExportBackup,
  disabled,
}: ConversationDetailViewProps) {
  const conv = detail.conversation || detail;
  const convId: string | undefined = conv.conversation_id || detail.conversation_id;
  const messages = detail.messages || [];
  const sessions = detail.sessions || [];
  const assistResults = detail.assist_results || [];
  const askTurns = detail.ask_turns || [];
  const summary = detail.summary;
  const profiles = detail.profiles || [];
  const summaryMarkdown =
    typeof summary === "string" ? summary : summary?.summary_markdown || "";

  const connected = useEngineStore((s) => s.connected);
  const askLoading = useEngineStore((s) => s.askLoading);
  const askError = useEngineStore((s) => s.askError);
  const askMeeting = useEngineStore((s) => s.askMeeting);
  const clearAsk = useEngineStore((s) => s.clearAsk);
  const assistantConfigured = useEngineStore(isAssistantConfigured);
  const [askInput, setAskInput] = useState("");

  const sessionMap = useMemo(() => {
    const map = new Map<string, { num: number; startedAtMs?: number }>();
    sessions.forEach((s: any, idx: number) => {
      if (s?.session_id) {
        map.set(s.session_id, { num: idx + 1, startedAtMs: s.started_at_ms });
      }
    });
    return map;
  }, [sessions]);

  const assistByMessageId = useMemo(() => {
    const map = new Map<string, any>();
    for (const row of assistResults) {
      if (row?.message_id) {
        map.set(row.message_id, row);
      }
    }
    return map;
  }, [assistResults]);

  const handleSubmitAsk = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = askInput.trim();
    if (!trimmed || !convId || askLoading || !connected || !assistantConfigured) return;
    askMeeting(trimmed, convId);
    setAskInput("");
  };

  const toExportEntries = (): ExportEntry[] =>
    messages.map((m: any) => {
      const tsMs = typeof m.timestamp_ms === "number" ? m.timestamp_ms : Date.now();
      return {
        text: m.stt_text || m.text || "",
        translation: m.translation_text ?? m.translation ?? null,
        timestamp: tsMs > 1e12 ? tsMs / 1000 : tsMs,
        startMs: typeof m.start_ms === "number" ? m.start_ms : null,
        endMs: typeof m.end_ms === "number" ? m.end_ms : null,
        speaker:
          m.speaker_label ||
          conv.source_labels?.[m.source] ||
          (m.source === "mic" ? "Me" : "Others"),
      };
    });

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
        <div className="history-detail-actions">
          <ExportMenu
            entries={toExportEntries()}
            title={conv.title || "EchoFlux Conversation"}
            summaryMarkdown={summaryMarkdown || null}
            onExportBackup={onExportBackup}
          />
          <button
            className="btn btn-secondary"
            onClick={onLoadToActive}
            disabled={disabled}
            title={disabled ? "Stop recording to load" : "View in transcript workspace (read-only)"}
          >
            Load Into Session
          </button>
          <button
            className="btn btn-primary"
            onClick={onContinueSession}
            disabled={disabled}
            title={disabled ? "Stop recording to continue" : "Append new recording to this conversation"}
          >
            Continue Recording
          </button>
        </div>
      </div>

      {profiles.length > 0 && (
        <div className="history-detail-profiles">
          <strong>Attached Profiles:</strong>{" "}
          {profiles.map((p: any) => p.name || p.profile_id).join(", ")}
        </div>
      )}

      {summaryMarkdown && (
        <div className="history-detail-summary-card">
          <h3 className="history-detail-section-title">Meeting Summary</h3>
          <div className="history-detail-summary-text">
            <MarkdownRenderer text={summaryMarkdown} />
          </div>
        </div>
      )}

      {(askTurns.length > 0 || (assistantConfigured && connected && convId)) && (
        <div className="history-detail-ask">
          <div className="history-detail-ask-header">
            <h3 className="history-detail-section-title">Quick Ask History</h3>
            {askTurns.length > 0 && convId && (
              <button
                type="button"
                className="btn-text"
                onClick={() => clearAsk(convId)}
              >
                Clear Q&amp;A
              </button>
            )}
          </div>
          {askTurns.length > 0 && (
            <div className="history-detail-ask-list">
              {askTurns.map((turn: any, idx: number) => (
                <div key={turn.ask_id || idx} className="history-detail-ask-turn">
                  <div className="history-detail-ask-q">Q: {turn.question}</div>
                  {turn.error ? (
                    <div className="history-detail-ask-error">⚠ {turn.error}</div>
                  ) : (
                    <div className="history-detail-ask-a">
                      <MarkdownRenderer text={turn.answer_markdown ?? turn.answer ?? ""} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {assistantConfigured && connected && convId && (
            <form className="summary-ask-form" onSubmit={handleSubmitAsk}>
              <input
                type="text"
                className="summary-ask-input"
                placeholder="Ask a follow-up question about this archived meeting..."
                value={askInput}
                onChange={(e) => setAskInput(e.target.value)}
                disabled={askLoading}
                aria-label="Ask about this archived conversation"
              />
              <button
                type="submit"
                className="btn btn-primary btn-sm"
                disabled={askLoading || !askInput.trim()}
              >
                {askLoading ? "Asking..." : "Ask"}
              </button>
            </form>
          )}
          {askError && <div className="summary-ask-error">⚠ {askError}</div>}
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
              const speaker =
                m.speaker_label ||
                conv.source_labels?.[m.source] ||
                (m.source === "mic" ? "Me" : "Others");
              const text = m.stt_text || m.text || "";
              const translation = m.translation_text || m.translation || null;
              const prevSessionId = idx > 0 ? messages[idx - 1]?.session_id : null;
              const showSessionDivider =
                sessions.length > 1 && m.session_id && m.session_id !== prevSessionId;
              const sessionMeta = m.session_id ? sessionMap.get(m.session_id) : undefined;
              const assist = m.message_id ? assistByMessageId.get(m.message_id) : undefined;
              const replies = assist?.suggested_replies || assist?.options || [];
              const questions = assist?.clarifying_questions || [];
              return (
                <div key={m.message_id || idx} className="history-detail-message-wrap">
                  {showSessionDivider && (
                    <div className="history-session-divider">
                      <span>
                        Session #{sessionMeta?.num ?? "?"}
                        {sessionMeta?.startedAtMs
                          ? ` · ${new Date(sessionMeta.startedAtMs).toLocaleString()}`
                          : ""}
                      </span>
                    </div>
                  )}
                  <div className="history-detail-message">
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
                    {(replies.length > 0 || questions.length > 0) && (
                      <div className="history-detail-assist">
                        {replies.map((r: any, rIdx: number) => (
                          <div key={rIdx} className="history-detail-assist-item">
                            <span className="history-detail-assist-tag">
                              {typeof r === "string" ? "Reply" : r.strategy || "Reply"}
                            </span>
                            <span>{typeof r === "string" ? r : r.text}</span>
                          </div>
                        ))}
                        {questions.map((q: any, qIdx: number) => (
                          <div key={`q-${qIdx}`} className="history-detail-assist-item">
                            <span className="history-detail-assist-tag">Question</span>
                            <span>{typeof q === "string" ? q : q.question}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
