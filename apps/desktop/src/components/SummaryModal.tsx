import { useEffect, useRef, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { isAssistantConfigured } from "../store/transcriptSlice";
import { MarkdownRenderer } from "./MarkdownRenderer";

export function SummaryModal() {
  const {
    summaryVisible,
    summaryLoading,
    summaryText,
    summaryWarning,
    summaryError,
    requestSummary,
    cancelSummary,
    closeSummary,
    askAnswer,
    askHistory,
    askLoading,
    askError,
    askMeeting,
    clearAsk,
    connected,
    entries,
    activeConversationId,
    engineConversationId,
  } = useEngineStore();
  const isAssistantEnabled = useEngineStore(isAssistantConfigured);
  const contentRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const [copiedTurnId, setCopiedTurnId] = useState<string | null>(null);
  const [questionDraft, setQuestionDraft] = useState("");
  const canAsk = Boolean(
    isAssistantEnabled && connected && (engineConversationId || activeConversationId),
  );
  const canGenerateSummary = Boolean(
    isAssistantEnabled && connected && (engineConversationId || activeConversationId || entries.length > 0),
  );

  // Auto-scroll as content streams in
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [summaryText]);

  useEffect(() => {
    if (!summaryVisible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSummary();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [summaryVisible, closeSummary]);

  if (!summaryVisible) return null;

  const handleCopyAll = async () => {
    try {
      await navigator.clipboard.writeText(summaryText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable
    }
  };

  return (
    <div className="summary-overlay" onClick={closeSummary}>
      <div className="summary-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="summary-header">
          <div className="summary-title">
            <span className="summary-title-icon">📝</span>
            Meeting Summary
            {summaryLoading && (
              <span className="summary-loading-badge">
                <span className="summary-loading-dot" />
                <span className="summary-loading-dot" />
                <span className="summary-loading-dot" />
              </span>
            )}
          </div>
          <div className="summary-header-actions">
            {canGenerateSummary && !summaryLoading && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={requestSummary}
                title={summaryText ? "Regenerate meeting summary" : "Generate meeting summary"}
              >
                {summaryText ? "↻ Regenerate" : "✨ Generate Summary"}
              </button>
            )}
            {summaryLoading && (
              <button
                className="btn btn-secondary summary-cancel-btn"
                onClick={cancelSummary}
                title="Cancel streaming summary"
              >
                ⏹ Stop
              </button>
            )}
            {summaryText && !summaryLoading && (
              <button className="btn btn-secondary summary-copy-all-btn" onClick={handleCopyAll}>
                {copied ? "✓ Copied" : "📋 Copy All"}
              </button>
            )}
            <button className="btn-icon summary-close-btn" onClick={closeSummary} title="Close">
              <CloseIcon />
            </button>
          </div>
        </div>

        {summaryError && (
          <div className="summary-error-banner" role="alert">
            <span>⚠️ {summaryError}</span>
          </div>
        )}

        {summaryWarning && summaryWarning.omitted_message_count > 0 && (
          <div className="summary-warning-banner" role="status">
            <span>
              ⚠️ Transcript was truncated: {summaryWarning.omitted_message_count} earlier message
              {summaryWarning.omitted_message_count > 1 ? "s" : ""} omitted to fit model context
              {summaryWarning.used_tokens && summaryWarning.max_tokens
                ? ` (${summaryWarning.used_tokens}/${summaryWarning.max_tokens} tokens)`
                : ""}
              . Summary covers the most recent {summaryWarning.included_message_count} messages.
            </span>
          </div>
        )}

        {/* Content */}
        <div className="summary-content" ref={contentRef}>
          {summaryText ? (
            summaryLoading ? (
              <div className="summary-markdown" style={{ whiteSpace: "pre-wrap" }}>
                {summaryText}
              </div>
            ) : (
              <MarkdownRenderer text={summaryText} />
            )
          ) : summaryLoading ? (
            <div className="summary-placeholder">
              <span className="summary-placeholder-spinner" />
              <span>Analyzing your conversation...</span>
            </div>
          ) : summaryError ? (
            <div className="summary-placeholder">
              <span>Failed to generate summary.</span>
            </div>
          ) : (
            <div className="summary-placeholder">
              <span>No summary generated yet. Ask a quick question below or run Summarize.</span>
            </div>
          )}
          {/* Typewriter cursor while loading */}
          {summaryLoading && summaryText && (
            <span className="typing-cursor" style={{ display: "inline-block", marginLeft: 2 }} />
          )}

          {canAsk && (
            <div className="summary-ask-box" aria-label="Quick Ask">
              <div className="summary-ask-header">
                <span className="summary-ask-title">💬 Quick Ask</span>
                {(askHistory.length > 0 || askAnswer) && (
                  <button type="button" className="btn-text" onClick={() => clearAsk()}>
                    Clear
                  </button>
                )}
              </div>
              <form
                className="summary-ask-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!questionDraft.trim() || askLoading) return;
                  void askMeeting(questionDraft).then(() => {
                    if (!useEngineStore.getState().askError) {
                      setQuestionDraft("");
                    }
                  });
                }}
              >
                <input
                  type="text"
                  className="summary-ask-input"
                  value={questionDraft}
                  onChange={(e) => setQuestionDraft(e.target.value)}
                  placeholder="Ask about decisions, deadlines, or what someone said..."
                  disabled={askLoading}
                  aria-label="Ask AI about this meeting"
                />
                <button
                  type="submit"
                  className="btn btn-primary btn-sm"
                  disabled={!questionDraft.trim() || askLoading}
                >
                  {askLoading ? "Asking..." : "Ask"}
                </button>
              </form>
              {askError && <p className="summary-ask-error">{askError}</p>}
              {askHistory.length > 0 ? (
                <div className="summary-ask-history">
                  {askHistory.map((turn) => (
                    <div key={turn.id} className="summary-ask-turn">
                      <div className="summary-ask-question-row">
                        <span className="summary-ask-question">{turn.question}</span>
                        <button
                          type="button"
                          className="btn-text summary-ask-copy"
                          onClick={async () => {
                            try {
                              await navigator.clipboard.writeText(turn.answer);
                              setCopiedTurnId(turn.id);
                              setTimeout(() => setCopiedTurnId((cur) => (cur === turn.id ? null : cur)), 2000);
                            } catch {
                              // Clipboard API unavailable
                            }
                          }}
                        >
                          {copiedTurnId === turn.id ? "✓ Copied" : "Copy"}
                        </button>
                      </div>
                      <div className="summary-ask-answer">
                        <MarkdownRenderer text={turn.answer} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                askAnswer && (
                  <div className="summary-ask-answer">
                    <MarkdownRenderer text={askAnswer} />
                  </div>
                )
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  );
}
