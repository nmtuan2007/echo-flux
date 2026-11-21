import { useEffect, useRef, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { MarkdownRenderer } from "./MarkdownRenderer";

export function SummaryModal() {
  const {
    summaryVisible,
    summaryLoading,
    summaryText,
    summaryWarning,
    summaryError,
    cancelSummary,
    closeSummary,
  } = useEngineStore();
  const contentRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

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
            <MarkdownRenderer text={summaryText} />
          ) : summaryLoading ? (
            <div className="summary-placeholder">
              <span className="summary-placeholder-spinner" />
              <span>Analyzing your conversation...</span>
            </div>
          ) : summaryError ? (
            <div className="summary-placeholder">
              <span>Failed to generate summary.</span>
            </div>
          ) : null}
          {/* Typewriter cursor while loading */}
          {summaryLoading && summaryText && (
            <span className="typing-cursor" style={{ display: "inline-block", marginLeft: 2 }} />
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
