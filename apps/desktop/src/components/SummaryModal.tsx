import { useEffect, useRef } from "react";
import { useEngineStore } from "../store/engineStore";

export function SummaryModal() {
  const { summaryVisible, summaryLoading, summaryText, closeSummary } = useEngineStore();
  const contentRef = useRef<HTMLDivElement>(null);

  // Auto-scroll as content streams in
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [summaryText]);

  if (!summaryVisible) return null;

  const handleCopyAll = async () => {
    try {
      await navigator.clipboard.writeText(summaryText);
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
            {summaryText && !summaryLoading && (
              <button className="btn btn-secondary summary-copy-all-btn" onClick={handleCopyAll}>
                📋 Copy All
              </button>
            )}
            <button className="btn-icon summary-close-btn" onClick={closeSummary} title="Close">
              <CloseIcon />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="summary-content" ref={contentRef}>
          {summaryText ? (
            <MarkdownRenderer text={summaryText} />
          ) : (
            <div className="summary-placeholder">
              <span className="summary-placeholder-spinner" />
              <span>Analyzing your conversation...</span>
            </div>
          )}
          {/* Typewriter cursor while loading */}
          {summaryLoading && summaryText && (
            <span className="typing-cursor" style={{ display: "inline-block", marginLeft: 2 }} />
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Very lightweight markdown renderer — handles bold, italic, code, and lists
 * without a full markdown library dependency.
 */
function MarkdownRenderer({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Heading
    if (line.startsWith("### ")) {
      elements.push(<h3 key={i} className="summary-h3">{parseInline(line.slice(4))}</h3>);
    } else if (line.startsWith("## ")) {
      elements.push(<h2 key={i} className="summary-h2">{parseInline(line.slice(3))}</h2>);
    } else if (line.startsWith("# ")) {
      elements.push(<h1 key={i} className="summary-h1">{parseInline(line.slice(2))}</h1>);
    }
    // Bullet list item
    else if (line.match(/^[-*] /)) {
      elements.push(
        <div key={i} className="summary-list-item">
          <span className="summary-bullet">•</span>
          <span>{parseInline(line.slice(2))}</span>
        </div>
      );
    }
    // Numbered list item
    else if (line.match(/^\d+\. /)) {
      const content = line.replace(/^\d+\. /, "");
      const num = line.match(/^(\d+)\./)?.[1];
      elements.push(
        <div key={i} className="summary-list-item summary-numbered">
          <span className="summary-number">{num}.</span>
          <span>{parseInline(content)}</span>
        </div>
      );
    }
    // Blank line
    else if (line.trim() === "") {
      elements.push(<div key={i} className="summary-spacer" />);
    }
    // Paragraph
    else {
      elements.push(<p key={i} className="summary-para">{parseInline(line)}</p>);
    }
  }

  return <div className="summary-markdown">{elements}</div>;
}

/** Render inline bold (**text**), italic (*text*), and code (`text`) */
function parseInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  // Split on **bold**, *italic*, `code`
  const regex = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("**")) {
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("*")) {
      parts.push(<em key={key++}>{token.slice(1, -1)}</em>);
    } else if (token.startsWith("`")) {
      parts.push(<code key={key++} className="summary-inline-code">{token.slice(1, -1)}</code>);
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 ? parts[0] : parts;
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  );
}
