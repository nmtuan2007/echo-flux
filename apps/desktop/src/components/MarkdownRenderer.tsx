import React from "react";

/**
 * Lightweight markdown renderer — handles headings, bold, italic, inline code,
 * and bullet/numbered lists without a heavy runtime dependency.
 */
export function MarkdownRenderer({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("### ")) {
      elements.push(
        <h3 key={i} className="summary-h3">
          {parseInline(line.slice(4))}
        </h3>,
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h2 key={i} className="summary-h2">
          {parseInline(line.slice(3))}
        </h2>,
      );
    } else if (line.startsWith("# ")) {
      elements.push(
        <h1 key={i} className="summary-h1">
          {parseInline(line.slice(2))}
        </h1>,
      );
    } else if (line.match(/^[-*] /)) {
      elements.push(
        <div key={i} className="summary-list-item">
          <span className="summary-bullet">•</span>
          <span>{parseInline(line.slice(2))}</span>
        </div>,
      );
    } else if (line.match(/^\d+\. /)) {
      const content = line.replace(/^\d+\. /, "");
      const num = line.match(/^(\d+)\./)?.[1];
      elements.push(
        <div key={i} className="summary-list-item summary-numbered">
          <span className="summary-number">{num}.</span>
          <span>{parseInline(content)}</span>
        </div>,
      );
    } else if (line.trim() === "") {
      elements.push(<div key={i} className="summary-spacer" />);
    } else {
      elements.push(
        <p key={i} className="summary-para">
          {parseInline(line)}
        </p>,
      );
    }
  }

  return <div className="summary-markdown">{elements}</div>;
}

function parseInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
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
      parts.push(
        <code key={key++} className="summary-inline-code">
          {token.slice(1, -1)}
        </code>,
      );
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 ? parts[0] : parts;
}
