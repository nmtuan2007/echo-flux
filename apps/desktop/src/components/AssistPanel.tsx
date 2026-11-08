import { memo, useEffect, useId, useMemo, useRef, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import {
  ASSIST_UNAVAILABLE_TEXT,
  assistKeyOf,
  assistUnavailableReason,
  deriveAssistView,
  isAssistantConfigured,
} from "../store/transcriptSlice";
import type { AssistError, ClarifyingQuestion, SuggestedReply, TranscriptEntry } from "../store/types";

const STRATEGY_CLASS: Record<string, string> = {
  cooperative: "suggestion-card-cooperative",
  clarifying: "suggestion-card-clarifying",
  assertive: "suggestion-card-assertive",
};

const ERROR_TITLE: Record<AssistError["kind"], string> = {
  provider: "Assistant provider error",
  structured: "Assist could not run",
  interrupted: "Assist interrupted",
};

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable
    }
  };

  return (
    <button className="suggestion-copy-btn" onClick={handleCopy} aria-label={label} title="Copy to clipboard">
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function InsertButton({ text, label }: { text: string; label: string }) {
  const [inserted, setInserted] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const handleInsert = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setInserted(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setInserted(false), 1500);
    } catch {
      // Clipboard API unavailable
    }
  };

  return (
    <button
      className="suggestion-insert-btn"
      onClick={handleInsert}
      aria-label={label}
      title="Insert / Copy to clipboard"
    >
      {inserted ? "Inserted" : "Insert"}
    </button>
  );
}

function ReplyGroup({ replies }: { replies: SuggestedReply[] }) {
  return (
    <div className="assist-group" role="group" aria-label="Suggested replies">
      <h4 className="assist-group-title">Suggested replies</h4>
      {replies.length === 0 ? (
        <p className="assist-group-empty">No suggested replies.</p>
      ) : (
        replies.map((reply, i) => (
          <div
            key={i}
            className={`suggestion-card ${STRATEGY_CLASS[reply.strategy.toLowerCase()] ?? "suggestion-card-default"}`}
          >
            <div className="suggestion-card-header">
              <span className="suggestion-strategy">{reply.strategy || "Reply"}</span>
              <div className="suggestion-card-actions">
                <CopyButton text={reply.text} label={`Copy reply: ${reply.text}`} />
                <InsertButton text={reply.text} label={`Insert reply: ${reply.text}`} />
              </div>
            </div>
            <p className="suggestion-text">{reply.text}</p>
            {reply.reasoning && <p className="assist-note">{reply.reasoning}</p>}
          </div>
        ))
      )}
    </div>
  );
}

function QuestionGroup({ questions }: { questions: ClarifyingQuestion[] }) {
  return (
    <div className="assist-group" role="group" aria-label="Clarifying questions">
      <h4 className="assist-group-title">Clarifying questions</h4>
      {questions.length === 0 ? (
        <p className="assist-group-empty">No clarifying questions.</p>
      ) : (
        questions.map((q, i) => (
          <div key={i} className="suggestion-card assist-question">
            <div className="suggestion-card-header">
              <span className="suggestion-strategy">Question</span>
              <div className="suggestion-card-actions">
                <CopyButton text={q.question} label={`Copy question: ${q.question}`} />
                <InsertButton text={q.question} label={`Insert question: ${q.question}`} />
              </div>
            </div>
            <p className="suggestion-text">{q.question}</p>
            {q.purpose && <p className="assist-note">{q.purpose}</p>}
          </div>
        ))
      )}
    </div>
  );
}

// Per-message Assist: one explicit action on a finalized message, then its result. All state
// and eligibility come from the store; this component only renders and dispatches.
export const AssistPanel = memo(function AssistPanel({ entry }: { entry: TranscriptEntry }) {
  const key = assistKeyOf(entry);
  const configured = useEngineStore(isAssistantConfigured);
  const loading = useEngineStore((s) => s.suggestionLoading[key] === true);
  const request = useEngineStore((s) => s.assistRequests[key]);
  const unavailable = useEngineStore((s) => assistUnavailableReason(s, entry));
  const requestAssist = useEngineStore((s) => s.requestAssist);
  const [expanded, setExpanded] = useState(true);
  const reasonId = useId();

  const view = useMemo(
    () => deriveAssistView({ configured, entry, loading, request, unavailable }),
    [configured, entry, loading, request, unavailable],
  );

  if (view.phase === "hidden") return null;

  const reason = view.unavailable ? (
    <span id={reasonId} className="assist-reason">
      {ASSIST_UNAVAILABLE_TEXT[view.unavailable]}
    </span>
  ) : null;

  const onRequest = (e: React.MouseEvent) => {
    e.stopPropagation();
    requestAssist(entry.id);
  };

  return (
    <section className="assist-panel" data-phase={view.phase} aria-label="Assist">
      {(view.phase === "idle" || view.phase === "unavailable") && (
        <div className="assist-row">
          <button
            className="suggest-btn assist-btn"
            onClick={onRequest}
            disabled={!view.canRequest}
            aria-describedby={reason ? reasonId : undefined}
            title="Suggest replies and questions for this message"
          >
            Assist
          </button>
          {reason}
        </div>
      )}

      {view.phase === "loading" && (
        <div className="suggestion-spinner" role="status">
          <span className="suggestion-spinner-dot" />
          <span className="suggestion-spinner-dot" />
          <span className="suggestion-spinner-dot" />
          <span className="suggestion-spinner-text">Assist is thinking...</span>
        </div>
      )}

      {view.phase === "success" && (
        <>
          <button
            className={`suggest-btn suggest-btn-toggle ${expanded ? "active" : ""}`}
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? "Hide Assist" : "Show Assist"}
          </button>
          {expanded && (
            <div className="suggestion-cards assist-split-view">
              <ReplyGroup replies={view.suggestedReplies} />
              <QuestionGroup questions={view.clarifyingQuestions} />
            </div>
          )}
        </>
      )}

      {view.phase === "empty" && (
        <div className="assist-row">
          <span className="assist-empty">No suggestions for this message.</span>
          <button className="suggest-btn assist-retry-btn" onClick={onRequest} disabled={!view.canRequest}>
            Try again
          </button>
          {reason}
        </div>
      )}

      {view.phase === "error" && view.error && (
        <div className="assist-error" role="alert" data-kind={view.error.kind}>
          <span className="assist-error-title">{ERROR_TITLE[view.error.kind]}</span>
          <span className="assist-error-message">{view.error.message}</span>
          {view.error.retryable && (
            <div className="assist-row">
              <button
                className="suggest-btn assist-retry-btn"
                onClick={onRequest}
                disabled={!view.canRequest}
                aria-describedby={reason ? reasonId : undefined}
              >
                Retry
              </button>
              {reason}
            </div>
          )}
        </div>
      )}
    </section>
  );
});
