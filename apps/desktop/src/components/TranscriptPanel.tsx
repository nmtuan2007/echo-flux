import { useState, useCallback, useEffect, useRef } from "react";
import { TranscriptEntry, SuggestionOption, useEngineStore } from "../store/engineStore";

type AudioSource = TranscriptEntry["source"];

const SOURCE_LABELS: Record<NonNullable<AudioSource>, { icon: string; label: string; cls: string }> = {
  mic:    { icon: "🎤", label: "Mic",     cls: "source-badge-mic" },
  system: { icon: "🔊", label: "Speaker", cls: "source-badge-system" },
  both:   { icon: "🔄", label: "Both",    cls: "source-badge-both" },
};

const STRATEGY_META: Record<string, { icon: string; cls: string }> = {
  Cooperative: { icon: "🤝", cls: "suggestion-card-cooperative" },
  Clarifying:  { icon: "🔍", cls: "suggestion-card-clarifying" },
  Assertive:   { icon: "💪", cls: "suggestion-card-assertive" },
};

function SourceBadge({ source }: { source: AudioSource }) {
  if (!source) return null;
  const meta = SOURCE_LABELS[source];
  if (!meta) return null;
  return (
    <span className={`source-badge ${meta.cls}`}>
      {meta.icon} {meta.label}
    </span>
  );
}

function sourceBorderClass(source: AudioSource): string {
  if (!source) return "";
  return `transcript-source-${source}`;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable
    }
  };

  return (
    <button className="suggestion-copy-btn" onClick={handleCopy} title="Copy to clipboard">
      {copied ? "✓" : "Copy"}
    </button>
  );
}

function SuggestionCards({ options, error }: { options?: SuggestionOption[]; error?: string }) {
  if (error) {
    return <div className="suggestion-error">⚠ {error}</div>;
  }
  if (!options || options.length === 0) return null;

  return (
    <div className="suggestion-cards">
      {options.map((opt, i) => {
        const meta = STRATEGY_META[opt.strategy] ?? { icon: "💬", cls: "suggestion-card-default" };
        return (
          <div key={i} className={`suggestion-card ${meta.cls}`}>
            <div className="suggestion-card-header">
              <span className="suggestion-strategy">{meta.icon} {opt.strategy}</span>
              <CopyButton text={opt.text} />
            </div>
            <p className="suggestion-text">{opt.text}</p>
          </div>
        );
      })}
    </div>
  );
}

function FinalEntry({ entry, index }: {
  entry: TranscriptEntry;
  index: number;
}) {
  const { config, suggestionLoading, requestSuggestion } = useEngineStore();
  // Local UI state — expand/collapse, not persisted
  const [showSuggestions, setShowSuggestions] = useState(false);

  const isLoading = suggestionLoading[entry.id] ?? false;
  const hasSuggestion = !!entry.suggestions;

  const handleSuggest = (e: React.MouseEvent) => {
    e.stopPropagation();
    const contextEntries = useEngineStore.getState().entries;
    const contextSlice = contextEntries
      .slice(Math.max(0, index - 5), index)
      .map((e) => e.text);
    requestSuggestion(entry.id, entry.text, contextSlice);
    // Auto-expand after requesting
    setShowSuggestions(true);
  };

  // Auto-show when result arrives
  useEffect(() => {
    if (hasSuggestion && !isLoading) {
      setShowSuggestions(true);
    }
  }, [hasSuggestion, isLoading]);

  return (
    <div className={`transcript-entry transcript-final ${sourceBorderClass(entry.source)}`}>
      {entry.source && (
        <div className="transcript-entry-header">
          <SourceBadge source={entry.source} />
        </div>
      )}
      <div className="transcript-text">{entry.text}</div>
      {config.translationEnabled && entry.translation && (
        <div className="transcript-translation">{entry.translation}</div>
      )}
      <span className="transcript-time">{formatTime(entry.timestamp)}</span>

      {/* AI Suggestion area */}
      {config.llmEnabled && (
        <div className="suggest-action-row">
          {/* Loading state */}
          {isLoading && (
            <div className="suggestion-spinner">
              <span className="suggestion-spinner-dot" />
              <span className="suggestion-spinner-dot" />
              <span className="suggestion-spinner-dot" />
              <span className="suggestion-spinner-text">AI is thinking...</span>
            </div>
          )}

          {/* Not loading, no suggestion yet → Get button */}
          {!isLoading && !hasSuggestion && (
            <button className="suggest-btn" onClick={handleSuggest} title="Get AI reply suggestions">
              💡 Suggest Reply
            </button>
          )}

          {/* Has suggestion → Show/Hide toggle + Refresh */}
          {!isLoading && hasSuggestion && (
            <div className="suggest-toggle-row">
              <button
                className={`suggest-btn suggest-btn-toggle ${showSuggestions ? "active" : ""}`}
                onClick={() => setShowSuggestions((v) => !v)}
              >
                {showSuggestions ? "💡 Hide Reply Ideas" : "💡 Show Reply Ideas"}
              </button>
              <button
                className="suggest-btn suggest-btn-refresh"
                onClick={handleSuggest}
                title="Refresh suggestions"
              >
                ↺
              </button>
            </div>
          )}

          {/* Expanded suggestion cards */}
          {!isLoading && hasSuggestion && showSuggestions && (
            <SuggestionCards
              options={entry.suggestions?.options}
              error={entry.suggestions?.error}
            />
          )}
        </div>
      )}
    </div>
  );
}

export function TranscriptPanel() {
  const { entries, partials, running, downloading, downloadModel, downloadPercent } = useEngineStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const isNearBottom = useRef(true);

  const checkNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  useEffect(() => {
    if (isNearBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, partials]);

  const activePartials = Object.values(partials);
  const hasContent = entries.length > 0 || activePartials.length > 0;

  return (
    <div className="transcript-panel" ref={scrollRef} onScroll={checkNearBottom}>
      {downloading && (
        <div className="download-progress-overlay">
          <div className="download-progress-card">
            <h4>Downloading AI model for the first time...</h4>
            <p>{downloadModel}</p>
            <div className="progress-bar-bg">
              <div className="progress-bar-fill" style={{ width: `${downloadPercent}%` }}></div>
            </div>
            <div className="progress-text-container">
              <span className="progress-text">{downloadPercent}%</span>
            </div>
          </div>
        </div>
      )}

      {!hasContent && !downloading && (
        <div className="transcript-empty">
          <p>No transcript yet.</p>
          <p>Press <strong>Start</strong> to begin transcription.</p>
        </div>
      )}

      {entries.map((entry, index) => (
        <FinalEntry key={entry.id} entry={entry} index={index} />
      ))}

      {activePartials.map((partial, idx) => (
        <div
          key={`partial-${idx}`}
          className={`transcript-entry transcript-partial ${sourceBorderClass(partial.source)}`}
        >
          {partial.source && (
            <div className="transcript-entry-header">
              <SourceBadge source={partial.source} />
            </div>
          )}
          <div className="transcript-text">
            {partial.text}
            <span className="typing-cursor" />
          </div>
          {useEngineStore.getState().config.translationEnabled && partial.translation && (
            <div className="transcript-translation">{partial.translation}</div>
          )}
        </div>
      ))}

      {running && activePartials.length === 0 && entries.length > 0 && (
        <div className="transcript-listening">
          <span className="listening-dot" />
          <span className="listening-dot" />
          <span className="listening-dot" />
        </div>
      )}
    </div>
  );
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
