import { useCallback, useEffect, useRef } from "react";
import { useEngineStore } from "../store/engineStore";

export function TranscriptPanel() {
  const { entries, partialText, partialTranslation, config, running } = useEngineStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const isNearBottom = useRef(true);

  const checkNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const threshold = 80;
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  }, []);

  useEffect(() => {
    if (isNearBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, partialText]);

  const hasContent = entries.length > 0 || partialText;

  return (
    <div className="transcript-panel" ref={scrollRef} onScroll={checkNearBottom}>
      {!hasContent && (
        <div className="transcript-empty">
          <p>No transcript yet.</p>
          <p>
            Press <strong>Start</strong> to begin transcription.
          </p>
        </div>
      )}

      {entries.map((entry) => (
        <div key={entry.id} className="transcript-entry transcript-final">
          <div className="transcript-text">{entry.text}</div>
          {config.translationEnabled && entry.translation && (
            <div className="transcript-translation">{entry.translation}</div>
          )}
          <span className="transcript-time">{formatTime(entry.timestamp)}</span>
        </div>
      ))}

      {partialText && (
        <div className="transcript-entry transcript-partial">
          <div className="transcript-text">
            {partialText}
            <span className="typing-cursor" />
          </div>
          {config.translationEnabled && partialTranslation && (
            <div className="transcript-translation">{partialTranslation}</div>
          )}
        </div>
      )}

      {running && !partialText && entries.length > 0 && (
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
