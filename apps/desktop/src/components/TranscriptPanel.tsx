import { useEffect, useRef } from "react";
import { useEngineStore } from "../store/engineStore";

export function TranscriptPanel() {
  const { entries, partialText, partialTranslation, config } = useEngineStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, partialText]);

  const hasContent = entries.length > 0 || partialText;

  return (
    <div className="transcript-panel" ref={scrollRef}>
      {!hasContent && (
        <div className="transcript-empty">
          <p>No transcript yet.</p>
          <p>
            Press <strong>Start</strong> to begin transcription.
          </p>
        </div>
      )}

      {entries.map((entry) => (
        <div key={entry.id} className="transcript-entry">
          <div className="transcript-text">{entry.text}</div>
          {config.translationEnabled && entry.translation && (
            <div className="transcript-translation">{entry.translation}</div>
          )}
          <span className="transcript-time">{formatTime(entry.timestamp)}</span>
        </div>
      ))}

      {partialText && (
        <div className="transcript-entry transcript-partial">
          <div className="transcript-text">{partialText}</div>
          {config.translationEnabled && partialTranslation && (
            <div className="transcript-translation">{partialTranslation}</div>
          )}
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
