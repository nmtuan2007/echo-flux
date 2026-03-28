import { useCallback, useEffect, useRef } from "react";
import { TranscriptEntry, useEngineStore } from "../store/engineStore";

type AudioSource = TranscriptEntry["source"];

const SOURCE_LABELS: Record<NonNullable<AudioSource>, { icon: string; label: string; cls: string }> = {
  mic:    { icon: "🎤", label: "Mic",     cls: "source-badge-mic" },
  system: { icon: "🔊", label: "Speaker", cls: "source-badge-system" },
  both:   { icon: "🔄", label: "Both",    cls: "source-badge-both" },
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

export function TranscriptPanel() {
  const { entries, partials, config, running } = useEngineStore();
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
  }, [entries, partials]);

  const activePartials = Object.values(partials);
  const hasContent = entries.length > 0 || activePartials.length > 0;

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
        <div
          key={entry.id}
          className={`transcript-entry transcript-final ${sourceBorderClass(entry.source)}`}
        >
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
        </div>
      ))}

      {activePartials.map((partial, idx) => (
        <div key={`partial-${idx}`} className={`transcript-entry transcript-partial ${sourceBorderClass(partial.source)}`}>
          {partial.source && (
            <div className="transcript-entry-header">
              <SourceBadge source={partial.source} />
            </div>
          )}
          <div className="transcript-text">
            {partial.text}
            <span className="typing-cursor" />
          </div>
          {config.translationEnabled && partial.translation && (
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
