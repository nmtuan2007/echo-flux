import { memo, useCallback, useEffect, useMemo, useRef } from "react";
import { useEngineStore } from "../store/engineStore";
import { deriveSttNotice, streamRank, sttProviderLabel } from "../store/transcriptSlice";
import type {
  ContinuityInterval,
  PartialTranscript,
  TranscriptEntry,
  TranscriptSource,
} from "../store/types";
import { AssistPanel } from "./AssistPanel";

const SOURCE_LABELS: Record<NonNullable<TranscriptSource>, { icon: string; label: string; cls: string }> = {
  mic:    { icon: "🎤", label: "Me",     cls: "source-badge-mic" },
  system: { icon: "🔊", label: "Others", cls: "source-badge-system" },
  both:   { icon: "🔄", label: "Both",   cls: "source-badge-both" },
};

function sourceLabel(source: string): string {
  return SOURCE_LABELS[source as NonNullable<TranscriptSource>]?.label ?? source;
}

function SourceBadge({ source }: { source: TranscriptSource | undefined }) {
  if (!source) return null;
  const meta = SOURCE_LABELS[source];
  if (!meta) return null;
  return (
    <span className={`source-badge ${meta.cls}`}>
      {meta.icon} {meta.label}
    </span>
  );
}

function sourceBorderClass(source: TranscriptSource | undefined): string {
  if (!source) return "";
  return `transcript-source-${source}`;
}

function formatMs(ms: number): string {
  const total = Math.max(0, ms) / 1000;
  const minutes = Math.floor(total / 60);
  const seconds = (total - minutes * 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

function intervalText(interval: ContinuityInterval): string {
  return `${formatMs(interval.startMs)}–${formatMs(interval.endMs)}`;
}

// Memoized on the entry object: the store replaces only entries that changed, so a partial on
// either stream re-renders no final entry.
const FinalEntry = memo(function FinalEntry({ entry }: { entry: TranscriptEntry }) {
  const translationEnabled = useEngineStore((s) => s.config.translationEnabled);

  return (
    <div
      className={`transcript-entry transcript-final ${sourceBorderClass(entry.source)}`}
      data-entry-id={entry.id}
      data-message-id={entry.messageId ?? entry.id}
    >
      {(entry.source || entry.continuity || entry.sttFallbackFrom) && (
        <div className="transcript-entry-header">
          <SourceBadge source={entry.source} />
          {entry.sttFallbackFrom && (
            <span
              className="transcript-fallback-badge"
              title={`Switched to Local from ${sttProviderLabel(entry.sttFallbackFrom)}`}
            >
              {sttProviderLabel(entry.sttProvider ?? "local")} (Sticky Fallback)
            </span>
          )}
          {entry.continuity && (
            <span
              className="transcript-continuity"
              title="Words in this interval may be missing or repeated"
            >
              Continuity uncertain {intervalText(entry.continuity)}
            </span>
          )}
        </div>
      )}
      <div className="transcript-text">{entry.text}</div>
      {translationEnabled && entry.translation && (
        <div
          className="transcript-translation"
          data-translation-state={entry.translationFinal === false ? "draft" : "final"}
        >
          {entry.translation}
        </div>
      )}
      <span className="transcript-time">{formatTime(entry.timestamp)}</span>
      <AssistPanel entry={entry} />
    </div>
  );
});

// Keyed by stream, and memoized on that stream's partial object, so mic and system updates
// never re-render or remount each other.
const PartialEntry = memo(function PartialEntry({
  stream,
  partial,
}: {
  stream: string;
  partial: PartialTranscript;
}) {
  const translationEnabled = useEngineStore((s) => s.config.translationEnabled);

  return (
    <div
      className={`transcript-entry transcript-partial ${sourceBorderClass(partial.source)}`}
      data-stream={stream}
      aria-live="off"
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
      {translationEnabled && partial.translation && (
        <div className="transcript-translation" data-translation-state="draft">
          {partial.translation}
        </div>
      )}
    </div>
  );
});

function SttNoticeBar() {
  const entries = useEngineStore((s) => s.entries);
  const sttStreams = useEngineStore((s) => s.sttStreams);
  const running = useEngineStore((s) => s.running);
  const notice = useMemo(
    () => deriveSttNotice(entries, sttStreams, running),
    [entries, sttStreams, running],
  );

  if (notice.fallbacks.length === 0 && notice.continuityCount === 0) return null;

  return (
    <div className="stt-notice" role="status">
      {notice.fallbacks.map((f) => (
        <span key={f.source} className="stt-notice-item stt-notice-fallback">
          {sourceLabel(f.source)}: switched to Local from {sttProviderLabel(f.from)}
        </span>
      ))}
      {notice.latestContinuity && (
        <span className="stt-notice-item stt-notice-continuity">
          Continuity uncertain in {notice.continuityCount} interval
          {notice.continuityCount !== 1 ? "s" : ""} (latest {intervalText(notice.latestContinuity)})
        </span>
      )}
    </div>
  );
}

export function TranscriptPanel() {
  const entries = useEngineStore((s) => s.entries);
  const partials = useEngineStore((s) => s.partials);
  const running = useEngineStore((s) => s.running);
  const downloading = useEngineStore((s) => s.downloading);
  const downloadModel = useEngineStore((s) => s.downloadModel);
  const downloadPercent = useEngineStore((s) => s.downloadPercent);
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

  const partialStreams = useMemo(
    () => Object.keys(partials).sort((a, b) => streamRank(a) - streamRank(b)),
    [partials],
  );
  const hasContent = entries.length > 0 || partialStreams.length > 0;

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

      <SttNoticeBar />

      {!hasContent && !downloading && (
        <div className="transcript-empty">
          <p>No transcript yet.</p>
          <p>Press <strong>Start</strong> to begin transcription.</p>
        </div>
      )}

      {entries.map((entry) => (
        <FinalEntry key={entry.id} entry={entry} />
      ))}

      {partialStreams.map((stream) => (
        <PartialEntry key={`partial-${stream}`} stream={stream} partial={partials[stream]} />
      ))}

      {running && partialStreams.length === 0 && entries.length > 0 && (
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
