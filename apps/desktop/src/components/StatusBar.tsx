import { useMemo } from "react";
import { useEngineStore } from "../store/engineStore";
import { isAssistantConfigured, streamRank, sttProviderLabel } from "../store/transcriptSlice";

const STREAM_LABELS: Record<string, string> = { mic: "Me", system: "Others" };

function languageLabel(code: string): string {
  return !code || code === "auto" ? "Auto" : code.toUpperCase();
}

export function StatusBar() {
  const connected = useEngineStore((s) => s.connected);
  const running = useEngineStore((s) => s.running);
  const entryCount = useEngineStore((s) => s.entries.length);
  const sourceLabels = useEngineStore((s) => s.sourceLabels);
  const language = useEngineStore(
    (s) => (s.services?.stt?.options?.language as string) || s.config.language,
  );
  const translationEnabled = useEngineStore(
    (s) => s.services?.translation?.enabled ?? s.config.translationEnabled,
  );
  const targetLang = useEngineStore(
    (s) => (s.services?.translation?.options?.target_lang as string) || s.config.targetLang,
  );
  const sttStreams = useEngineStore((s) => s.sttStreams);
  const assistant = useEngineStore(isAssistantConfigured);

  const streams = useMemo(
    () => Object.values(sttStreams).sort((a, b) => streamRank(a.source) - streamRank(b.source)),
    [sttStreams],
  );

  return (
    <footer className="status-bar">
      <div className="status-bar-left">
        <span className={`status-dot ${connected ? "status-connected" : "status-disconnected"}`} />
        <span className="status-label">{connected ? "Connected" : "Disconnected"}</span>
        {running && (
          <>
            <span className="status-separator">|</span>
            <span className="status-recording">
              <PulsingDot />
              Listening
            </span>
          </>
        )}
      </div>

      <div className="status-bar-center">
        {running && (
          <>
            <span className="status-info" title="Spoken language and translation target">
              {languageLabel(language)}
              {translationEnabled && ` → ${languageLabel(targetLang)}`}
            </span>
            {streams.map((stream) => (
              <span
                key={stream.source}
                className="status-provider"
                data-fallback={stream.fallbackFrom ? "true" : "false"}
                title={
                  stream.fallbackFrom
                    ? `Switched to Local from ${sttProviderLabel(stream.fallbackFrom)}`
                    : "Speech-to-text provider"
                }
              >
                {sourceLabels?.[stream.source as "mic" | "system"] ||
                  STREAM_LABELS[stream.source] ||
                  stream.source}
                : {sttProviderLabel(stream.provider)}
                {stream.fallbackFrom && <span className="status-fallback"> (fallback)</span>}
                {stream.continuityCount > 0 && (
                  <span className="status-continuity" title="Intervals whose transcript may be missing or repeated">
                    {" "}
                    · {stream.continuityCount} gap{stream.continuityCount !== 1 ? "s" : ""}
                  </span>
                )}
              </span>
            ))}
            {assistant && <span className="status-capability">Assist on</span>}
          </>
        )}
      </div>

      <div className="status-bar-right">
        <span className="status-count">
          {entryCount} segment{entryCount !== 1 ? "s" : ""}
        </span>
      </div>
    </footer>
  );
}

function PulsingDot() {
  return <span className="pulsing-dot" />;
}
