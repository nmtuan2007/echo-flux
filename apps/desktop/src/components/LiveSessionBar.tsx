import { useEffect, useRef, useState } from "react";
import { EngineConfig, useEngineStore } from "../store/engineStore";
import { isAssistantConfigured } from "../store/transcriptSlice";
import {
  downloadFile,
  exportAsJSON,
  exportAsMarkdown,
  exportAsSRT,
  exportAsText,
  type ExportEntry,
} from "../utils/export";

const TARGET_LANG_OPTIONS = [
  { code: "vi", label: "Vietnamese (vi)" },
  { code: "en", label: "English (en)" },
  { code: "ja", label: "Japanese (ja)" },
  { code: "ko", label: "Korean (ko)" },
  { code: "zh", label: "Chinese (zh)" },
  { code: "fr", label: "French (fr)" },
  { code: "de", label: "German (de)" },
  { code: "es", label: "Spanish (es)" },
];

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function LiveSessionBar() {
  const {
    running,
    connected,
    isToggling,
    entries,
    partials,
    config,
    services,
    activeConversationId,
    engineConversationId,
    pendingTitle,
    attachedProfileIds,
    sourceLabels,
    summaryText,
    updateConfig,
    updateServiceConfig,
    setServiceOptionInput,
    openSetup,
    clearTranscript,
    requestSummary,
    openSummary,
  } = useEngineStore();
  const isAssistantEnabled = useEngineStore(isAssistantConfigured);

  const [elapsedSec, setElapsedSec] = useState(0);
  const [exportOpen, setExportOpen] = useState(false);
  const [copiedAll, setCopiedAll] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!running) {
      setElapsedSec(0);
      return;
    }
    const start = Date.now();
    const timer = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!exportOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [exportOpen]);

  const translationEnabled = services?.translation?.enabled ?? config.translationEnabled;
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";
  const hasCustomLang = !TARGET_LANG_OPTIONS.some((o) => o.code === targetLang);
  const hasActiveConversation = Boolean(
    activeConversationId || engineConversationId || entries.length > 0,
  );
  const canSummarize = isAssistantEnabled && hasActiveConversation && connected;
  const hasSummary = Boolean(summaryText);
  const activeStreams = Object.keys(partials);

  const buildExportEntries = (): ExportEntry[] =>
    entries.map((e) => ({
      text: e.text,
      translation: e.translation,
      timestamp: e.timestamp,
      speaker:
        e.source === "mic"
          ? sourceLabels.mic || "Me"
          : e.source === "system"
            ? sourceLabels.system || "Others"
            : e.source || undefined,
    }));

  const handleExport = (fmt: "txt" | "srt" | "md" | "json") => {
    setExportOpen(false);
    const items = buildExportEntries();
    const base =
      (pendingTitle || "echoflux-session")
        .toLowerCase()
        .replace(/[^a-z0-9-_]+/gi, "-")
        .replace(/^-|-$/g, "") || "echoflux-session";
    if (fmt === "txt") {
      downloadFile(exportAsText(items), `${base}.txt`);
    } else if (fmt === "srt") {
      downloadFile(exportAsSRT(items), `${base}.srt`);
    } else if (fmt === "md") {
      downloadFile(
        exportAsMarkdown(items, pendingTitle || "EchoFlux Session", summaryText || null),
        `${base}.md`,
        "text/markdown;charset=utf-8",
      );
    } else {
      downloadFile(exportAsJSON(items), `${base}.json`, "application/json;charset=utf-8");
    }
  };

  const handleCopyTranscript = async () => {
    setExportOpen(false);
    try {
      await navigator.clipboard.writeText(exportAsText(buildExportEntries()));
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1500);
    } catch {
      // Clipboard unavailable
    }
  };

  return (
    <div className="live-session-bar" role="region" aria-label="Session controls">
      <div className="live-session-left">
        <button
          type="button"
          className="session-context-pill"
          onClick={openSetup}
          disabled={running || !connected || isToggling}
          title={running ? "Session in progress" : "Configure session title, speaker labels, and profiles"}
        >
          <span className="session-context-title">
            {pendingTitle.trim() || "Quick Session"}
          </span>
          {attachedProfileIds.length > 0 && (
            <span className="session-profile-badge">
              {attachedProfileIds.length} profile{attachedProfileIds.length > 1 ? "s" : ""}
            </span>
          )}
          {!running && <span className="session-edit-hint">Edit</span>}
        </button>

        {running && (
          <div className="session-live-telemetry" aria-label="Live stream activity">
            <span className="session-timer">{formatDuration(elapsedSec)}</span>
            {(config.audioSource === "microphone" || config.audioSource === "both") && (
              <span
                className={`stream-meter-pill ${activeStreams.includes("mic") ? "active" : ""}`}
                title={`Microphone: ${sourceLabels.mic || "Me"}`}
              >
                <span className="stream-meter-bars">
                  <i />
                  <i />
                  <i />
                </span>
                {sourceLabels.mic || "Me"}
              </span>
            )}
            {(config.audioSource === "system" || config.audioSource === "both") && (
              <span
                className={`stream-meter-pill ${activeStreams.includes("system") ? "active" : ""}`}
                title={`System Audio: ${sourceLabels.system || "Others"}`}
              >
                <span className="stream-meter-bars">
                  <i />
                  <i />
                  <i />
                </span>
                {sourceLabels.system || "Others"}
              </span>
            )}
          </div>
        )}
      </div>

      {!running && (
        <div className="live-session-controls">
          <select
            className="quick-select"
            aria-label="Audio source"
            value={config.audioSource}
            onChange={(e) =>
              updateConfig({ audioSource: e.target.value as EngineConfig["audioSource"] })
            }
            disabled={isToggling}
            title="Audio capture source"
          >
            <option value="microphone">🎤 Mic only</option>
            <option value="system">🔊 System audio</option>
            <option value="both">🔄 Mic + System</option>
          </select>

          <label className="quick-toggle" title="Toggle live translation">
            <input
              type="checkbox"
              checked={translationEnabled}
              onChange={(e) => updateServiceConfig("translation", { enabled: e.target.checked })}
              disabled={isToggling}
            />
            <span>Translate</span>
          </label>

          {translationEnabled && (
            <select
              className="quick-select"
              aria-label="Target language"
              value={targetLang}
              onChange={(e) => {
                const lang = e.target.value;
                setServiceOptionInput("translation", "target_lang", lang);
                updateConfig({ targetLang: lang });
              }}
              disabled={isToggling}
              title="Translation target language"
            >
              {hasCustomLang && <option value={targetLang}>{targetLang}</option>}
              {TARGET_LANG_OPTIONS.map((opt) => (
                <option key={opt.code} value={opt.code}>
                  → {opt.label}
                </option>
              ))}
            </select>
          )}
        </div>
      )}

      <div className="live-session-actions">
        <button
          type="button"
          className={`btn btn-sm btn-summarize ${!canSummarize ? "btn-summarize-disabled" : ""}`}
          onClick={requestSummary}
          disabled={!canSummarize}
          title={
            !isAssistantEnabled
              ? "Enable AI Assistant in Settings first"
              : !hasActiveConversation
                ? "Start a meeting to summarize"
                : "Summarize this meeting with AI"
          }
        >
          📝 Summarize
        </button>

        {hasSummary && (
          <button
            type="button"
            className="btn btn-sm btn-view-summary"
            onClick={openSummary}
            title="View saved meeting summary"
          >
            📋 Summary
          </button>
        )}

        {entries.length > 0 && (
          <div className="export-dropdown" ref={exportMenuRef}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setExportOpen((v) => !v)}
              aria-expanded={exportOpen}
              title="Export or copy transcript"
            >
              {copiedAll ? "✓ Copied" : "Export ▾"}
            </button>
            {exportOpen && (
              <div className="export-dropdown-menu" role="menu">
                <button type="button" role="menuitem" onClick={handleCopyTranscript}>
                  Copy All Text
                </button>
                <button type="button" role="menuitem" onClick={() => handleExport("md")}>
                  Export Markdown (.md)
                </button>
                <button type="button" role="menuitem" onClick={() => handleExport("txt")}>
                  Export Plain Text (.txt)
                </button>
                <button type="button" role="menuitem" onClick={() => handleExport("srt")}>
                  Export Subtitles (.srt)
                </button>
                <button type="button" role="menuitem" onClick={() => handleExport("json")}>
                  Export JSON (.json)
                </button>
              </div>
            )}
          </div>
        )}

        {entries.length > 0 && !running && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={clearTranscript}
            disabled={isToggling}
            title="Clear current transcript"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
