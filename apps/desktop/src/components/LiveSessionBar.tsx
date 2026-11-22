import { useEffect, useState } from "react";
import { isTauriRuntime } from "../lib/engineBridge";
import { EngineConfig, useEngineStore } from "../store/engineStore";
import { findProvider, PRESET_INPUT_LANGUAGES, PRESET_TARGET_LANGUAGES } from "../store/providerCatalog";
import { isAssistantConfigured } from "../store/transcriptSlice";
import type { ExportEntry } from "../utils/export";
import { ExportMenu } from "./ExportMenu";

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
    isViewingHistory,
    audioLevels,
    pendingTitle,
    attachedProfileIds,
    sourceLabels,
    summaryText,
    packStatus,
    consent,
    checkPackReadiness,
    setActiveView,
    updateConfig,
    updateServiceConfig,
    setServiceOptionInput,
    openSetup,
    clearTranscript,
    requestSummary,
    openSummary,
    renameConversation,
    setPendingTitle,
  } = useEngineStore();
  const isAssistantEnabled = useEngineStore(isAssistantConfigured);

  const [elapsedSec, setElapsedSec] = useState(0);
  const [renaming, setRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");

  useEffect(() => {
    if (connected && isTauriRuntime() && !packStatus.stt) {
      void checkPackReadiness();
    }
  }, [connected, packStatus.stt, checkPackReadiness]);

  useEffect(() => {
    if (!running) {
      setElapsedSec(0);
      return;
    }
    setRenaming(false);
    const start = Date.now();
    const timer = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [running]);

  const spokenLang =
    (services?.stt?.options?.language as string) || config.language || "auto";
  const hasCustomSpokenLang = !PRESET_INPUT_LANGUAGES.some((o) => o.code === spokenLang);
  const translationEnabled = services?.translation?.enabled ?? config.translationEnabled;
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";
  const hasCustomLang = !PRESET_TARGET_LANGUAGES.some((o) => o.code === targetLang);
  const convId = activeConversationId || engineConversationId;
  const hasActiveConversation = Boolean(convId || entries.length > 0);
  const canSummarize = isAssistantEnabled && hasActiveConversation && connected;
  const hasSummary = Boolean(summaryText);
  const summaryLoading = useEngineStore((s) => s.summaryLoading);
  const activeStreams = Object.keys(partials);

  const sttSvc = services?.stt;
  const sttManifest = sttSvc ? findProvider("stt", sttSvc.providerId) : undefined;
  const sttIsCloud = Boolean(sttManifest && sttManifest.protocolFamily !== "local");
  const sttMissingCred =
    sttIsCloud && (!consent?.stt || Object.keys(sttSvc?.credentialRefs || {}).length === 0);
  const sttMissingLocal = !sttIsCloud && packStatus.stt && !packStatus.stt.isReady;

  const transSvc = services?.translation;
  const transManifest = transSvc ? findProvider("translation", transSvc.providerId) : undefined;
  const transIsCloud = Boolean(transManifest && transManifest.protocolFamily !== "local");
  const transMissingCred =
    translationEnabled &&
    transIsCloud &&
    (!consent?.translation || Object.keys(transSvc?.credentialRefs || {}).length === 0);
  const transMissingLocal =
    translationEnabled && !transIsCloud && packStatus.translation && !packStatus.translation.isReady;
  const transUnsupportedLocal =
    translationEnabled &&
    !transIsCloud &&
    packStatus.translation?.languageSupported === false;

  const setupReason = !connected
    ? "Engine offline"
    : sttMissingCred
      ? "STT key/consent needed"
      : sttMissingLocal
        ? "Offline STT model needed"
        : transUnsupportedLocal
          ? `Offline Marian unsupported (→ ${targetLang})`
          : transMissingCred
            ? "Translation key/consent needed"
            : transMissingLocal
              ? "Offline Translation model needed"
              : null;

  const buildExportEntries = (): ExportEntry[] =>
    entries.map((e) => ({
      text: e.text,
      translation: e.translation,
      timestamp: e.timestamp,
      startMs: e.startMs ?? null,
      endMs: e.endMs ?? null,
      speaker:
        e.source === "mic"
          ? sourceLabels.mic || "Me"
          : e.source === "system"
            ? sourceLabels.system || "Others"
            : e.source || undefined,
    }));

  const handleSaveRename = () => {
    const trimmed = renameDraft.trim();
    if (trimmed) {
      setPendingTitle(trimmed);
      if (convId) renameConversation(convId, trimmed);
    }
    setRenaming(false);
  };

  return (
    <div className="live-session-bar" role="region" aria-label="Session controls">
      <div className="live-session-left">
        {renaming && !running ? (
          <div className="history-edit-row">
            <input
              className="history-input"
              value={renameDraft}
              onChange={(e) => setRenameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSaveRename();
                if (e.key === "Escape") setRenaming(false);
              }}
              aria-label="Session title"
              autoFocus
            />
            <button type="button" className="btn-icon-sm success" onClick={handleSaveRename} title="Save title">
              ✓
            </button>
            <button type="button" className="btn-icon-sm" onClick={() => setRenaming(false)} title="Cancel">
              ✕
            </button>
          </div>
        ) : (
          <>
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
            {!running && isViewingHistory && (
              <span className="session-profile-badge" title="Viewing archived conversation">
                History
              </span>
            )}
            {!running && convId && (
              <button
                type="button"
                className="btn-icon-sm"
                onClick={() => {
                  setRenameDraft(pendingTitle || "");
                  setRenaming(true);
                }}
                title="Rename session"
                aria-label="Rename session"
              >
                ✎
              </button>
            )}
            {!running && isViewingHistory && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={clearTranscript}
                title="Return to a fresh session"
              >
                + New Session
              </button>
            )}
          </>
        )}

        {running && (
          <div className="session-live-telemetry" aria-label="Live stream activity">
            <span className="session-timer">{formatDuration(elapsedSec)}</span>
            {(config.audioSource === "microphone" || config.audioSource === "both") && (
              <span
                className={`stream-meter-pill ${
                  (audioLevels.mic ?? 0) >= 0.02 || activeStreams.includes("mic") ? "active" : ""
                }`}
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
                className={`stream-meter-pill ${
                  (audioLevels.system ?? 0) >= 0.02 || activeStreams.includes("system")
                    ? "active"
                    : ""
                }`}
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

          <select
            className="quick-select"
            aria-label="Spoken language"
            value={spokenLang}
            onChange={(e) => {
              const lang = e.target.value;
              setServiceOptionInput("stt", "language", lang);
              updateConfig({ language: lang });
            }}
            disabled={isToggling}
            title="Spoken language for Speech-to-Text"
          >
            {hasCustomSpokenLang && <option value={spokenLang}>{spokenLang}</option>}
            {PRESET_INPUT_LANGUAGES.map((opt) => (
              <option key={opt.code} value={opt.code}>
                🗣 {opt.label}
              </option>
            ))}
          </select>

          <label className="quick-toggle" title="Toggle live translation">
            <input
              type="checkbox"
              checked={translationEnabled}
              onChange={(e) => {
                const enabled = e.target.checked;
                updateServiceConfig("translation", { enabled });
                updateConfig({ translationEnabled: enabled });
              }}
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
                if (connected && isTauriRuntime()) void checkPackReadiness("translation");
              }}
              disabled={isToggling}
              title="Translation target language"
            >
              {hasCustomLang && <option value={targetLang}>{targetLang}</option>}
              {PRESET_TARGET_LANGUAGES.map((opt) => (
                <option key={opt.code} value={opt.code}>
                  → {opt.label}
                </option>
              ))}
            </select>
          )}
        </div>
      )}

      <div className="live-session-actions">
        {!running && (
          <button
            type="button"
            className={`readiness-pill ${setupReason ? "readiness-warn" : "readiness-ok"}`}
            onClick={() => {
              if (setupReason && connected) setActiveView("settings");
            }}
            title={
              setupReason
                ? `${setupReason} — click to open Settings`
                : "Audio & AI pipeline ready"
            }
          >
            {setupReason ? `⚠ ${setupReason}` : "✓ Ready"}
          </button>
        )}

        {isAssistantEnabled && (
          <button
            type="button"
            className={`btn btn-sm btn-summarize ${!canSummarize ? "btn-summarize-disabled" : ""}`}
            onClick={requestSummary}
            disabled={!canSummarize}
            title={!hasActiveConversation ? "Start a meeting to summarize" : "Summarize this meeting with AI"}
          >
            📝 Summarize
          </button>
        )}

        {isAssistantEnabled && hasActiveConversation && connected && (
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={openSummary}
            title="Ask AI a question about this meeting"
          >
            💬 Ask AI
          </button>
        )}

        {(hasSummary || summaryLoading) && (
          <button
            type="button"
            className="btn btn-sm btn-view-summary"
            onClick={openSummary}
            title={summaryLoading ? "View streaming meeting summary" : "View saved meeting summary"}
          >
            {summaryLoading ? "⏳ Summarizing..." : "📋 Summary"}
          </button>
        )}

        {entries.length > 0 && (
          <ExportMenu
            entries={buildExportEntries()}
            title={pendingTitle || "EchoFlux Session"}
            summaryMarkdown={summaryText || null}
          />
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
