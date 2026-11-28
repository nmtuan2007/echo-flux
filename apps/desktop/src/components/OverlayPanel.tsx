import { emit } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/tauri";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { isTauriRuntime } from "../lib/engineBridge";
import { useEngineStore } from "../store/engineStore";
import { PRESET_TARGET_LANGUAGES } from "../store/providerCatalog";
import {
  assistKeyOf,
  deriveSttNotice,
  isAssistantConfigured,
  streamRank,
  sttProviderLabel,
} from "../store/transcriptSlice";

export const OverlayPanel: React.FC = () => {
  const {
    config,
    services,
    entries,
    partials,
    running,
    connected,
    isToggling,
    appError,
    clearAppError,
    sourceLabels,
    engineConversationId,
    suggestionLoading,
    audioLevels,
    sttStreams,
    shortcutsEnabled,
    overlayFontSize: fontSize,
    overlayHighOpacity: highOpacity,
    setOverlayPrefs,
    configureLiveTranslation,
    requestAssist,
  } = useEngineStore();
  const assistantEnabled = useEngineStore(isAssistantConfigured);
  const [isLocked, setIsLocked] = useState(false);
  const [dismissedAssistKey, setDismissedAssistKey] = useState<string | null>(null);
  const [copiedReplyIdx, setCopiedReplyIdx] = useState<number | null>(null);
  const [silenceHint, setSilenceHint] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  const hasLiveSignal =
    (config.audioSource !== "system" && (audioLevels?.mic ?? 0) >= 0.01) ||
    (config.audioSource !== "microphone" && (audioLevels?.system ?? 0) >= 0.01) ||
    Object.keys(partials).length > 0;

  useEffect(() => {
    if (!running || !connected || hasLiveSignal) {
      setSilenceHint(false);
      return;
    }
    const timer = setTimeout(() => setSilenceHint(true), 15000);
    return () => clearTimeout(timer);
  }, [running, connected, hasLiveSignal, entries.length]);

  useEffect(() => {
    document.documentElement.style.backgroundColor = "transparent";
    document.body.style.backgroundColor = "transparent";
    const root = document.getElementById("root");
    if (root) root.style.backgroundColor = "transparent";
  }, []);

  useEffect(() => {
    invoke("set_stealth_mode", { enabled: config.stealthMode }).catch(console.error);
  }, [config.stealthMode]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let active = true;

    void import("@tauri-apps/api/event")
      .then((mod) => {
        if (!active || typeof mod.listen !== "function") return;
        return mod.listen("overlay_unlock", async () => {
          await invoke("set_ignore_cursor_events", { ignore: false });
          setIsLocked(false);
          void emit("overlay_lock_changed", { locked: false }).catch(() => {});
        });
      })
      .then((fn) => {
        if (typeof fn === "function") {
          if (active) unlisten = fn;
          else fn();
        }
      })
      .catch(() => {});

    return () => {
      active = false;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (isNearBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, partials]);

  const handleCaptionsScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  };

  useEffect(() => {
    if (!shortcutsEnabled && isLocked) {
      void invoke("set_ignore_cursor_events", { ignore: false }).catch(() => {});
      setIsLocked(false);
      void emit("overlay_lock_changed", { locked: false }).catch(() => {});
    }
  }, [shortcutsEnabled, isLocked]);

  const toggleLock = async () => {
    if (!shortcutsEnabled) return;
    try {
      const newLocked = !isLocked;
      await invoke("set_ignore_cursor_events", { ignore: newLocked });
      setIsLocked(newLocked);
      void emit("overlay_lock_changed", { locked: newLocked }).catch(() => {});
    } catch (e) {
      console.error("Failed to set ignore cursor events", e);
    }
  };

  const handleBackToMain = async () => {
    void emit("overlay_lock_changed", { locked: false }).catch(() => {});
    await emit("overlay_closed").catch(() => {});
    await invoke("revert_to_main_window").catch(console.error);
  };

  const handleToggleLive = async () => {
    await emit("relay_toggle_pipeline").catch(console.error);
  };

  const handleToggleTranslation = async () => {
    await emit("relay_toggle_translation").catch(console.error);
    if (!isTauriRuntime()) {
      await configureLiveTranslation({ enabled: !translationEnabled });
    }
  };

  const cycleFontSize = () => {
    const next = fontSize === "sm" ? "md" : fontSize === "md" ? "lg" : "sm";
    setOverlayPrefs({ fontSize: next });
    void emit("relay_set_overlay_prefs", { fontSize: next }).catch(() => {});
  };

  const toggleHighOpacity = () => {
    const next = !highOpacity;
    setOverlayPrefs({ highOpacity: next });
    void emit("relay_set_overlay_prefs", { highOpacity: next }).catch(() => {});
  };

  const translationEnabled = services?.translation?.enabled ?? config.translationEnabled;
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";
  const hasCustomLang = !PRESET_TARGET_LANGUAGES.some((o) => o.code === targetLang);
  const recentEntries = entries.slice(-30);
  const partialStreams = useMemo(
    () => Object.keys(partials).sort((a, b) => streamRank(a) - streamRank(b)),
    [partials],
  );

  const labelFor = (source: string | null | undefined) => {
    if (source === "mic") return sourceLabels.mic || "Me";
    if (source === "system") return sourceLabels.system || "Others";
    return source || "";
  };

  const fallbackNotice = useMemo(
    () => deriveSttNotice(entries, sttStreams, running),
    [entries, sttStreams, running],
  );
  const hasFallback =
    fallbackNotice.fallbacks.length > 0 || fallbackNotice.translationFallback;
  const fallbackTitle = hasFallback
    ? [
        ...fallbackNotice.fallbacks.map(
          (f) => `${labelFor(f.source)}: switched to Local from ${sttProviderLabel(f.from)}`,
        ),
        ...(fallbackNotice.translationFallback
          ? ["Translation switched to Local fallback"]
          : []),
      ].join(" · ")
    : "";

  const assistTarget = useMemo(() => {
    const finals = entries.filter((e) => e.isFinal && e.text.trim());
    if (finals.length === 0) return null;
    return [...finals].reverse().find((e) => e.source !== "mic") || finals[finals.length - 1];
  }, [entries]);

  const activeAssistEntry = useMemo(() => {
    if (!assistantEnabled) return null;
    for (let i = entries.length - 1; i >= 0; i--) {
      const e = entries[i];
      const key = assistKeyOf(e);
      if (suggestionLoading[key] || e.assist) return e;
    }
    return null;
  }, [assistantEnabled, entries, suggestionLoading]);

  const activeAssistKey = activeAssistEntry ? assistKeyOf(activeAssistEntry) : null;
  const activeAssistLoading = activeAssistKey ? suggestionLoading[activeAssistKey] === true : false;
  const showAssistDrawer = Boolean(
    assistantEnabled &&
      activeAssistEntry &&
      activeAssistKey &&
      activeAssistKey !== dismissedAssistKey &&
      (activeAssistLoading || activeAssistEntry.assist),
  );

  const canRequestOverlayAssist = Boolean(
    assistantEnabled &&
      assistTarget &&
      connected &&
      (assistTarget.conversationId ?? engineConversationId) &&
      !suggestionLoading[assistKeyOf(assistTarget)],
  );

  const handleTriggerAssist = () => {
    if (!assistTarget) return;
    setDismissedAssistKey(null);
    requestAssist(assistTarget.id);
  };

  const handleCopyOverlayReply = async (text: string, idx: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedReplyIdx(idx);
      setTimeout(() => setCopiedReplyIdx(null), 1200);
    } catch {}
  };

  return (
    <div
      className={`overlay-wrapper overlay-font-${fontSize} ${isLocked ? "locked" : ""} ${highOpacity ? "overlay-opaque" : ""}`}
    >
      {!isLocked && (
        <div className="overlay-controls" data-tauri-drag-region>
          <div className="overlay-drag-handle" data-tauri-drag-region>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 9h14M5 15h14" />
            </svg>
            <span>Drag</span>
            {running && (
              <span className="overlay-live-dots" title="Live audio activity">
                <span
                  className={`overlay-live-dot ${(audioLevels?.mic ?? 0) >= 0.02 ? "active" : ""}`}
                  title="Microphone"
                />
                <span
                  className={`overlay-live-dot system ${(audioLevels?.system ?? 0) >= 0.02 ? "active" : ""}`}
                  title="System audio"
                />
              </span>
            )}
            {hasFallback && (
              <span className="overlay-fallback-badge" title={fallbackTitle}>
                Local Fallback
              </span>
            )}
          </div>

          <div className="overlay-btn-group">
            <button
              type="button"
              className={`overlay-btn ${running ? "overlay-btn-stop" : "overlay-btn-start"}`}
              onClick={() => void handleToggleLive()}
              disabled={!connected || isToggling}
              title={running ? "Stop Transcription (Ctrl+Shift+M)" : "Start Transcription (Ctrl+Shift+M)"}
            >
              {isToggling ? "..." : running ? "■ Stop" : "● Start"}
            </button>
            <button
              type="button"
              className="overlay-btn"
              onClick={cycleFontSize}
              title="Cycle subtitle text size"
            >
              {fontSize.toUpperCase()}
            </button>
            <button
              type="button"
              className="overlay-btn"
              onClick={toggleHighOpacity}
              title="Toggle background opacity"
            >
              {highOpacity ? "90%" : "60%"}
            </button>
            <button
              type="button"
              className={`overlay-btn ${translationEnabled ? "overlay-btn-primary" : ""}`}
              onClick={() => void handleToggleTranslation()}
              disabled={isToggling}
              title={
                translationEnabled ? "Disable live translation" : "Enable live translation"
              }
              aria-label="Toggle translation"
            >
              {translationEnabled ? "🌐 On" : "🌐 Off"}
            </button>
            {translationEnabled && (
              <select
                className="overlay-select"
                aria-label="Target language"
                value={targetLang}
                onChange={(e) => {
                  const lang = e.target.value;
                  void emit("relay_set_target_lang", { targetLang: lang }).catch(console.error);
                  if (!isTauriRuntime()) {
                    void configureLiveTranslation({ targetLang: lang });
                  }
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
            {assistantEnabled && (
              <button
                type="button"
                className="overlay-btn overlay-btn-assist"
                onClick={handleTriggerAssist}
                disabled={!canRequestOverlayAssist}
                title="Suggest reply for latest speaker (Ctrl+Shift+S)"
              >
                ✨ Assist
              </button>
            )}
            <button
              type="button"
              className="overlay-btn overlay-btn-primary"
              onClick={handleBackToMain}
            >
              ← Main
            </button>
            <button
              type="button"
              className="overlay-btn"
              onClick={toggleLock}
              disabled={!shortcutsEnabled}
              title={
                shortcutsEnabled
                  ? "Lock to click through to apps behind it. Unlock anytime with Ctrl+Shift+U"
                  : "Enable Global Keyboard Shortcuts in Settings to use click-through Lock"
              }
            >
              Lock
            </button>
          </div>
        </div>
      )}

      {isLocked && (
        <div className="overlay-unlock-hint">
          <span>Ctrl+Shift+U to Unlock</span>
        </div>
      )}

      {appError && !isLocked && (
        <div className="overlay-error-banner" role="alert">
          <span>⚠ {appError}</span>
          <button
            type="button"
            className="overlay-assist-close"
            onClick={clearAppError}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      <div className="overlay-captions" ref={scrollRef} onScroll={handleCaptionsScroll}>
        {running && silenceHint && !isLocked && (
          <div className="overlay-silence-hint" role="status">
            No audio signal detected yet — check Mic/System device or mute state.
          </div>
        )}
        {recentEntries.length === 0 && partialStreams.length === 0 && (
          <div className="overlay-empty">
            {running ? "Listening for speech..." : "Waiting to start session..."}
          </div>
        )}

        {recentEntries.map((entry) => (
          <div
            key={entry.id}
            className={`overlay-caption-line ${entry.source ? `overlay-source-${entry.source}` : ""}`}
          >
            {entry.source && (
              <span className="overlay-speaker-tag">{labelFor(entry.source)}</span>
            )}
            <span className="overlay-caption-text">{entry.text}</span>
            {translationEnabled && entry.translation && (
              <div className="overlay-caption-translation">{entry.translation}</div>
            )}
          </div>
        ))}

        {partialStreams.map((stream) => {
          const partial = partials[stream];
          return (
            <div
              key={`overlay-partial-${stream}`}
              className={`overlay-caption-line overlay-caption-partial ${
                partial.source ? `overlay-source-${partial.source}` : ""
              }`}
            >
              {partial.source && (
                <span className="overlay-speaker-tag">{labelFor(partial.source)}</span>
              )}
              <span className="overlay-caption-text">
                {partial.text}
                <span className="typing-cursor" />
              </span>
              {translationEnabled && partial.translation && (
                <div className="overlay-caption-translation">{partial.translation}</div>
              )}
            </div>
          );
        })}
      </div>

      {showAssistDrawer && activeAssistEntry && activeAssistKey && (
        <div className="overlay-assist-drawer" role="region" aria-label="Quick Assist">
          <div className="overlay-assist-header">
            <span className="overlay-assist-title">✨ Quick Assist</span>
            <button
              type="button"
              className="overlay-assist-close"
              onClick={() => setDismissedAssistKey(activeAssistKey)}
              title="Dismiss Assist"
            >
              ×
            </button>
          </div>
          {activeAssistLoading ? (
            <div className="overlay-assist-status">Thinking...</div>
          ) : activeAssistEntry.assist?.error ? (
            <div className="overlay-assist-error-row">
              <span className="overlay-assist-error">{activeAssistEntry.assist.error.message}</span>
              {activeAssistEntry.assist.error.retryable && (
                <button
                  type="button"
                  className="overlay-btn"
                  onClick={() => requestAssist(activeAssistEntry.id)}
                >
                  Retry
                </button>
              )}
            </div>
          ) : (
            <div className="overlay-assist-items">
              {activeAssistEntry.assist?.suggestedReplies.slice(0, 3).map((reply, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="overlay-assist-chip"
                  onClick={() => void handleCopyOverlayReply(reply.text, idx)}
                  title="Click to copy reply"
                >
                  <span className="overlay-assist-strategy">
                    {copiedReplyIdx === idx ? "✓ Copied" : reply.strategy || "Reply"}
                  </span>
                  <span className="overlay-assist-text">{reply.text}</span>
                </button>
              ))}
              {activeAssistEntry.assist?.clarifyingQuestions.slice(0, 2).map((q, idx) => (
                <button
                  key={`q-${idx}`}
                  type="button"
                  className="overlay-assist-chip overlay-assist-question"
                  onClick={() => void handleCopyOverlayReply(q.question, 100 + idx)}
                  title="Click to copy clarifying question"
                >
                  <span className="overlay-assist-strategy">
                    {copiedReplyIdx === 100 + idx ? "✓ Copied" : "Ask"}
                  </span>
                  <span className="overlay-assist-text">{q.question}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
