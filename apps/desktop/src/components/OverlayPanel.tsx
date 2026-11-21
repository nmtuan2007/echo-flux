import { invoke } from "@tauri-apps/api/tauri";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { assistKeyOf, isAssistantConfigured, streamRank } from "../store/transcriptSlice";

type FontSize = "sm" | "md" | "lg";

export const OverlayPanel: React.FC = () => {
  const {
    config,
    services,
    entries,
    partials,
    running,
    connected,
    sourceLabels,
    engineConversationId,
    suggestionLoading,
    requestAssist,
  } = useEngineStore();
  const assistantEnabled = useEngineStore(isAssistantConfigured);
  const [isLocked, setIsLocked] = useState(false);
  const [fontSize, setFontSize] = useState<FontSize>("md");
  const [highOpacity, setHighOpacity] = useState(false);
  const [dismissedAssistKey, setDismissedAssistKey] = useState<string | null>(null);
  const [copiedReplyIdx, setCopiedReplyIdx] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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
    const SHORTCUT = "CommandOrControl+Shift+U";
    let registered = false;

    const setupUnlock = async () => {
      try {
        const { register, unregister, isRegistered } = await import("@tauri-apps/api/globalShortcut");
        if (await isRegistered(SHORTCUT)) {
          await unregister(SHORTCUT);
        }
        await register(SHORTCUT, async () => {
          await invoke("set_ignore_cursor_events", { ignore: false });
          setIsLocked(false);
        });
        registered = true;
      } catch (e) {
        console.error("Failed to bind unlock shortcut", e);
      }
    };

    setupUnlock();

    return () => {
      if (registered) {
        import("@tauri-apps/api/globalShortcut").then(({ unregister }) => {
          unregister(SHORTCUT).catch(() => {});
        });
      }
    };
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, partials]);

  const toggleLock = async () => {
    try {
      const newLocked = !isLocked;
      await invoke("set_ignore_cursor_events", { ignore: newLocked });
      setIsLocked(newLocked);
    } catch (e) {
      console.error("Failed to set ignore cursor events", e);
    }
  };

  const handleBackToMain = async () => {
    await invoke("revert_to_main_window").catch(console.error);
  };

  const cycleFontSize = () => {
    setFontSize((prev) => (prev === "sm" ? "md" : prev === "md" ? "lg" : "sm"));
  };

  const translationEnabled = services?.translation?.enabled ?? config.translationEnabled;
  const recentEntries = entries.slice(-8);
  const partialStreams = useMemo(
    () => Object.keys(partials).sort((a, b) => streamRank(a) - streamRank(b)),
    [partials],
  );

  const labelFor = (source: string | null | undefined) => {
    if (source === "mic") return sourceLabels.mic || "Me";
    if (source === "system") return sourceLabels.system || "Others";
    return source || "";
  };

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
          </div>

          <div className="overlay-btn-group">
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
              onClick={() => setHighOpacity((v) => !v)}
              title="Toggle background opacity"
            >
              {highOpacity ? "90%" : "60%"}
            </button>
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
              title="Lock to click through to apps behind it. Unlock anytime with Ctrl+Shift+U"
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

      <div className="overlay-captions" ref={scrollRef}>
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
            <div className="overlay-assist-error">{activeAssistEntry.assist.error.message}</div>
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
            </div>
          )}
        </div>
      )}
    </div>
  );
};
