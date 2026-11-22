import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import {
  MAX_LABEL_CHARS,
  MAX_PROFILES_PER_SESSION,
  MAX_SESSION_PROFILE_BYTES,
  MAX_TITLE_CHARS,
} from "../store/types";

export interface ConversationSetupData {
  title: string;
  sourceLabels: { mic: string; system: string };
  attachedProfileIds: string[];
}

export interface ConversationSetupPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onStart: (setup?: ConversationSetupData) => void;
}

export function ConversationSetupPanel({ isOpen, onClose, onStart }: ConversationSetupPanelProps) {
  const {
    profiles,
    loadProfiles,
    attachedProfileIds,
    setAttachedProfileIds,
    sourceLabels,
    setSourceLabels,
    pendingTitle,
    setPendingTitle,
    setActiveView,
  } = useEngineStore();

  const [title, setTitle] = useState(pendingTitle);
  const [micLabel, setMicLabel] = useState(sourceLabels.mic || "Me");
  const [systemLabel, setSystemLabel] = useState(sourceLabels.system || "Others");

  useEffect(() => {
    if (isOpen) {
      void loadProfiles();
      setTitle(pendingTitle);
      setMicLabel(sourceLabels.mic || "Me");
      setSystemLabel(sourceLabels.system || "Others");
    }
  }, [isOpen, loadProfiles, pendingTitle, sourceLabels]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Calculate attached profiles aggregate byte size
  const attachedProfiles = profiles.filter((p) => attachedProfileIds.includes(p.profileId));
  const aggregateBytes = attachedProfiles.reduce((acc, p) => acc + (p.contentBytes || 0), 0);
  const countAttached = attachedProfileIds.length;

  const toggleProfile = (profileId: string) => {
    if (attachedProfileIds.includes(profileId)) {
      setAttachedProfileIds(attachedProfileIds.filter((id) => id !== profileId));
    } else {
      const target = profiles.find((p) => p.profileId === profileId);
      const targetBytes = target?.contentBytes || 0;
      if (
        countAttached < MAX_PROFILES_PER_SESSION &&
        aggregateBytes + targetBytes <= MAX_SESSION_PROFILE_BYTES
      ) {
        setAttachedProfileIds([...attachedProfileIds, profileId]);
      }
    }
  };

  const applySetupDraft = (markDirtyForNext = false): ConversationSetupData => {
    const finalLabels = {
      mic: micLabel.trim().slice(0, MAX_LABEL_CHARS) || "Me",
      system: systemLabel.trim().slice(0, MAX_LABEL_CHARS) || "Others",
    };
    const finalTitle = title.trim().slice(0, MAX_TITLE_CHARS);
    setSourceLabels(finalLabels);
    setPendingTitle(finalTitle, markDirtyForNext);
    return {
      title: finalTitle,
      sourceLabels: finalLabels,
      attachedProfileIds,
    };
  };

  const handleSaveOnly = () => {
    applySetupDraft(true);
    onClose();
  };

  const handleStart = (e: React.FormEvent) => {
    e.preventDefault();
    onStart(applySetupDraft());
  };

  return (
    <div className="setup-dialog-overlay" role="dialog" aria-modal="true" aria-labelledby="setup-title">
      <div className="setup-dialog-panel">
        <div className="setup-dialog-header">
          <h2 id="setup-title" className="setup-title">Pre-Session Conversation Setup</h2>
          <button className="btn-icon setup-close-btn" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        <form onSubmit={handleStart} className="setup-form">
          {/* Conversation Title */}
          <div className="setup-form-group">
            <label className="setup-label">
              Session Title
              <span className="setup-char-counter">
                {title.length} / {MAX_TITLE_CHARS} chars
              </span>
            </label>
            <input
              type="text"
              className="setup-input"
              value={title}
              maxLength={MAX_TITLE_CHARS}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Q4 Strategy Review (auto-titled if blank)"
            />
          </div>

          {/* Speaker Stream Labels */}
          <div className="setup-form-group">
            <label className="setup-label">Speaker Stream Identification</label>
            <div className="setup-stream-labels-grid setup-labels-row">
              <div className="setup-label-col">
                <span className="setup-sublabel">Microphone Audio</span>
                <input
                  type="text"
                  className="setup-input"
                  value={micLabel}
                  maxLength={MAX_LABEL_CHARS}
                  onChange={(e) => setMicLabel(e.target.value)}
                  placeholder="Me"
                />
              </div>
              <div className="setup-label-col">
                <span className="setup-sublabel">System Loopback Audio</span>
                <input
                  type="text"
                  className="setup-input"
                  value={systemLabel}
                  maxLength={MAX_LABEL_CHARS}
                  onChange={(e) => setSystemLabel(e.target.value)}
                  placeholder="Others"
                />
              </div>
            </div>
            <p className="setup-note-inline">
              Labels match physical devices (Mic vs System audio); voice diarization is not performed.
            </p>
          </div>

          {/* Context Profiles Attachment */}
          <div className="setup-form-group">
            <div className="setup-profiles-header">
              <label className="setup-label">Attach Context Profiles</label>
              <span className="setup-limits-badge">
                Attached: {countAttached} / {MAX_PROFILES_PER_SESSION} (
                {Math.round((aggregateBytes / 1024) * 10) / 10} / 256 KB)
              </span>
            </div>

            {profiles.length === 0 ? (
              <div className="setup-profiles-empty">
                <span>No context profiles available yet. </span>
                <button
                  type="button"
                  className="btn-text"
                  onClick={() => {
                    onClose();
                    setActiveView("profiles");
                  }}
                >
                  Create a profile →
                </button>
              </div>
            ) : (
              <div className="setup-profiles-checklist setup-profiles-list">
                {profiles.map((p) => {
                  const isChecked = attachedProfileIds.includes(p.profileId);
                  const wouldExceedCount = !isChecked && countAttached >= MAX_PROFILES_PER_SESSION;
                  const wouldExceedBytes =
                    !isChecked && aggregateBytes + p.contentBytes > MAX_SESSION_PROFILE_BYTES;
                  const disabled = wouldExceedCount || wouldExceedBytes;

                  return (
                    <label
                      key={p.profileId}
                      className={`setup-profile-check-item setup-profile-item ${isChecked ? "selected" : ""} ${
                        disabled ? "disabled" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={disabled}
                        onChange={() => toggleProfile(p.profileId)}
                      />
                      <span className="setup-profile-name">{p.name}</span>
                      <span className="setup-profile-size">
                        {Math.round((p.contentBytes / 1024) * 10) / 10} KB
                      </span>
                    </label>
                  );
                })}
              </div>
            )}

            <p className="setup-note-inline">
              Attached profiles are snapshotted when the session starts so future edits never alter history.
            </p>
          </div>

          {/* Actions */}
          <div className="setup-dialog-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleSaveOnly}>
              Save Setup
            </button>
            <button type="submit" className="btn btn-primary btn-large">
              Save &amp; Start 🚀
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
