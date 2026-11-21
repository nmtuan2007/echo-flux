import { useState } from "react";
import { useEngineStore } from "../store/engineStore";
import type { CloudConsentState } from "../store/types";

export interface CloudConsentProps {
  mode?: "onboarding" | "settings";
  onConsentChange?: (accepted: boolean) => void;
}

export function CloudConsent({ mode = "onboarding", onConsentChange }: CloudConsentProps) {
  const { consent, hasConsentedToCloud, acceptCloudConsent, revokeCloudConsent } = useEngineStore();

  const [toggles, setToggles] = useState({
    stt: consent.stt ?? hasConsentedToCloud,
    translation: consent.translation ?? hasConsentedToCloud,
    assistant: consent.assistant ?? hasConsentedToCloud,
    summary: consent.summary ?? hasConsentedToCloud,
  });

  const handleToggle = (key: keyof typeof toggles) => {
    setToggles((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleAcceptAll = () => {
    const updated: Partial<CloudConsentState> = {
      stt: toggles.stt,
      translation: toggles.translation,
      assistant: toggles.assistant,
      summary: toggles.summary,
    };
    acceptCloudConsent(updated);
    onConsentChange?.(true);
  };

  const handleRevoke = () => {
    revokeCloudConsent();
    setToggles({ stt: false, translation: false, assistant: false, summary: false });
    onConsentChange?.(false);
  };

  return (
    <div className="cloud-consent-card">
      <div className="cloud-consent-header">
        <span className="cloud-consent-icon">☁️</span>
        <div>
          <h3 className="cloud-consent-title">Cloud AI Capability Consent</h3>
          <p className="cloud-consent-subtitle">
            EchoFlux defaults to 100% on-device processing. Enable external cloud providers only if desired.
          </p>
        </div>
      </div>

      <div className="consent-disclosure-list cloud-disclosures-list">
        <div className="consent-opt-in-box">
          <label className="consent-disclosure-item consent-checkbox-label cloud-disclosure-item">
            <input
              type="checkbox"
              checked={toggles.stt}
              onChange={() => handleToggle("stt")}
              className="consent-checkbox cloud-checkbox"
            />
            <div className="consent-disclosure-text cloud-disclosure-text">
              <strong>Speech-to-Text (Audio Streaming)</strong>
              <p>Transmits raw real-time microphone and system audio streams to configured external cloud endpoints (e.g. OpenAI, Deepgram, Gemini).</p>
            </div>
          </label>
        </div>

        <div className="consent-opt-in-box">
          <label className="consent-disclosure-item consent-checkbox-label cloud-disclosure-item">
            <input
              type="checkbox"
              checked={toggles.translation}
              onChange={() => handleToggle("translation")}
              className="consent-checkbox cloud-checkbox"
            />
            <div className="consent-disclosure-text cloud-disclosure-text">
              <strong>Translation (Transcript Streaming)</strong>
              <p>Transmits finalized transcript text segments to external cloud translation services.</p>
            </div>
          </label>
        </div>

        <div className="consent-opt-in-box">
          <label className="consent-disclosure-item consent-checkbox-label cloud-disclosure-item">
            <input
              type="checkbox"
              checked={toggles.assistant}
              onChange={() => handleToggle("assistant")}
              className="consent-checkbox cloud-checkbox"
            />
            <div className="consent-disclosure-text cloud-disclosure-text">
              <strong>AI Assistant (Replies & Questions)</strong>
              <p>Transmits selected message context windows and attached context profile snapshots to external LLM providers.</p>
            </div>
          </label>
        </div>

        <div className="consent-opt-in-box">
          <label className="consent-disclosure-item consent-checkbox-label cloud-disclosure-item">
            <input
              type="checkbox"
              checked={toggles.summary}
              onChange={() => handleToggle("summary")}
              className="consent-checkbox cloud-checkbox"
            />
            <div className="consent-disclosure-text cloud-disclosure-text">
              <strong>Meeting Summary (Full Transcripts)</strong>
              <p>Transmits complete meeting transcripts to external LLM providers when meeting summarization is requested.</p>
            </div>
          </label>
        </div>
      </div>

      <div className="cloud-consent-disclaimer" role="note">
        <strong>Privacy Notice:</strong> Third-party cloud providers process data according to their individual privacy policies and terms of service. EchoFlux cannot guarantee provider-side data retention or privacy guarantees. Local transcripts and profile data remain on your device.
      </div>

      <div className="cloud-consent-actions">
        {mode === "settings" && hasConsentedToCloud ? (
          <div className="cloud-status-row">
            <span className="cloud-status-tag">Cloud Consent Active (v{consent.version || 1})</span>
            <button className="btn btn-danger btn-sm" onClick={handleRevoke}>
              Revoke Cloud Consent
            </button>
          </div>
        ) : (
          <button className="btn btn-primary" onClick={handleAcceptAll}>
            {mode === "onboarding" ? "Grant Selected Cloud Permissions" : "Enable Cloud Services"}
          </button>
        )}
      </div>
    </div>
  );
}
