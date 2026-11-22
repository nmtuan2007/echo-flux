import { useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { CloudConsent } from "./CloudConsent";
import { LocalPackStatus } from "./LocalPackStatus";

export function OnboardingDialog() {
  const { onboardingCompleted, hasConsentedToCloud, setOnboardingCompleted } = useEngineStore();
  const [step, setStep] = useState<number>(1);
  const [sttReady, setSttReady] = useState<boolean>(false);
  const [transReady, setTransReady] = useState<boolean>(false);

  if (onboardingCompleted) return null;

  const handleComplete = () => {
    setOnboardingCompleted(true);
  };

  return (
    <div
      className="onboarding-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-step-title"
    >
      <div className="onboarding-modal">
        <div className="onboarding-stepper">
          <div className={`onboarding-step-indicator ${step >= 1 ? "active" : ""}`}>1</div>
          <span className="onboarding-step-label">Local AI</span>
          <div className="onboarding-step-divider" />
          <div className={`onboarding-step-indicator ${step >= 2 ? "active" : ""}`}>2</div>
          <span className="onboarding-step-label">Privacy &amp; Shortcuts</span>
        </div>

        {step === 1 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Welcome to EchoFlux
            </h2>
            <p className="onboarding-step-desc">
              Local-first real-time transcription and translation. Verify or download offline models
              now, or skip to configure a cloud provider in Settings.
            </p>

            <div className="onboarding-packs-container">
              <LocalPackStatus capability="stt" onStatusChange={setSttReady} />
              <LocalPackStatus capability="translation" onStatusChange={setTransReady} />
            </div>

            <div className="onboarding-footer">
              <button type="button" className="btn btn-secondary" onClick={handleComplete}>
                Skip Setup
              </button>
              <button type="button" className="btn btn-primary" onClick={() => setStep(2)}>
                {sttReady && transReady ? "Next →" : "Continue →"}
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Privacy &amp; Quick Shortcuts
            </h2>
            <p className="onboarding-step-desc">
              By default, all audio stays on your device. Enable optional cloud providers below only
              if needed.
            </p>

            <CloudConsent mode="onboarding" />

            <div className="onboarding-tips-card">
              <h4>Quick Shortcuts</h4>
              <ul>
                <li>
                  <kbd>Ctrl+Shift+M</kbd>: Start or Stop live session
                </li>
                <li>
                  <kbd>Ctrl+Shift+O</kbd>: Toggle Floating PiP Overlay
                </li>
                <li>
                  <kbd>Ctrl+Shift+S</kbd>: Request AI Assist on latest remote utterance
                </li>
              </ul>
            </div>

            <div className="onboarding-footer">
              <button type="button" className="btn btn-secondary" onClick={() => setStep(1)}>
                ← Back
              </button>
              <button type="button" className="btn btn-primary btn-large" onClick={handleComplete}>
                {hasConsentedToCloud ? "Save & Launch 🚀" : "Keep 100% Local & Launch 🚀"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
