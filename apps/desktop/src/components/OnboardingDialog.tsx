import { useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { CloudConsent } from "./CloudConsent";
import { LocalPackStatus } from "./LocalPackStatus";

export function OnboardingDialog() {
  const { onboardingCompleted, setOnboardingCompleted } = useEngineStore();
  const [step, setStep] = useState<number>(1);
  const [sttReady, setSttReady] = useState<boolean>(false);
  const [transReady, setTransReady] = useState<boolean>(false);

  if (onboardingCompleted) return null;

  const handleNext = () => setStep((s) => Math.min(s + 1, 4));
  const handleBack = () => setStep((s) => Math.max(s - 1, 1));

  const handleComplete = () => {
    setOnboardingCompleted(true);
  };

  const handleSkipCloud = () => {
    // Keep 100% local by default, advance to completion
    setStep(4);
  };

  return (
    <div className="onboarding-overlay" role="dialog" aria-modal="true" aria-labelledby="onboarding-step-title">
      <div className="onboarding-modal">
        {/* Progress Stepper */}
        <div className="onboarding-stepper">
          <div className={`onboarding-step-indicator ${step >= 1 ? "active" : ""}`}>1</div>
          <span className="onboarding-step-label">Welcome</span>
          <div className="onboarding-step-divider" />
          <div className={`onboarding-step-indicator ${step >= 2 ? "active" : ""}`}>2</div>
          <span className="onboarding-step-label">Local AI</span>
          <div className="onboarding-step-divider" />
          <div className={`onboarding-step-indicator ${step >= 3 ? "active" : ""}`}>3</div>
          <span className="onboarding-step-label">Cloud Consent</span>
          <div className="onboarding-step-divider" />
          <div className={`onboarding-step-indicator ${step >= 4 ? "active" : ""}`}>4</div>
          <span className="onboarding-step-label">Ready</span>
        </div>

        {/* Step 1: Welcome & Architecture */}
        {step === 1 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Welcome to EchoFlux
            </h2>
            <p className="onboarding-step-desc">
              Private, local-first real-time speech translation, transcript revisions, and conversation assistance.
            </p>

            <div className="onboarding-welcome-hero">
              <div className="onboarding-feature-item">
                <span className="onboarding-feature-icon">🔒</span>
                <div className="onboarding-feature-text">
                  <h4>Zero-Cloud Guarantee</h4>
                  <p>All transcription, translation, and profiles can run entirely on your local machine with zero data transmission.</p>
                </div>
              </div>
              <div className="onboarding-feature-item">
                <span className="onboarding-feature-icon">⚡</span>
                <div className="onboarding-feature-text">
                  <h4>Hardware-Optimized Models</h4>
                  <p>Powered by Faster-Whisper and Marian neural translation tailored to your CPU/GPU hardware capabilities.</p>
                </div>
              </div>
              <div className="onboarding-feature-item">
                <span className="onboarding-feature-icon">🛡️</span>
                <div className="onboarding-feature-text">
                  <h4>OS Credential Vault</h4>
                  <p>Third-party API keys are never stored in plain text and are protected inside your operating system keyring.</p>
                </div>
              </div>
            </div>

            <div className="onboarding-footer">
              <div />
              <button className="btn btn-primary" onClick={handleNext}>
                Get Started →
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Local AI Readiness */}
        {step === 2 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Local AI Model Readiness
            </h2>
            <p className="onboarding-step-desc">
              Check and download curated open-source model packs for offline speech recognition and translation.
            </p>

            <div className="onboarding-packs-container">
              <LocalPackStatus capability="stt" onStatusChange={setSttReady} />
              <LocalPackStatus capability="translation" onStatusChange={setTransReady} />
            </div>

            <div className="onboarding-footer">
              <button className="btn btn-secondary" onClick={handleBack}>
                ← Back
              </button>
              <button className="btn btn-primary" onClick={handleNext}>
                {sttReady && transReady ? "Continue →" : "Continue Anyway →"}
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Cloud Consent */}
        {step === 3 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Optional Cloud Services & Privacy
            </h2>
            <p className="onboarding-step-desc">
              EchoFlux can connect to external providers (OpenAI, Gemini, Deepgram, AssemblyAI) if you desire cloud accuracy.
            </p>

            <CloudConsent mode="onboarding" onConsentChange={() => handleNext()} />

            <div className="onboarding-footer">
              <button className="btn btn-secondary" onClick={handleBack}>
                ← Back
              </button>
              <button className="btn btn-secondary" onClick={handleSkipCloud}>
                Keep 100% Local (Skip Cloud)
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Ready */}
        {step === 4 && (
          <div className="onboarding-body onboarding-step-ready">
            <span className="ready-icon">🎉</span>
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              You're Ready to Flow!
            </h2>
            <p className="onboarding-step-desc">
              EchoFlux is configured and ready. You can modify model selections, hardware tiers, and context profiles anytime in Settings.
            </p>

            <div className="onboarding-tips-card">
              <h4>Quick Shortcuts</h4>
              <ul>
                <li><kbd>Ctrl+Shift+M</kbd>: Start or Stop live session</li>
                <li><kbd>Ctrl+Shift+O</kbd>: Toggle Picture-in-Picture floating overlay</li>
                <li><kbd>Ctrl+Shift+S</kbd>: Request AI Assist on the last speaker utterance</li>
              </ul>
            </div>

            <div className="onboarding-footer">
              <button className="btn btn-secondary" onClick={handleBack}>
                ← Back
              </button>
              <button className="btn btn-primary btn-large" onClick={handleComplete}>
                Launch EchoFlux 🚀
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
