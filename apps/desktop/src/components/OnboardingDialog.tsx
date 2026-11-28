import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { PRESET_INPUT_LANGUAGES, PRESET_TARGET_LANGUAGES } from "../store/providerCatalog";
import { LocalPackStatus } from "./LocalPackStatus";

export function OnboardingDialog() {
  const {
    onboardingCompleted,
    setOnboardingCompleted,
    connected,
    services,
    config,
    availableMics,
    availableSpeakers,
    audioProbeLoading,
    audioProbeResult,
    updateServiceConfig,
    updateConfig,
    setServiceOptionInput,
    requestDeviceList,
    probeAudioInput,
  } = useEngineStore();
  const [step, setStep] = useState<number>(1);
  const [sttReady, setSttReady] = useState<boolean>(false);
  const [transReady, setTransReady] = useState<boolean>(false);
  const translationEnabled = services.translation.enabled;
  const spokenLang =
    (services?.stt?.options?.language as string) || config.language || "auto";
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";

  useEffect(() => {
    if (step === 2 && connected) {
      requestDeviceList();
    }
  }, [step, connected, requestDeviceList]);

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
          <span className="onboarding-step-label">Audio &amp; Shortcuts</span>
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
              <div className="settings-field">
                <label htmlFor="onboarding-source-lang">Spoken Language</label>
                <select
                  id="onboarding-source-lang"
                  className="settings-select"
                  value={spokenLang}
                  onChange={(e) => {
                    const lang = e.target.value;
                    setServiceOptionInput("stt", "language", lang);
                    setServiceOptionInput("translation", "source_lang", lang);
                    updateConfig({ language: lang, sourceLang: lang });
                  }}
                >
                  {PRESET_INPUT_LANGUAGES.map((opt) => (
                    <option key={opt.code} value={opt.code}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <LocalPackStatus capability="stt" onStatusChange={setSttReady} />
              <label className="onboarding-toggle-row">
                <input
                  type="checkbox"
                  checked={translationEnabled}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    updateServiceConfig("translation", { enabled });
                    updateConfig({ translationEnabled: enabled });
                    if (!enabled) setTransReady(false);
                  }}
                />
                <span>Enable Live Translation (downloads ~300 MB offline pack)</span>
              </label>
              {translationEnabled && (
                <>
                  <div className="settings-field">
                    <label htmlFor="onboarding-target-lang">Translate to</label>
                    <select
                      id="onboarding-target-lang"
                      className="settings-select"
                      value={targetLang}
                      onChange={(e) => {
                        const lang = e.target.value;
                        setServiceOptionInput("translation", "target_lang", lang);
                        updateConfig({ targetLang: lang });
                      }}
                    >
                      {PRESET_TARGET_LANGUAGES.map((opt) => (
                        <option key={opt.code} value={opt.code}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <LocalPackStatus capability="translation" onStatusChange={setTransReady} />
                </>
              )}
            </div>

            <div className="onboarding-footer">
              <button type="button" className="btn btn-secondary" onClick={handleComplete}>
                Skip Setup
              </button>
              <button type="button" className="btn btn-primary" onClick={() => setStep(2)}>
                {sttReady && (!translationEnabled || transReady) ? "Next →" : "Continue →"}
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="onboarding-body">
            <h2 id="onboarding-step-title" className="onboarding-step-title">
              Audio Setup &amp; Quick Shortcuts
            </h2>
            <p className="onboarding-step-desc">
              Choose your audio streams and test signal detection before your first meeting. Audio
              stays 100% on-device unless you enable a cloud provider in Settings.
            </p>

            <div className="onboarding-audio-card">
              <div className="settings-field">
                <label htmlFor="onboarding-audio-source">Audio Source</label>
                <select
                  id="onboarding-audio-source"
                  className="settings-select"
                  value={config.audioSource}
                  onChange={(e) =>
                    updateConfig({
                      audioSource: e.target.value as "microphone" | "system" | "both",
                    })
                  }
                >
                  <option value="both">Mic + System Audio (Recommended for meetings)</option>
                  <option value="microphone">Microphone Only</option>
                  <option value="system">System Audio Only</option>
                </select>
              </div>

              {config.audioSource !== "system" && (
                <div className="settings-field">
                  <label htmlFor="onboarding-mic-device">Microphone</label>
                  <select
                    id="onboarding-mic-device"
                    className="settings-select"
                    value={config.micDeviceId ?? ""}
                    onChange={(e) => updateConfig({ micDeviceId: e.target.value || undefined })}
                    disabled={!connected}
                  >
                    <option value="">System Default Microphone</option>
                    {availableMics.map((dev) => (
                      <option key={dev.id} value={dev.id}>
                        {dev.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {config.audioSource !== "microphone" && (
                <div className="settings-field">
                  <label htmlFor="onboarding-speaker-device">System Audio (Loopback)</label>
                  <select
                    id="onboarding-speaker-device"
                    className="settings-select"
                    value={config.speakerDeviceId ?? ""}
                    onChange={(e) =>
                      updateConfig({ speakerDeviceId: e.target.value || undefined })
                    }
                    disabled={!connected}
                  >
                    <option value="">System Default Output</option>
                    {availableSpeakers.map((dev) => (
                      <option key={dev.id} value={dev.id}>
                        {dev.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="onboarding-audio-actions">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => void probeAudioInput()}
                  disabled={!connected || audioProbeLoading}
                >
                  {audioProbeLoading ? "Checking..." : "Check Audio"}
                </button>
                <button
                  type="button"
                  className="btn-text"
                  onClick={requestDeviceList}
                  disabled={!connected}
                >
                  ↻ Refresh Devices
                </button>
              </div>

              {audioProbeResult && (
                <div className="audio-probe-panel" role="status" aria-label="Audio signal check">
                  {Object.entries(audioProbeResult).map(([stream, info]) => {
                    const pct = Math.round((info.level || 0) * 100);
                    return (
                      <div key={stream} className="audio-probe-row">
                        <span className="audio-probe-label">
                          {stream === "mic" ? "Mic" : "System"}
                        </span>
                        <div className="audio-probe-meter">
                          <div className="audio-probe-fill" style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`audio-probe-status ${info.ok ? "" : "error"}`}>
                          {!info.ok
                            ? `Error (${info.error || "failed"})`
                            : pct > 0
                              ? `Signal OK (${pct}%)`
                              : "Silence (0%)"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

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
                <li>
                  <kbd>Ctrl+Shift+U</kbd>: Unlock click-through Overlay
                </li>
              </ul>
            </div>

            <div className="onboarding-footer">
              <button type="button" className="btn btn-secondary" onClick={() => setStep(1)}>
                ← Back
              </button>
              <button type="button" className="btn btn-primary btn-large" onClick={handleComplete}>
                Save &amp; Launch 🚀
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
