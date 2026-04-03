import { useEffect } from "react";
import { AudioDeviceInfo, EngineConfig, useEngineStore } from "../store/engineStore";
const DEVICES = ["auto", "cpu", "cuda"];

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "zh", label: "Chinese" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "de", label: "German" },
  { code: "fr", label: "French" },
  { code: "es", label: "Spanish" },
  { code: "pt", label: "Portuguese" },
  { code: "ru", label: "Russian" },
  { code: "vi", label: "Vietnamese" },
  { code: "ar", label: "Arabic" },
  { code: "th", label: "Thai" },
  { code: "it", label: "Italian" },
  { code: "nl", label: "Dutch" },
  { code: "pl", label: "Polish" },
];



const AUDIO_SOURCES: { value: EngineConfig["audioSource"]; label: string }[] = [
  { value: "microphone", label: "Microphone only" },
  { value: "system", label: "Speaker / System audio only" },
  { value: "both", label: "Both (Mic + Speaker)" },
];

function DeviceSelect({
  id,
  value,
  devices,
  placeholder,
  onChange,
  disabled,
}: {
  id: string;
  value: string;
  devices: AudioDeviceInfo[];
  placeholder: string;
  onChange: (v: string) => void;
  disabled: boolean;
}) {
  return (
    <select id={id} value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      <option value="">{placeholder}</option>
      {devices.map((d) => (
        <option key={d.id} value={d.id}>
          {d.name}
        </option>
      ))}
    </select>
  );
}

export function SettingsPanel() {
  const {
    config,
    updateConfig,
    running,
    activeTranslationBackend,
    connected,
    availableMics,
    availableSpeakers,
    requestDeviceList,
    setActiveView,
    modelsList,
    requestModelsList,
  } = useEngineStore();

  // Auto-fetch device and model lists when connected.
  useEffect(() => {
    if (connected) {
      requestDeviceList();
      requestModelsList();
    }
  }, [connected]); // eslint-disable-line react-hooks/exhaustive-deps

  const downloadedAsrModels = modelsList.asr.filter((m) => m.is_downloaded);
  const downloadedTranslationModels = modelsList.translation.filter((m) => m.is_downloaded);

  // Dynamically disable marian if no models are downloaded
  const translationBackends = [
    { value: "online", label: "Online (Google Translate) → Marian fallback" },
    { value: "marian", label: "Local only (MarianMT)", disabled: downloadedTranslationModels.length === 0 },
  ];

  const showMicSelect = config.audioSource === "microphone" || config.audioSource === "both";
  const showSpeakerSelect = config.audioSource === "system" || config.audioSource === "both";

  return (
    <div className="settings-panel">
      <h2 className="settings-title">Settings</h2>
      {running && <p className="settings-warning">Stop the pipeline before changing settings.</p>}

      {/* ── Audio Input ──────────────────────────────────────────────── */}
      <section className="settings-section">
        <h3 className="settings-section-title">Audio Input</h3>

        <div className="settings-field">
          <label htmlFor="audio-source">Source</label>
          <select
            id="audio-source"
            value={config.audioSource}
            onChange={(e) => {
              updateConfig({ audioSource: e.target.value as EngineConfig["audioSource"] });
            }}
            disabled={running}
          >
            {AUDIO_SOURCES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        {showMicSelect && (
          <div className="settings-field">
            <label htmlFor="mic-device">Microphone</label>
            <DeviceSelect
              id="mic-device"
              value={config.micDeviceId}
              devices={availableMics}
              placeholder="Default microphone"
              onChange={(v) => updateConfig({ micDeviceId: v })}
              disabled={running}
            />
          </div>
        )}

        {showSpeakerSelect && (
          <div className="settings-field">
            <label htmlFor="speaker-device">Speaker (Loopback)</label>
            <DeviceSelect
              id="speaker-device"
              value={config.speakerDeviceId}
              devices={availableSpeakers}
              placeholder="Default speaker loopback"
              onChange={(v) => updateConfig({ speakerDeviceId: v })}
              disabled={running}
            />
          </div>
        )}

        <div className="settings-field settings-field-row">
          <label>Device List</label>
          <button
            className="btn-text"
            onClick={requestDeviceList}
            disabled={running || !connected}
            title={!connected ? "Connect the engine first" : "Re-enumerate audio devices"}
          >
            ↻ Refresh
          </button>
        </div>

        {config.audioSource === "both" && (
          <p className="settings-note">
            Both mic and speaker audio are mixed at 50% gain each before being sent to Whisper.
            Make sure both devices are selected correctly to avoid clipping.
          </p>
        )}
      </section>

      {/* ── Speech Recognition ───────────────────────────────────────── */}
      <section className="settings-section">
        <h3 className="settings-section-title">Speech Recognition</h3>

        <div className="settings-field">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
            <label htmlFor="model-size" style={{ marginBottom: 0 }}>Transcription Model</label>
            <button className="btn-text" onClick={() => setActiveView("model_manager")} disabled={running}>
              Manage Models
            </button>
          </div>
          {downloadedAsrModels.length > 0 ? (
            <select
              id="model-size"
              value={config.modelSize}
              onChange={(e) => updateConfig({ modelSize: e.target.value })}
              disabled={running}
            >
              {downloadedAsrModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id} ({m.size_mb} MB)
                </option>
              ))}
              {!downloadedAsrModels.some((m) => m.id === config.modelSize) && (
                <option value={config.modelSize}>
                  {config.modelSize} (Pending Download)
                </option>
              )}
            </select>
          ) : (
            <div style={{ flex: 1, padding: "8px 12px", background: "var(--bg-secondary)", borderRadius: "6px", fontSize: "13px", color: "var(--text-secondary)", fontStyle: "italic" }}>
              No models downloaded. Go to Manage Models to download one, or start to auto-download the default model.
            </div>
          )}
        </div>

        <div className="settings-field">
          <label htmlFor="language">Input Language</label>
          <select
            id="language"
            value={config.language}
            onChange={(e) => {
              updateConfig({ language: e.target.value, sourceLang: e.target.value });
            }}
            disabled={running}
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-field">
          <label htmlFor="device">Device</label>
          <select
            id="device"
            value={config.device}
            onChange={(e) => updateConfig({ device: e.target.value })}
            disabled={running}
          >
            {DEVICES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-field settings-field-row">
          <label htmlFor="vad-enabled">Voice Activity Detection</label>
          <input
            id="vad-enabled"
            type="checkbox"
            checked={config.vadEnabled}
            onChange={(e) => updateConfig({ vadEnabled: e.target.checked })}
            disabled={running}
          />
        </div>
      </section>

      {/* ── Translation ──────────────────────────────────────────────── */}
      <section className="settings-section">
        <h3 className="settings-section-title">Translation</h3>

        <div className="settings-field settings-field-row">
          <label htmlFor="translation-enabled">Enable Translation</label>
          <input
            id="translation-enabled"
            type="checkbox"
            checked={config.translationEnabled}
            onChange={(e) => updateConfig({ translationEnabled: e.target.checked })}
            disabled={running}
          />
        </div>

        {config.translationEnabled && (
          <>
            <div className="settings-field">
              <label htmlFor="translation-backend">Translation Backend</label>
              <select
                id="translation-backend"
                value={config.translationBackend}
                onChange={(e) =>
                  updateConfig({
                    translationBackend: e.target.value as EngineConfig["translationBackend"],
                  })
                }
                disabled={running}
              >
                {translationBackends.map((b) => (
                  <option key={b.value} value={b.value} disabled={b.disabled}>
                    {b.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="settings-field">
              <label htmlFor="target-lang">Target Language</label>
              <select
                id="target-lang"
                value={config.targetLang}
                onChange={(e) => {
                  updateConfig({ targetLang: e.target.value, translationModel: "" });
                }}
                disabled={running}
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="settings-field">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                <label htmlFor="translation-model" style={{ marginBottom: 0 }}>
                  Translation Model (MarianMT)
                </label>
                <button className="btn-text" onClick={() => setActiveView("model_manager")} disabled={running}>
                  Manage Models
                </button>
              </div>
              <div className="settings-field-group">
                {downloadedTranslationModels.length > 0 ? (
                  <select
                    id="translation-model"
                    value={config.translationModel}
                    onChange={(e) => updateConfig({ translationModel: e.target.value })}
                    disabled={running}
                  >
                    {downloadedTranslationModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                    {!downloadedTranslationModels.some((m) => m.id === config.translationModel) && config.translationModel !== "" && (
                      <option value={config.translationModel}>
                        {config.translationModel} (Not downloaded)
                      </option>
                    )}
                    {config.translationModel === "" && (
                      <option value="">-- Select a local model --</option>
                    )}
                  </select>
                ) : (
                  <div style={{ flex: 1, padding: "8px 12px", background: "var(--bg-secondary)", borderRadius: "6px", fontSize: "13px", color: "var(--text-secondary)", fontStyle: "italic" }}>
                    No local models downloaded. Go to Manage Models to download one.
                  </div>
                )}
              </div>
            </div>

            {running && activeTranslationBackend && (
              <div className="settings-status">
                Active backend:{" "}
                <span
                  className={
                    activeTranslationBackend === "online" ? "status-online" : "status-marian"
                  }
                >
                  {activeTranslationBackend === "online" ? "Online (Google)" : "Local (Marian)"}
                </span>
              </div>
            )}

            <p className="settings-note">
              {config.translationBackend === "online"
                ? "Uses Google Translate with automatic fallback to local Marian model if the online service is unavailable."
                : "Uses local MarianMT model (~300MB per language pair, downloaded on first use)."}
            </p>
          </>
        )}
      </section>

      {/* ── Invisible/Pro UI ──────────────────────────────────────────────── */}
      <section className="settings-section">
         <h3 className="settings-section-title">✨ Pro UX Experience</h3>
         <div className="settings-field settings-field-row">
            <label htmlFor="stealth-mode">Hide from Screen Share (Stealth Mode)</label>
            <input
               id="stealth-mode"
               type="checkbox"
               checked={config.stealthMode}
               onChange={(e) => updateConfig({ stealthMode: e.target.checked })}
            />
         </div>
         <p className="settings-note">
            When enabled, the Overlay Window will be invisible to OBS, Zoom, and Teams screen captures.
         </p>
      </section>

      {/* ── AI Assistant ─────────────────────────────────────────────── */}
      <section className="settings-section settings-section-ai">
        <h3 className="settings-section-title">🤖 AI Assistant</h3>

        <div className="settings-field settings-field-row">
          <label htmlFor="llm-enabled">Enable AI Assistant</label>
          <input
            id="llm-enabled"
            type="checkbox"
            checked={config.llmEnabled}
            onChange={(e) => updateConfig({ llmEnabled: e.target.checked })}
            disabled={running}
          />
        </div>

        {config.llmEnabled && (
          <>
            <div className="settings-field">
              <label htmlFor="llm-provider-url">Provider URL</label>
              <input
                id="llm-provider-url"
                type="text"
                value={config.llmProviderUrl}
                onChange={(e) => updateConfig({ llmProviderUrl: e.target.value })}
                placeholder="https://openrouter.ai/api/v1"
                disabled={running}
              />
            </div>

            <div className="settings-field">
              <label htmlFor="llm-api-key">API Key</label>
              <input
                id="llm-api-key"
                type="password"
                value={config.llmApiKey}
                onChange={(e) => updateConfig({ llmApiKey: e.target.value })}
                placeholder="sk-or-... / sk-..."
                disabled={running}
                className="settings-api-key-input"
              />
            </div>

            <div className="settings-field">
              <label htmlFor="llm-model">Model Name</label>
              <input
                id="llm-model"
                type="text"
                value={config.llmModel}
                onChange={(e) => updateConfig({ llmModel: e.target.value })}
                placeholder="openai/gpt-4o-mini, claude-3.5-sonnet, llama-3..."
                disabled={running}
              />
            </div>

            <p className="settings-note">
              Compatible with <strong>OpenAI</strong>, <strong>OpenRouter</strong>, <strong>Ollama</strong>,{" "}
              <strong>LM Studio</strong>, and any OpenAI-compatible API.
              For local models, set Provider URL to <code>http://localhost:11434/v1</code>.
            </p>
          </>
        )}

        {!config.llmEnabled && (
          <p className="settings-note">
            Enable to get inline reply suggestions and meeting summarization powered by any LLM.
          </p>
        )}
      </section>

      {/* ── Hugging Face ─────────────────────────────────────────────── */}
      <section className="settings-section">
        <h3 className="settings-section-title">Hugging Face (Model Hub)</h3>

        <div className="settings-field">
          <label htmlFor="hf-token">Access Token</label>
          <input
            id="hf-token"
            type="password"
            value={config.hfToken}
            onChange={(e) => updateConfig({ hfToken: e.target.value })}
            placeholder="hf_..."
            disabled={running}
            className="settings-api-key-input"
          />
        </div>
        <p className="settings-note">
          Required for downloading restricted/gated models (e.g., Pyannote, Llama). 
          Get your token from <strong>huggingface.co/settings/tokens</strong>.
        </p>
      </section>

      {/* ── Connection ───────────────────────────────────────────────── */}
      <section className="settings-section">
        <h3 className="settings-section-title">Connection</h3>

        <div className="settings-field">
          <label htmlFor="engine-host">Host</label>
          <input
            id="engine-host"
            type="text"
            value={config.host}
            onChange={(e) => updateConfig({ host: e.target.value })}
            disabled={running}
          />
        </div>

        <div className="settings-field">
          <label htmlFor="engine-port">Port</label>
          <input
            id="engine-port"
            type="number"
            value={config.port}
            onChange={(e) => updateConfig({ port: parseInt(e.target.value, 10) || 8765 })}
            disabled={running}
          />
        </div>
      </section>
    </div>
  );
}
