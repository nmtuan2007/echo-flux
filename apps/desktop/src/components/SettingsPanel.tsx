import { useEffect, useState } from "react";
import { AudioDeviceInfo, EngineConfig, useEngineStore } from "../store/engineStore";

const MODEL_SIZES = ["tiny", "base", "small", "medium", "large"];
const DEVICES = ["auto", "cpu", "cuda"];

const TRANSLATION_BACKENDS = [
  { value: "online", label: "Online (Google Translate) → Marian fallback" },
  { value: "marian", label: "Local only (MarianMT)" },
];

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

// Preset Marian models for common language pairs
const PRESET_MODELS: Record<string, { id: string; label: string }[]> = {
  "en-vi": [
    { id: "", label: "Default (opus-mt-en-vi)" },
    { id: "Helsinki-NLP/opus-mt-en-vi", label: "Helsinki opus-mt-en-vi" },
  ],
  "en-zh": [
    { id: "", label: "Default (opus-mt-en-zh)" },
    { id: "Helsinki-NLP/opus-mt-en-zh", label: "Helsinki opus-mt-en-zh" },
  ],
  "en-ja": [
    { id: "", label: "Default (opus-mt-en-jap)" },
    { id: "Helsinki-NLP/opus-mt-en-jap", label: "Helsinki opus-mt-en-jap" },
  ],
  "en-ko": [
    { id: "", label: "Default (opus-mt-tc-big-en-ko)" },
    { id: "Helsinki-NLP/opus-mt-tc-big-en-ko", label: "Helsinki opus-mt-tc-big-en-ko" },
  ],
  "en-de": [
    { id: "", label: "Default (opus-mt-en-de)" },
    { id: "Helsinki-NLP/opus-mt-en-de", label: "Helsinki opus-mt-en-de" },
  ],
  "en-fr": [
    { id: "", label: "Default (opus-mt-en-fr)" },
    { id: "Helsinki-NLP/opus-mt-en-fr", label: "Helsinki opus-mt-en-fr" },
  ],
  "en-es": [
    { id: "", label: "Default (opus-mt-en-es)" },
    { id: "Helsinki-NLP/opus-mt-en-es", label: "Helsinki opus-mt-en-es" },
  ],
  "en-ru": [
    { id: "", label: "Default (opus-mt-en-ru)" },
    { id: "Helsinki-NLP/opus-mt-en-ru", label: "Helsinki opus-mt-en-ru" },
  ],
};

function getPresetModels(sourceLang: string, targetLang: string) {
  const key = `${sourceLang}-${targetLang}`;
  return PRESET_MODELS[key] || [{ id: "", label: "Default (auto-detect)" }];
}

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
  } = useEngineStore();
  const [showCustomModel, setShowCustomModel] = useState(false);

  // Auto-fetch device list when the panel mounts and we're connected.
  useEffect(() => {
    if (connected) {
      requestDeviceList();
    }
  }, [connected]); // eslint-disable-line react-hooks/exhaustive-deps

  const presetModels = getPresetModels(config.sourceLang || config.language, config.targetLang);

  const isCustomModel =
    config.translationModel !== "" && !presetModels.some((m) => m.id === config.translationModel);

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
          <label htmlFor="model-size">Model Size</label>
          <select
            id="model-size"
            value={config.modelSize}
            onChange={(e) => updateConfig({ modelSize: e.target.value })}
            disabled={running}
          >
            {MODEL_SIZES.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
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
                {TRANSLATION_BACKENDS.map((b) => (
                  <option key={b.value} value={b.value}>
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
                  setShowCustomModel(false);
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
              <label htmlFor="translation-model">
                Marian Model {showCustomModel || isCustomModel ? "(Custom)" : "(Preset)"}
              </label>
              {!showCustomModel && !isCustomModel ? (
                <div className="settings-field-group">
                  <select
                    id="translation-model"
                    value={config.translationModel}
                    onChange={(e) => updateConfig({ translationModel: e.target.value })}
                    disabled={running}
                  >
                    {presetModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                  <button
                    className="btn-text"
                    onClick={() => setShowCustomModel(true)}
                    disabled={running}
                  >
                    Custom
                  </button>
                </div>
              ) : (
                <div className="settings-field-group">
                  <input
                    id="translation-model-custom"
                    type="text"
                    value={config.translationModel}
                    onChange={(e) => updateConfig({ translationModel: e.target.value })}
                    placeholder="HuggingFace model ID or local path"
                    disabled={running}
                  />
                  <button
                    className="btn-text"
                    onClick={() => {
                      updateConfig({ translationModel: "" });
                      setShowCustomModel(false);
                    }}
                    disabled={running}
                  >
                    Preset
                  </button>
                </div>
              )}
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
