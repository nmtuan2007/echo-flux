import { useEngineStore, EngineConfig } from "../store/engineStore";

const MODEL_SIZES = ["tiny", "base", "small", "medium", "large"];
const DEVICES = ["auto", "cpu", "cuda"];
const TRANSLATION_BACKENDS = [
  { value: "online", label: "Online (Google Translate)" },
  { value: "marian", label: "Local (MarianMT)" },
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

export function SettingsPanel() {
  const { config, updateConfig, running } = useEngineStore();

  return (
    <div className="settings-panel">
      <h2 className="settings-title">Settings</h2>
      {running && <p className="settings-warning">Stop the pipeline before changing settings.</p>}

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
            onChange={(e) => updateConfig({ language: e.target.value })}
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
                  updateConfig({ translationBackend: e.target.value as EngineConfig["translationBackend"] })
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
                onChange={(e) => updateConfig({ targetLang: e.target.value })}
                disabled={running}
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
            
             <p className="settings-note" style={{fontSize: "12px", color: "#666", marginTop: "8px"}}>
                Note: "Online" uses internet. "Local" downloads models (~300MB per pair).
            </p>
          </>
        )}
      </section>

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
