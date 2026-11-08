import { useEffect } from "react";
import { AudioDeviceInfo, EngineConfig, useEngineStore } from "../store/engineStore";
import { SERVICE_NAMES } from "../store/settingsSlice";
import { ServiceSettingsCard } from "./ServiceSettingsCard";
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
    connected,
    availableMics,
    availableSpeakers,
    requestDeviceList,
    refreshServiceCredentials,
  } = useEngineStore();

  // Auto-fetch device and model lists when connected.
  useEffect(() => {
    if (connected) {
      requestDeviceList();
    }
  }, [connected]); // eslint-disable-line react-hooks/exhaustive-deps

  // Which secret slots the vault holds for each service; values are never read back.
  useEffect(() => {
    for (const service of SERVICE_NAMES) void refreshServiceCredentials(service);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const showMicSelect = config.audioSource === "microphone" || config.audioSource === "both";
  const showSpeakerSelect = config.audioSource === "system" || config.audioSource === "both";

  return (
    <div className="settings-panel">
      <h2 className="settings-title">Settings</h2>
      {running && <p className="settings-warning">Stop the pipeline before changing settings.</p>}

      {/* ── Services ─────────────────────────────────────────────────── */}
      <div className="settings-group" role="group" aria-labelledby="settings-services-title">
        <h3 id="settings-services-title" className="settings-group-title">
          Services
        </h3>
        <p className="settings-group-note">
          Each service has its own provider. Source language is detected automatically.
        </p>
        <div className="service-cards">
          {SERVICE_NAMES.map((service) => (
            <ServiceSettingsCard key={service} service={service} />
          ))}
        </div>
      </div>

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
    </div>
  );
}
