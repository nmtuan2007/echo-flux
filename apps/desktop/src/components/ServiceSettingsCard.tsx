import { useEngineStore } from "../store/engineStore";
import {
  deriveTestPhase,
  describeCredentialError,
  describeTestError,
  fieldsFor,
  findProvider,
  modelRequired,
  providersFor,
  serviceIssues,
  type TestConnectionPhase,
} from "../store/settingsSlice";
import type { ManifestField, ServiceName } from "../store/types";
import { ProviderField, SecretProviderField } from "./ProviderField";

// One provider-aware service card. Presentational: every change is dispatched to the store,
// which owns validation invalidation, vault calls, and Test Connection.

const SERVICE_COPY: Record<ServiceName, { title: string; description: string }> = {
  stt: {
    title: "Speech-to-text",
    description: "Turns meeting audio into live transcript text.",
  },
  translation: {
    title: "Translation",
    description: "Translates each final transcript line into the target language.",
  },
  assistant: {
    title: "Assistant",
    description: "Reply suggestions and meeting summaries. Off until you enable it.",
  },
};

const LANGUAGE_HINT_KEYS = ["language", "source_lang"];

function testButtonLabel(phase: TestConnectionPhase): string {
  switch (phase) {
    case "loading":
      return "Testing...";
    case "retry":
      return "Retrying...";
    case "error":
      return "Retry";
    case "success":
    case "stale":
      return "Test again";
    default:
      return "Test Connection";
  }
}

export function ServiceSettingsCard({ service }: { service: ServiceName }) {
  const config = useEngineStore((s) => s.services[service]);
  const running = useEngineStore((s) => s.running);
  const testUi = useEngineStore((s) => s.connectionTests[service]);
  const credentialOps = useEngineStore((s) => s.credentialOps[service]);
  const drafts = useEngineStore((s) => s.optionDrafts[service]);
  const updateServiceConfig = useEngineStore((s) => s.updateServiceConfig);
  const selectServiceProvider = useEngineStore((s) => s.selectServiceProvider);
  const setServiceOptionInput = useEngineStore((s) => s.setServiceOptionInput);
  const saveServiceSecret = useEngineStore((s) => s.saveServiceSecret);
  const deleteServiceSecret = useEngineStore((s) => s.deleteServiceSecret);
  const testServiceConnection = useEngineStore((s) => s.testServiceConnection);
  const consent = useEngineStore((s) => s.consent);
  const hasConsentedToCloud = useEngineStore((s) => s.hasConsentedToCloud);
  const acceptCloudConsent = useEngineStore((s) => s.acceptCloudConsent);

  const id = `service-${service}`;
  const copy = SERVICE_COPY[service];
  const providers = providersFor(service);
  const provider = findProvider(service, config.providerId);
  const issues = serviceIssues(service, config, drafts);
  const phase = deriveTestPhase(config, testUi);
  const locked = running;
  const active = service === "stt" || config.enabled;
  const fields = provider ? fieldsFor(provider, service) : [];
  const mainFields = fields.filter((f) => !f.advanced);
  const advancedFields = fields.filter((f) => f.advanced);
  const needsModel = provider ? modelRequired(provider) : false;
  const hasHint = fields.some((f) => LANGUAGE_HINT_KEYS.includes(f.key));
  const blocked = Object.keys(issues).length > 0;
  const isCloudProvider = Boolean(provider && provider.protocolFamily !== "local");
  const serviceConsented = Boolean(
    consent?.[service === "assistant" ? "assistant" : service] || hasConsentedToCloud,
  );
  const needsCloudConsent = active && isCloudProvider && !serviceConsented;

  const renderField = (field: ManifestField) => {
    if (field.secret) {
      const op = credentialOps[field.key];
      return (
        <SecretProviderField
          key={`${config.providerId}-${field.key}`}
          idPrefix={id}
          field={field}
          configured={Boolean(config.credentialRefs[field.key])}
          pending={op?.pending ?? false}
          error={op?.errorCode ? describeCredentialError(op.errorCode) : null}
          disabled={locked}
          onSave={(secret) => saveServiceSecret(service, field.key, secret)}
          onDelete={() => void deleteServiceSecret(service, field.key)}
        />
      );
    }
    return (
      <ProviderField
        key={`${config.providerId}-${field.key}`}
        idPrefix={id}
        field={field}
        value={config.options[field.key]}
        draft={drafts[field.key]}
        error={issues[field.key] ?? null}
        disabled={locked}
        onChange={(raw) => setServiceOptionInput(service, field.key, raw)}
      />
    );
  };

  let testMessage: string;
  switch (phase) {
    case "loading":
      testMessage = "Testing the connection...";
      break;
    case "retry":
      testMessage = "Retrying the connection test...";
      break;
    case "success":
      testMessage = "Connection succeeded with these settings.";
      break;
    case "stale":
      testMessage = "Settings changed since the last test. Test again.";
      break;
    case "error":
      testMessage = describeTestError(config.validation.errorCode);
      break;
    default:
      testMessage = blocked ? "Complete the required fields to test." : "Not tested yet.";
  }

  const endpointRequired = Boolean(provider?.requiresEndpoint && !provider.recommendedEndpoint);
  const endpointHintId = `${id}-endpoint-hint`;
  const endpointErrorId = issues.endpoint ? `${id}-endpoint-error` : null;
  const modelErrorId = issues.model ? `${id}-model-error` : null;

  return (
    <section
      className={`service-card${locked ? " is-locked" : ""}`}
      aria-labelledby={`${id}-title`}
      data-service={service}
    >
      <div className="service-card-header">
        <h3 id={`${id}-title`} className="settings-section-title">
          {copy.title}
        </h3>
        {active && (
          <span className={`service-badge service-badge-${phase}`} data-phase={phase}>
            {phase === "retry" ? "retrying" : phase}
          </span>
        )}
      </div>
      <p className="settings-note service-card-description">{copy.description}</p>

      {(service === "assistant" || service === "translation") && (
        <div className="settings-field settings-field-row">
          <label htmlFor={`${id}-enabled`}>
            {service === "assistant" ? "Enable Assistant" : "Enable Translation"}
          </label>
          <input
            id={`${id}-enabled`}
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => updateServiceConfig(service, { enabled: e.target.checked })}
            disabled={locked}
          />
        </div>
      )}

      {active && (
        <>
          <div className="settings-field">
            <label htmlFor={`${id}-provider`}>Provider</label>
            <select
              id={`${id}-provider`}
              value={config.providerId}
              onChange={(e) => selectServiceProvider(service, e.target.value)}
              disabled={locked}
              aria-invalid={issues.provider ? true : undefined}
            >
              {!provider && <option value={config.providerId}>Unknown provider</option>}
              {providers.map((m) => (
                <option key={m.providerId} value={m.providerId}>
                  {m.displayName}
                </option>
              ))}
            </select>
            {issues.provider && <p className="field-error">{issues.provider}</p>}
          </div>

          {provider && (
            <>
              <div className={`settings-field${issues.endpoint ? " has-error" : ""}`}>
                <label htmlFor={`${id}-endpoint`}>
                  API URL
                  {endpointRequired && <span className="field-required"> (required)</span>}
                </label>
                <input
                  id={`${id}-endpoint`}
                  type="text"
                  value={config.endpoint}
                  placeholder={
                    provider.recommendedEndpoint ||
                    (provider.protocolFamily === "local"
                      ? "Not used by the local runtime"
                      : "https://")
                  }
                  onChange={(e) => updateServiceConfig(service, { endpoint: e.target.value })}
                  disabled={locked}
                  spellCheck={false}
                  aria-required={endpointRequired || undefined}
                  aria-invalid={issues.endpoint ? true : undefined}
                  aria-describedby={[endpointHintId, endpointErrorId].filter(Boolean).join(" ")}
                />
                <p id={endpointHintId} className="field-hint">
                  {provider.recommendedEndpoint
                    ? "Leave empty to use the recommended URL shown."
                    : provider.protocolFamily === "local"
                      ? "Optional. The local runtime runs on this device."
                      : "Enter the full API URL for your resource or server."}
                </p>
                {issues.endpoint && (
                  <p id={endpointErrorId ?? undefined} className="field-error">
                    {issues.endpoint}
                  </p>
                )}
              </div>

              {needsModel ? (
                <div className={`settings-field${issues.model ? " has-error" : ""}`}>
                  <label htmlFor={`${id}-model`}>
                    Model<span className="field-required"> (required)</span>
                  </label>
                  <input
                    id={`${id}-model`}
                    type="text"
                    value={config.modelOverride}
                    onChange={(e) =>
                      updateServiceConfig(service, { modelOverride: e.target.value })
                    }
                    disabled={locked}
                    spellCheck={false}
                    aria-required
                    aria-invalid={issues.model ? true : undefined}
                    aria-describedby={modelErrorId ?? undefined}
                  />
                  {issues.model && (
                    <p id={modelErrorId ?? undefined} className="field-error">
                      {issues.model}
                    </p>
                  )}
                </div>
              ) : (
                <p className="service-model">
                  {provider.modelSelectable ? (
                    <>
                      Recommended model: <code>{provider.recommendedModel}</code>
                      {config.modelOverride.trim() && (
                        <>
                          {" "}
                          (override: <code>{config.modelOverride.trim()}</code>)
                        </>
                      )}
                    </>
                  ) : provider.protocolFamily === "local" ? (
                    "Model: managed by the local model packs."
                  ) : (
                    "Model: this provider has no model setting."
                  )}
                </p>
              )}

              {service !== "assistant" && (
                <p className="settings-note service-language-note">
                  Source language: auto-detect.
                  {hasHint && " Language hints are under Advanced."}
                </p>
              )}

              {needsCloudConsent && (
                <div className="service-consent-banner" role="note">
                  <span>
                    Cloud permission is not granted for {copy.title}. Audio or text will be sent to{" "}
                    {provider.displayName} when active.
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={locked}
                    onClick={() => acceptCloudConsent({ [service]: true })}
                  >
                    Grant Cloud Consent
                  </button>
                </div>
              )}

              {mainFields.map(renderField)}

              {(advancedFields.length > 0 || (provider.modelSelectable && !needsModel)) && (
                <details className="service-advanced">
                  <summary>Advanced</summary>
                  <div className="service-advanced-body">
                    {provider.modelSelectable && !needsModel && (
                      <div className="settings-field">
                        <label htmlFor={`${id}-model-override`}>Model override</label>
                        <input
                          id={`${id}-model-override`}
                          type="text"
                          value={config.modelOverride}
                          placeholder={provider.recommendedModel}
                          onChange={(e) =>
                            updateServiceConfig(service, { modelOverride: e.target.value })
                          }
                          disabled={locked}
                          spellCheck={false}
                          aria-describedby={`${id}-model-override-hint`}
                        />
                        <p id={`${id}-model-override-hint`} className="field-hint">
                          Leave empty to use the recommended model.
                        </p>
                      </div>
                    )}
                    {advancedFields.map(renderField)}
                  </div>
                </details>
              )}

              <div className="service-test">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => void testServiceConnection(service)}
                  disabled={locked || testUi.inFlight || blocked}
                  aria-describedby={`${id}-test-status`}
                >
                  {testButtonLabel(phase)}
                </button>
                <p
                  id={`${id}-test-status`}
                  className={`service-test-status service-test-${phase}`}
                  role="status"
                  aria-live="polite"
                >
                  {testMessage}
                </p>
              </div>
            </>
          )}
        </>
      )}

      {service === "assistant" && !config.enabled && (
        <p className="settings-note">Assistant is off. Enable it to choose a provider.</p>
      )}
      {service === "translation" && !config.enabled && (
        <p className="settings-note">Translation is off. Enable it to translate live captions.</p>
      )}
    </section>
  );
}
