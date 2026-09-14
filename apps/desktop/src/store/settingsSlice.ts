import type { StateStorage } from "zustand/middleware";
import { errorCodeOf, testProviderConnection } from "../lib/engineBridge";
import {
  getCredentialStatus,
  LEGACY_CREDENTIALS,
  LEGACY_SECRET_FIELDS,
  migrateLegacySecret,
  scrubLegacyStorageSecrets,
  type LegacyMigrationOutcome,
  type LegacySecretField,
} from "../lib/secureStore";
import type { EngineSliceCreator } from "./engineStore";
import type {
  AppView,
  EngineConfig,
  ServiceConfig,
  ServiceName,
  ServiceOptionValue,
  ServicesConfig,
  Theme,
  ValidationMetadata,
} from "./types";

// Public settings, credential references/configured state, and provider validation state.
// Raw secrets reach the OS vault through lib/secureStore and are never persisted.

export const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "en",
  device: "auto",
  translationEnabled: true,
  translationBackend: "online",
  translationModel: "",
  sourceLang: "en",
  targetLang: "vi",
  vadEnabled: true,
  audioSource: "microphone",
  micDeviceId: "",
  speakerDeviceId: "",
  llmEnabled: false,
  llmProviderUrl: "https://openrouter.ai/api/v1",
  llmApiKey: "",
  llmModel: "openai/gpt-4o-mini",
  stealthMode: false,
  hfToken: "",
};

// Legacy raw secret fields. Never persisted. Until Settings moves to write-only secret
// inputs they can hold a typed draft in memory; the secure runtime commits that draft to the
// vault (bootstrap, Start, leaving Settings) and clears it.
export const LEGACY_SECRET_CONFIG_KEYS = ["llmApiKey", "hfToken"] as const;

type LegacySecretConfigKey = (typeof LEGACY_SECRET_CONFIG_KEYS)[number];
export type PersistedConfigKey = Exclude<keyof EngineConfig, LegacySecretConfigKey>;
export type PersistedConfig = Pick<EngineConfig, PersistedConfigKey>;

// A Record rather than an array so adding an EngineConfig field fails to compile until it
// is classified as persisted here or as a secret above.
const PERSISTED_CONFIG_FLAGS: Record<PersistedConfigKey, true> = {
  host: true,
  port: true,
  modelSize: true,
  language: true,
  device: true,
  translationEnabled: true,
  translationBackend: true,
  translationModel: true,
  sourceLang: true,
  targetLang: true,
  vadEnabled: true,
  audioSource: true,
  micDeviceId: true,
  speakerDeviceId: true,
  llmEnabled: true,
  llmProviderUrl: true,
  llmModel: true,
  stealthMode: true,
};

export const PERSISTED_CONFIG_KEYS = Object.keys(PERSISTED_CONFIG_FLAGS) as PersistedConfigKey[];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Builds a fresh object from the whitelist, so unknown or secret-shaped keys never pass.
export function sanitizeConfig(raw: unknown): Partial<PersistedConfig> {
  if (!isRecord(raw)) return {};
  const out: Record<string, unknown> = {};
  for (const key of PERSISTED_CONFIG_KEYS) {
    const value = raw[key];
    if (value !== undefined && typeof value === typeof DEFAULT_CONFIG[key]) {
      out[key] = value;
    }
  }
  return out as Partial<PersistedConfig>;
}

// Full EngineConfig shape with every legacy secret blanked, for anything that leaves the
// store (the overlay sync event, the public start config).
export function publicConfigSnapshot(config: EngineConfig): EngineConfig {
  const out = { ...config };
  for (const key of LEGACY_SECRET_CONFIG_KEYS) out[key] = "";
  return out;
}

export function sanitizeTheme(raw: unknown): Theme | undefined {
  return raw === "dark" || raw === "light" ? raw : undefined;
}

export function preferredTheme(): Theme {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

// Mirrors _SECRET_KEY_NAMES in engine/core/config.py (compared lowercased, "_"/"-" removed).
const SECRET_OPTION_KEY_NAMES = new Set([
  "apikey",
  "secret",
  "secretkey",
  "clientsecret",
  "token",
  "accesstoken",
  "authtoken",
  "password",
  "authorization",
  "bearer",
  "value",
]);

export function isSecretKeyName(key: string): boolean {
  return SECRET_OPTION_KEY_NAMES.has(key.toLowerCase().replace(/[_-]/g, ""));
}

export function unvalidated(): ValidationMetadata {
  return { status: "unvalidated", fingerprint: null, checkedAt: null, errorCode: null };
}

function defaultService(enabled: boolean, providerId: string): ServiceConfig {
  return {
    enabled,
    providerId,
    endpoint: "",
    modelOverride: "",
    options: {},
    credentialRefs: {},
    validation: unvalidated(),
  };
}

// Matches _DEFAULT_SERVICES in engine/core/config.py.
export function defaultServices(): ServicesConfig {
  return {
    stt: defaultService(true, "local"),
    translation: defaultService(true, "local"),
    assistant: defaultService(false, "openai"),
  };
}

export type ServiceConfigChanges = Partial<Omit<ServiceConfig, "validation">>;

// Same set as _FINGERPRINTED_CHANGES in the engine; `enabled` does not invalidate a test.
const FINGERPRINTED_FIELDS = [
  "providerId",
  "endpoint",
  "modelOverride",
  "options",
  "credentialRefs",
] as const;

function fingerprintedFieldsDiffer(a: ServiceConfig, b: ServiceConfig): boolean {
  return FINGERPRINTED_FIELDS.some((field) => JSON.stringify(a[field]) !== JSON.stringify(b[field]));
}

function withoutSecretOptions(
  options: Record<string, ServiceOptionValue>,
): Record<string, ServiceOptionValue> {
  const out: Record<string, ServiceOptionValue> = {};
  for (const [key, value] of Object.entries(options)) {
    if (!isSecretKeyName(key)) out[key] = value;
  }
  return out;
}

// Legacy flat fields that feed the assistant: editing any of them voids its last test.
const ASSISTANT_LEGACY_FIELDS = ["llmApiKey", "llmProviderUrl", "llmModel"] as const;

function invalidateService(services: ServicesConfig, service: ServiceName): ServicesConfig {
  return { ...services, [service]: { ...services[service], validation: unvalidated() } };
}

// Points services.assistant at the migrated legacy LLM key. A new version on every write is
// how a credential change reaches the validation fingerprint without the value itself.
function withLegacyAssistantRef(
  services: ServicesConfig,
  configured: boolean,
  bumpVersion: boolean,
): ServicesConfig {
  const { configId, slot } = LEGACY_CREDENTIALS.llmApiKey;
  const assistant = services.assistant;
  const current = assistant.credentialRefs[slot];
  let credentialRefs: ServiceConfig["credentialRefs"];
  if (configured) {
    if (current && !bumpVersion) return services;
    credentialRefs = {
      ...assistant.credentialRefs,
      [slot]: { configId, slot, version: current ? current.version + 1 : 1 },
    };
  } else {
    if (!current) return services;
    credentialRefs = { ...assistant.credentialRefs };
    delete credentialRefs[slot];
  }
  return {
    ...services,
    assistant: { ...assistant, credentialRefs, validation: unvalidated() },
  };
}

export type SecretStatus = Record<LegacySecretField, boolean>;

export interface SettingsSlice {
  config: EngineConfig;
  theme: Theme;
  activeView: AppView;
  // v2 public service config. In memory only until the engine becomes its source of truth.
  services: ServicesConfig;
  // Whether the OS vault holds each legacy secret. The value itself never enters the store.
  secretStatus: SecretStatus;

  setActiveView: (view: AppView) => void;
  setTheme: (theme: Theme) => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
  updateServiceConfig: (service: ServiceName, changes: ServiceConfigChanges) => void;
  setServiceValidation: (service: ServiceName, validation: ValidationMetadata) => void;
  // Moves any raw legacy secret held in memory into the vault. Defaults to localStorage for
  // the legacy blob cleanup.
  commitLegacySecrets: (storage?: StateStorage) => Promise<LegacyMigrationOutcome[]>;
  refreshCredentialStatus: () => Promise<void>;
  testServiceConnection: (service: ServiceName) => Promise<ValidationMetadata | null>;
}

export const createSettingsSlice: EngineSliceCreator<SettingsSlice> = (set, get) => {
  // Per store instance: a second caller joins the running migration instead of racing it.
  let commitInFlight: Promise<LegacyMigrationOutcome[]> | null = null;

  const runCommit = async (storage?: StateStorage): Promise<LegacyMigrationOutcome[]> => {
    const outcomes: LegacyMigrationOutcome[] = [];
    const migrated: LegacySecretField[] = [];
    for (const field of LEGACY_SECRET_FIELDS) {
      const raw = get().config[field];
      const outcome = await migrateLegacySecret(field, raw);
      outcomes.push(outcome);
      if (outcome.status !== "migrated") continue;
      migrated.push(field);
      set((state) => ({
        // A newer draft typed during the write is left for the next commit.
        config: state.config[field] === raw ? { ...state.config, [field]: "" } : state.config,
        secretStatus: { ...state.secretStatus, [field]: true },
        services:
          field === "llmApiKey"
            ? withLegacyAssistantRef(state.services, true, true)
            : state.services,
      }));
    }

    if (migrated.length > 0) {
      try {
        scrubLegacyStorageSecrets(storage ?? window.localStorage, migrated);
      } catch {
        // Storage unavailable: nothing left to scrub from it either.
      }
    }

    const failedCodes = outcomes
      .filter((outcome) => outcome.status === "failed")
      .map((outcome) => outcome.errorCode ?? "vault_failure");
    if (failedCodes.length > 0) {
      set({
        appError:
          "A saved secret could not be moved to the system vault and will be retried " +
          `(${failedCodes.join(", ")})`,
      });
    }
    return outcomes;
  };

  return {
    config: { ...DEFAULT_CONFIG },
    theme: preferredTheme(),
    activeView: "transcript",
    services: defaultServices(),
    secretStatus: { llmApiKey: false, hfToken: false },

    setActiveView: (view) => {
      const { activeView, transport, config } = get();
      set({ activeView: view });
      const leavingSettings = activeView === "settings" && view !== "settings";
      if (
        leavingSettings &&
        transport === "secure" &&
        LEGACY_SECRET_FIELDS.some((field) => config[field])
      ) {
        void get().commitLegacySecrets();
      }
    },

    setTheme: (theme) => {
      set({ theme });
    },

    updateConfig: (partial) => {
      set((state) => {
        const config = { ...state.config, ...partial };
        const assistantChanged = ASSISTANT_LEGACY_FIELDS.some(
          (key) => config[key] !== state.config[key],
        );
        return assistantChanged
          ? { config, services: invalidateService(state.services, "assistant") }
          : { config };
      });
    },

    updateServiceConfig: (service, changes) => {
      set((state) => {
        const current = state.services[service];
        const next: ServiceConfig = {
          ...current,
          ...changes,
          options: changes.options ? withoutSecretOptions(changes.options) : current.options,
          validation: current.validation,
        };
        if (fingerprintedFieldsDiffer(current, next)) next.validation = unvalidated();
        return { services: { ...state.services, [service]: next } };
      });
    },

    setServiceValidation: (service, validation) => {
      set((state) => ({
        services: { ...state.services, [service]: { ...state.services[service], validation } },
      }));
    },

    commitLegacySecrets: (storage) => {
      if (!commitInFlight) {
        commitInFlight = runCommit(storage).finally(() => {
          commitInFlight = null;
        });
      }
      return commitInFlight;
    },

    refreshCredentialStatus: async () => {
      for (const field of LEGACY_SECRET_FIELDS) {
        let configured: boolean;
        try {
          ({ configured } = await getCredentialStatus(LEGACY_CREDENTIALS[field]));
        } catch {
          continue; // Unknown is not "absent"; keep what the store already believes.
        }
        set((state) => ({
          secretStatus: { ...state.secretStatus, [field]: configured },
          services:
            field === "llmApiKey"
              ? withLegacyAssistantRef(state.services, configured, false)
              : state.services,
        }));
      }
    },

    testServiceConnection: async (service) => {
      const snapshot = get().services[service];
      let validation: ValidationMetadata;
      try {
        const result = await testProviderConnection(service, snapshot);
        validation = {
          status: result.status,
          fingerprint: result.fingerprint,
          checkedAt: result.testedAt,
          errorCode: result.errorCode,
        };
      } catch (error) {
        // A bridge failure says nothing about the provider, so it is not "invalid".
        validation = { ...unvalidated(), errorCode: errorCodeOf(error) };
      }
      // The config changed while the test ran; its result describes something else.
      if (fingerprintedFieldsDiffer(get().services[service], snapshot)) return null;
      get().setServiceValidation(service, validation);
      return validation;
    },
  };
};
