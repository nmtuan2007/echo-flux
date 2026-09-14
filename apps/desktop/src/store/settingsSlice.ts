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

// Public settings and provider validation state. No raw secret is ever written from here.

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

// Legacy raw secrets the current runtime still sends over the socket. They live in memory
// only; moving them to the OS vault is a later task.
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

function withoutSecretOptions(
  options: Record<string, ServiceOptionValue>,
): Record<string, ServiceOptionValue> {
  const out: Record<string, ServiceOptionValue> = {};
  for (const [key, value] of Object.entries(options)) {
    if (!isSecretKeyName(key)) out[key] = value;
  }
  return out;
}

export interface SettingsSlice {
  config: EngineConfig;
  theme: Theme;
  activeView: AppView;
  // v2 public service config. In memory only until the engine becomes its source of truth.
  services: ServicesConfig;

  setActiveView: (view: AppView) => void;
  setTheme: (theme: Theme) => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
  updateServiceConfig: (service: ServiceName, changes: ServiceConfigChanges) => void;
  setServiceValidation: (service: ServiceName, validation: ValidationMetadata) => void;
}

export const createSettingsSlice: EngineSliceCreator<SettingsSlice> = (set) => ({
  config: { ...DEFAULT_CONFIG },
  theme: preferredTheme(),
  activeView: "transcript",
  services: defaultServices(),

  setActiveView: (view) => {
    set({ activeView: view });
  },

  setTheme: (theme) => {
    set({ theme });
  },

  updateConfig: (partial) => {
    set((state) => ({
      config: { ...state.config, ...partial },
    }));
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
      const fingerprintChanged = FINGERPRINTED_FIELDS.some(
        (field) => JSON.stringify(current[field]) !== JSON.stringify(next[field]),
      );
      if (fingerprintChanged) next.validation = unvalidated();
      return { services: { ...state.services, [service]: next } };
    });
  },

  setServiceValidation: (service, validation) => {
    set((state) => ({
      services: { ...state.services, [service]: { ...state.services[service], validation } },
    }));
  },
});
