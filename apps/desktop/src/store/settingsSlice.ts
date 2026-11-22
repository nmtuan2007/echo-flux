import type { StateStorage } from "zustand/middleware";
import { errorCodeOf, forwardCommand, testProviderConnection } from "../lib/engineBridge";
import {
  deleteCredential,
  getCredentialStatus,
  LEGACY_CREDENTIALS,
  LEGACY_SECRET_FIELDS,
  migrateLegacySecret,
  scrubLegacyStorageSecrets,
  writeCredential,
  type CredentialLocator,
  type LegacyMigrationOutcome,
  type LegacySecretField,
} from "../lib/secureStore";
import type { EngineSliceCreator } from "./engineStore";
import {
  credentialConfigId,
  defaultServices,
  deriveTestPhase,
  fieldsFor,
  findProvider,
  fingerprintInput,
  fingerprintedFieldsDiffer,
  initialServiceUiState,
  invalidateService,
  normalizeManifest,
  parseFieldInput,
  setRuntimeCatalog,
  unvalidated,
  withLegacyAssistantRef,
  withoutSecretOptions,
  type ConnectionTestUi,
  type CredentialOpState,
  type FieldInput,
} from "./providerCatalog";
import type {
  AppView,
  CloudConsentState,
  EngineConfig,
  HardwareInfo,
  ManifestField,
  PackStatusItem,
  ProviderManifest,
  ServiceConfig,
  ServiceName,
  Theme,
  ValidationMetadata,
} from "./types";
import { CLOUD_CONSENT_VERSION } from "./types";

export {
  credentialConfigId,
  CUSTOM_STT_PROFILE_FIELD,
  defaultServices,
  deriveTestPhase,
  describeCredentialError,
  describeTestError,
  fieldsFor,
  findProvider,
  initialServiceUiState,
  isSecretKeyName,
  isValidConfigId,
  modelRequired,
  parseFieldInput,
  PROVIDER_CATALOG,
  providersFor,
  sanitizeServiceConfig,
  sanitizeServices,
  SERVICE_NAMES,
  serviceIssues,
  unvalidated,
} from "./providerCatalog";
export type {
  ConnectionTestUi,
  CredentialOpState,
  FieldInput,
  ParsedFieldInput,
  TestConnectionPhase,
} from "./providerCatalog";

export const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "auto",
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

export const LEGACY_SECRET_CONFIG_KEYS = ["llmApiKey", "hfToken"] as const;

type LegacySecretConfigKey = (typeof LEGACY_SECRET_CONFIG_KEYS)[number];
export type PersistedConfigKey = Exclude<keyof EngineConfig, LegacySecretConfigKey>;
export type PersistedConfig = Pick<EngineConfig, PersistedConfigKey>;

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

const ASSISTANT_LEGACY_FIELDS = ["llmApiKey", "llmProviderUrl", "llmModel"] as const;

export type ServiceConfigChanges = Partial<Omit<ServiceConfig, "validation">>;
export type SecretStatus = Record<LegacySecretField, boolean>;

export interface PackDownloadProgress {
  percent: number;
  model: string;
  stage?: string;
}

const SHORTCUTS_ENABLED_KEY = "echoflux:shortcuts_enabled";

function loadShortcutsEnabled(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return window.localStorage?.getItem(SHORTCUTS_ENABLED_KEY) !== "false";
  } catch {
    return true;
  }
}

const CONSENT_STATE_KEY = "echoflux:cloud_consent_state";
const LEGACY_CONSENT_KEY = "echoflux:cloud_consent";

function loadInitialConsent(): CloudConsentState {
  const legacyConsent =
    typeof window !== "undefined" && window.localStorage?.getItem(LEGACY_CONSENT_KEY) === "true";
  const base: CloudConsentState = {
    version: CLOUD_CONSENT_VERSION,
    accepted: legacyConsent,
    acceptedAt: legacyConsent ? Date.now() : null,
    revoked: false,
    revokedAt: null,
    stt: legacyConsent,
    translation: legacyConsent,
    assistant: legacyConsent,
    summary: legacyConsent,
  };
  if (typeof window === "undefined") return base;
  try {
    const raw = window.localStorage?.getItem(CONSENT_STATE_KEY);
    if (!raw) return base;
    const parsed = JSON.parse(raw) as Partial<CloudConsentState>;
    if (!parsed || typeof parsed !== "object") return base;
    return {
      version: typeof parsed.version === "number" ? parsed.version : CLOUD_CONSENT_VERSION,
      accepted: Boolean(parsed.accepted),
      acceptedAt: typeof parsed.acceptedAt === "number" ? parsed.acceptedAt : null,
      revoked: Boolean(parsed.revoked),
      revokedAt: typeof parsed.revokedAt === "number" ? parsed.revokedAt : null,
      stt: Boolean(parsed.stt),
      translation: Boolean(parsed.translation),
      assistant: Boolean(parsed.assistant),
      summary: Boolean(parsed.summary),
    };
  } catch {
    return base;
  }
}

function persistConsentState(consent: CloudConsentState): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage?.setItem(CONSENT_STATE_KEY, JSON.stringify(consent));
    window.localStorage?.setItem(LEGACY_CONSENT_KEY, String(Boolean(consent.accepted)));
  } catch {}
}

export interface SettingsSlice {
  config: EngineConfig;
  theme: Theme;
  activeView: AppView;
  services: ReturnType<typeof defaultServices>;
  secretStatus: SecretStatus;

  setActiveView: (view: AppView) => void;
  setTheme: (theme: Theme) => void;
  updateConfig: (partial: Partial<EngineConfig>) => void;
  updateServiceConfig: (service: ServiceName, changes: ServiceConfigChanges) => void;
  setServiceValidation: (service: ServiceName, validation: ValidationMetadata) => void;
  commitLegacySecrets: (storage?: StateStorage) => Promise<LegacyMigrationOutcome[]>;
  refreshCredentialStatus: () => Promise<void>;
  testServiceConnection: (service: ServiceName) => Promise<ValidationMetadata | null>;

  connectionTests: Record<ServiceName, ConnectionTestUi>;
  credentialOps: Record<ServiceName, Record<string, CredentialOpState>>;
  optionDrafts: Record<ServiceName, Record<string, string>>;

  selectServiceProvider: (service: ServiceName, providerId: string) => void;
  setServiceOptionInput: (service: ServiceName, key: string, raw: FieldInput) => void;
  saveServiceSecret: (service: ServiceName, slot: string, secret: string) => Promise<boolean>;
  deleteServiceSecret: (service: ServiceName, slot: string) => Promise<boolean>;
  refreshServiceCredentials: (service: ServiceName) => Promise<void>;

  onboardingCompleted: boolean;
  hasConsentedToCloud: boolean;
  cloudConsentVersion: number;
  consent: CloudConsentState;
  packStatus: Record<string, PackStatusItem>;
  packDownloads: Record<string, PackDownloadProgress>;
  packInstallErrors: Record<string, string>;
  hardwareInfo: HardwareInfo | null;
  isPackChecking: boolean;
  shortcutsEnabled: boolean;
  settingsFocusService: ServiceName | null;

  setShortcutsEnabled: (enabled: boolean) => void;
  openSettingsFor: (service?: ServiceName) => void;
  clearSettingsFocus: () => void;
  setOnboardingCompleted: (completed: boolean) => void;
  setHasConsentedToCloud: (consented: boolean) => void;
  acceptCloudConsent: (consents?: Partial<CloudConsentState>) => void;
  revokeCloudConsent: () => void;
  manifestsLoaded: boolean;
  loadProviderManifests: () => Promise<void>;
  checkPackReadiness: (capability?: "stt" | "translation") => Promise<void>;
  installLocalPack: (
    packId: string,
    options?: { sourceLang?: string; targetLang?: string },
  ) => Promise<boolean>;
  cancelLocalPack: (packId: string) => Promise<boolean>;
  retryLocalPack: (
    packId: string,
    options?: { sourceLang?: string; targetLang?: string },
  ) => Promise<boolean>;
}

export const createSettingsSlice: EngineSliceCreator<SettingsSlice> = (set, get) => {
  let commitInFlight: Promise<LegacyMigrationOutcome[]> | null = null;

  const setTestUi = (service: ServiceName, patch: Partial<ConnectionTestUi>) => {
    set((state) => ({
      connectionTests: {
        ...state.connectionTests,
        [service]: { ...state.connectionTests[service], ...patch },
      },
    }));
  };

  const setCredentialOp = (service: ServiceName, slot: string, op: CredentialOpState | null) => {
    set((state) => {
      const ops = { ...state.credentialOps[service] };
      if (op) ops[slot] = op;
      else delete ops[slot];
      return { credentialOps: { ...state.credentialOps, [service]: ops } };
    });
  };

  const secretFieldOf = (service: ServiceName, slot: string): ManifestField | undefined => {
    const provider = findProvider(service, get().services[service].providerId);
    return provider ? fieldsFor(provider, service).find((f) => f.secret && f.key === slot) : undefined;
  };

  const setCredentialRef = (
    service: ServiceName,
    slot: string,
    ref: ServiceConfig["credentialRefs"][string] | null,
  ) => {
    const credentialRefs = { ...get().services[service].credentialRefs };
    if (ref) credentialRefs[slot] = ref;
    else delete credentialRefs[slot];
    get().updateServiceConfig(service, { credentialRefs });
  };

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
      } catch {}
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

  const initialOnboarding =
    typeof window !== "undefined" &&
    window.localStorage?.getItem("echoflux:onboarding_completed") === "true";
  const initialConsent = loadInitialConsent();

  return {
    config: { ...DEFAULT_CONFIG },
    theme: preferredTheme(),
    activeView: "transcript",
    services: defaultServices(),
    secretStatus: { llmApiKey: false, hfToken: false },
    onboardingCompleted: initialOnboarding,
    hasConsentedToCloud: initialConsent.accepted,
    cloudConsentVersion: CLOUD_CONSENT_VERSION,
    consent: initialConsent,
    packStatus: {},
    packDownloads: {},
    packInstallErrors: {},
    hardwareInfo: null,
    isPackChecking: false,
    shortcutsEnabled: loadShortcutsEnabled(),
    settingsFocusService: null,
    manifestsLoaded: false,

    setShortcutsEnabled: (enabled) => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem(SHORTCUTS_ENABLED_KEY, String(enabled));
        }
      } catch {}
      set({ shortcutsEnabled: enabled });
    },

    openSettingsFor: (service) => {
      set({ settingsFocusService: service ?? null });
      get().setActiveView("settings");
    },

    clearSettingsFocus: () => {
      set({ settingsFocusService: null });
    },

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
          continue;
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
      if (get().connectionTests[service].inFlight) return null;
      const snapshot = get().services[service];
      const retry = deriveTestPhase(snapshot, get().connectionTests[service]) === "error";
      setTestUi(service, { inFlight: true, retry });
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
        validation = { ...unvalidated(), errorCode: errorCodeOf(error) };
      }
      if (fingerprintedFieldsDiffer(get().services[service], snapshot)) {
        setTestUi(service, { inFlight: false, retry: false });
        return null;
      }
      get().setServiceValidation(service, validation);
      setTestUi(service, { inFlight: false, retry: false, testedInput: fingerprintInput(snapshot) });
      return validation;
    },

    ...initialServiceUiState(),

    selectServiceProvider: (service, providerId) => {
      if (get().running || !findProvider(service, providerId)) return;
      if (get().services[service].providerId === providerId) return;
      get().updateServiceConfig(service, {
        providerId,
        endpoint: "",
        modelOverride: "",
        options: {},
        credentialRefs: {},
      });
      set((state) => ({
        optionDrafts: { ...state.optionDrafts, [service]: {} },
        credentialOps: { ...state.credentialOps, [service]: {} },
      }));
      void get().refreshServiceCredentials(service);
    },

    setServiceOptionInput: (service, key, raw) => {
      if (get().running) return;
      const current = get().services[service];
      const provider = findProvider(service, current.providerId);
      const field = provider
        ? fieldsFor(provider, service).find((f) => f.key === key && !f.secret)
        : undefined;
      if (!field) return;
      const { value, error } = parseFieldInput(field, raw);
      const options = { ...current.options };
      if (value === undefined) delete options[key];
      else options[key] = value;
      get().updateServiceConfig(service, { options });
      set((state) => {
        const drafts = { ...state.optionDrafts[service] };
        if (error) drafts[key] = String(raw);
        else delete drafts[key];
        return { optionDrafts: { ...state.optionDrafts, [service]: drafts } };
      });
    },

    saveServiceSecret: async (service, slot, secret) => {
      const providerId = get().services[service].providerId;
      if (get().running || !secretFieldOf(service, slot)) return false;
      if (get().credentialOps[service][slot]?.pending) return false;
      if (!secret) {
        setCredentialOp(service, slot, { pending: false, errorCode: "empty_secret" });
        return false;
      }
      const configId = credentialConfigId(providerId);
      if (!configId) {
        setCredentialOp(service, slot, { pending: false, errorCode: "invalid_config_id" });
        return false;
      }
      const locator: CredentialLocator = { capability: service, configId, slot };
      setCredentialOp(service, slot, { pending: true, errorCode: null });
      try {
        const { configured } = await writeCredential(locator, secret);
        if (!configured) {
          setCredentialOp(service, slot, { pending: false, errorCode: "vault_unconfirmed" });
          return false;
        }
      } catch (error) {
        setCredentialOp(service, slot, {
          pending: false,
          errorCode: errorCodeOf(error, "vault_failure"),
        });
        return false;
      }
      setCredentialOp(service, slot, null);
      if (get().services[service].providerId !== providerId) return true;
      const previous = get().services[service].credentialRefs[slot];
      setCredentialRef(service, slot, { configId, slot, version: (previous?.version ?? 0) + 1 });
      return true;
    },

    deleteServiceSecret: async (service, slot) => {
      const snapshot = get().services[service];
      if (get().running || !secretFieldOf(service, slot)) return false;
      if (get().credentialOps[service][slot]?.pending) return false;
      const configId = snapshot.credentialRefs[slot]?.configId ?? credentialConfigId(snapshot.providerId);
      if (!configId) {
        setCredentialOp(service, slot, { pending: false, errorCode: "invalid_config_id" });
        return false;
      }
      setCredentialOp(service, slot, { pending: true, errorCode: null });
      try {
        const { configured } = await deleteCredential({ capability: service, configId, slot });
        if (configured) {
          setCredentialOp(service, slot, { pending: false, errorCode: "vault_unconfirmed" });
          return false;
        }
      } catch (error) {
        setCredentialOp(service, slot, {
          pending: false,
          errorCode: errorCodeOf(error, "vault_failure"),
        });
        return false;
      }
      setCredentialOp(service, slot, null);
      const legacy = LEGACY_CREDENTIALS.llmApiKey;
      if (service === legacy.capability && configId === legacy.configId && slot === legacy.slot) {
        set((state) => ({ secretStatus: { ...state.secretStatus, llmApiKey: false } }));
      }
      const current = get().services[service];
      if (current.providerId === snapshot.providerId && current.credentialRefs[slot]?.configId === configId) {
        setCredentialRef(service, slot, null);
      }
      return true;
    },

    refreshServiceCredentials: async (service) => {
      const providerId = get().services[service].providerId;
      const provider = findProvider(service, providerId);
      if (!provider) return;
      for (const field of fieldsFor(provider, service)) {
        if (!field.secret) continue;
        const ref = get().services[service].credentialRefs[field.key];
        const configId = ref?.configId ?? credentialConfigId(providerId);
        if (!configId) continue;
        let configured: boolean;
        try {
          ({ configured } = await getCredentialStatus({ capability: service, configId, slot: field.key }));
        } catch {
          continue;
        }
        const now = get().services[service];
        if (now.providerId !== providerId) return;
        const currentRef = now.credentialRefs[field.key];
        if (configured && !currentRef) {
          setCredentialRef(service, field.key, { configId, slot: field.key, version: 1 });
        } else if (!configured && currentRef?.configId === configId) {
          setCredentialRef(service, field.key, null);
        }
      }
    },

    setOnboardingCompleted: (completed: boolean) => {
      try {
        if (typeof window !== "undefined") {
          window.localStorage?.setItem("echoflux:onboarding_completed", String(completed));
        }
      } catch {}
      set({ onboardingCompleted: completed });
    },

    setHasConsentedToCloud: (consented: boolean) => {
      set((state) => {
        const next: CloudConsentState = {
          ...state.consent,
          accepted: consented,
          acceptedAt: consented ? Date.now() : state.consent.acceptedAt,
          revoked: !consented,
          revokedAt: !consented ? Date.now() : null,
        };
        persistConsentState(next);
        return { hasConsentedToCloud: consented, consent: next };
      });
    },

    acceptCloudConsent: (consents?: Partial<CloudConsentState>) => {
      set((state) => {
        const next: CloudConsentState = {
          ...state.consent,
          ...consents,
          version: CLOUD_CONSENT_VERSION,
          accepted: true,
          acceptedAt: Date.now(),
          revoked: false,
          revokedAt: null,
        };
        persistConsentState(next);
        return {
          hasConsentedToCloud: true,
          cloudConsentVersion: CLOUD_CONSENT_VERSION,
          consent: next,
        };
      });
    },

    revokeCloudConsent: () => {
      set((state) => {
        const next: CloudConsentState = {
          ...state.consent,
          accepted: false,
          revoked: true,
          revokedAt: Date.now(),
          stt: false,
          translation: false,
          assistant: false,
          summary: false,
        };
        persistConsentState(next);
        return { hasConsentedToCloud: false, consent: next };
      });
    },

    loadProviderManifests: async () => {
      try {
        const res = await forwardCommand({ type: "get_provider_manifests" });
        const msg = res.messages.find((m) => Array.isArray((m as any).manifests));
        const rawList = (msg as any)?.manifests;
        if (Array.isArray(rawList) && rawList.length > 0) {
          const parsed = rawList
            .map(normalizeManifest)
            .filter((m): m is ProviderManifest => m !== null);
          if (parsed.length > 0) {
            setRuntimeCatalog(parsed);
            set({ manifestsLoaded: true });
          }
        }
      } catch {}
    },

    checkPackReadiness: async (capability?: "stt" | "translation") => {
      set({ isPackChecking: true });
      try {
        const caps: ("stt" | "translation")[] = capability ? [capability] : ["stt", "translation"];
        const { services, config } = get();
        for (const cap of caps) {
          try {
            const cmd: Record<string, unknown> = { type: "pack_check_readiness", capability: cap };
            if (cap === "translation") {
              cmd.source_lang =
                (services?.translation?.options?.source_lang as string) || config.sourceLang || "en";
              cmd.target_lang =
                (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";
            }
            const res = await forwardCommand(cmd as any);
            const statusMsg = res.messages.find(
              (m) => m.type === "pack_status" || (m as any).ready !== undefined || (m as any).packs,
            );
            if (statusMsg) {
              const data = ((statusMsg as any).data as any) || statusMsg;
              const packId =
                (data.pack_id as string) ||
                (cap === "stt" ? "legacy-faster-whisper-small" : "legacy-marian-opus-mt");
              const ready = Boolean(data.ready);
              const sizeBytes =
                (data.size_bytes as number) || (cap === "stt" ? 500_000_000 : 300_000_000);
              const languageSupported =
                typeof data.language_supported === "boolean" ? data.language_supported : true;
              set((state) => ({
                packStatus: {
                  ...state.packStatus,
                  [cap]: {
                    packId,
                    kind: cap,
                    displayName:
                      cap === "stt" ? "Whisper (Local STT)" : "Marian MT (Local Translation)",
                    version: (data.version as string) || "1.0",
                    sizeBytes,
                    isReady: ready,
                    state: ready ? "ready" : "unknown",
                    downloadPercent: ready ? 100 : 0,
                    downloadedBytes: ready ? sizeBytes : 0,
                    totalBytes: sizeBytes,
                    issues: Array.isArray(data.issues) ? data.issues : [],
                    tier: (data.tier as string) || "standard",
                    languageSupported,
                  },
                },
              }));
            }
          } catch {}
        }
        try {
          const statusRes = await forwardCommand({ type: "pack_status" });
          const first = statusRes.messages[0] as any;
          if (first?.hardware) {
            set({
              hardwareInfo: {
                cpuCores: first.hardware.cpu_cores || 8,
                ramGb: first.hardware.ram_gb || 16,
                platform: first.hardware.platform || "windows",
                architecture: first.hardware.architecture || "x86_64",
                hasGpu: Boolean(first.hardware.has_gpu),
              },
            });
          }
        } catch {}
      } finally {
        set({ isPackChecking: false });
      }
    },

    installLocalPack: async (
      packId: string,
      options?: { sourceLang?: string; targetLang?: string },
    ) => {
      set((state) => {
        const nextErrors = { ...state.packInstallErrors };
        delete nextErrors[packId];
        return {
          packInstallErrors: nextErrors,
          packDownloads: {
            ...state.packDownloads,
            [packId]: { percent: 0, model: packId, stage: "downloading" },
          },
        };
      });
      const pollTimer = setInterval(async () => {
        try {
          const res = await forwardCommand({ type: "pack_status", pack_id: packId });
          const first = res.messages[0] as any;
          const prog = first?.download_progress;
          if (prog && typeof prog.percent === "number") {
            set((state) => ({
              packDownloads: {
                ...state.packDownloads,
                [packId]: {
                  percent: prog.percent,
                  model: typeof prog.model === "string" && prog.model ? prog.model : packId,
                  stage: typeof prog.stage === "string" ? prog.stage : "downloading",
                },
              },
            }));
          }
        } catch {}
      }, 600);
      try {
        await forwardCommand({
          type: "pack_install",
          pack_id: packId,
          ...(options?.sourceLang ? { source_lang: options.sourceLang } : {}),
          ...(options?.targetLang ? { target_lang: options.targetLang } : {}),
        });
        set((state) => ({
          packDownloads: {
            ...state.packDownloads,
            [packId]: { percent: 100, model: packId, stage: "ready" },
          },
        }));
        return true;
      } catch (err) {
        const code = errorCodeOf(err);
        const msg =
          code === "timeout"
            ? "Download timed out. Check your connection and retry."
            : code === "command_rejected"
              ? "Download cancelled or another installation is already running."
              : `Download failed (${code}).`;
        set((state) => ({
          packInstallErrors: { ...state.packInstallErrors, [packId]: msg },
        }));
        return false;
      } finally {
        clearInterval(pollTimer);
        set((state) => {
          const next = { ...state.packDownloads };
          delete next[packId];
          return { packDownloads: next };
        });
      }
    },

    cancelLocalPack: async (packId: string) => {
      try {
        await forwardCommand({ type: "pack_cancel", pack_id: packId });
        set((state) => {
          const next = { ...state.packDownloads };
          delete next[packId];
          return { packDownloads: next };
        });
        return true;
      } catch {
        return false;
      }
    },

    retryLocalPack: async (
      packId: string,
      options?: { sourceLang?: string; targetLang?: string },
    ) => {
      return get().installLocalPack(packId, options);
    },
  };
};
