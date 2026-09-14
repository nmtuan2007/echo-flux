import { invoke } from "@tauri-apps/api/tauri";
import type { StateStorage } from "zustand/middleware";
import type { EngineConfig, ProviderCapability } from "../store/types";
import { BridgeFailure, errorCodeOf } from "./engineBridge";

// Write/delete/status-only wrapper over src-tauri/src/secure_store.rs. There is no read:
// a secret goes into the OS vault and the webview only ever learns `configured`.

export interface CredentialLocator {
  capability: ProviderCapability;
  configId: string;
  slot: string;
}

export interface CredentialStatus {
  configured: boolean;
}

function locatorArgs(locator: CredentialLocator): Record<string, string> {
  return { capability: locator.capability, configId: locator.configId, slot: locator.slot };
}

function normalizeStatus(raw: unknown, summary: string): CredentialStatus {
  const configured = (raw as { configured?: unknown } | null)?.configured;
  if (typeof configured !== "boolean") throw new BridgeFailure("invalid_response", summary);
  return { configured };
}

// The secret is an argument only; it is never part of a thrown error or a return value.
async function writeCommand(
  command: "credential_save" | "credential_replace",
  locator: CredentialLocator,
  secret: string,
): Promise<CredentialStatus> {
  const summary = "The credential could not be saved to the system vault";
  let raw: unknown;
  try {
    raw = await invoke<unknown>(command, { ...locatorArgs(locator), secret });
  } catch (error) {
    throw new BridgeFailure(errorCodeOf(error, "vault_failure"), summary);
  }
  return normalizeStatus(raw, summary);
}

export function saveCredential(locator: CredentialLocator, secret: string) {
  return writeCommand("credential_save", locator, secret);
}

export function replaceCredential(locator: CredentialLocator, secret: string) {
  return writeCommand("credential_replace", locator, secret);
}

async function statusCommand(
  command: "credential_delete" | "credential_status",
  locator: CredentialLocator,
  summary: string,
): Promise<CredentialStatus> {
  let raw: unknown;
  try {
    raw = await invoke<unknown>(command, locatorArgs(locator));
  } catch (error) {
    throw new BridgeFailure(errorCodeOf(error, "vault_failure"), summary);
  }
  return normalizeStatus(raw, summary);
}

export function deleteCredential(locator: CredentialLocator) {
  return statusCommand("credential_delete", locator, "The credential could not be removed");
}

export function getCredentialStatus(locator: CredentialLocator) {
  return statusCommand("credential_status", locator, "Credential status is unavailable");
}

// Save, or replace when the slot already holds a value.
export async function writeCredential(
  locator: CredentialLocator,
  secret: string,
): Promise<CredentialStatus> {
  try {
    return await saveCredential(locator, secret);
  } catch (error) {
    if (errorCodeOf(error) !== "already_configured") throw error;
    return replaceCredential(locator, secret);
  }
}

// ---------------------------------------------------------------------------
// Legacy browser-held secrets
// ---------------------------------------------------------------------------

export type LegacySecretField = Extract<keyof EngineConfig, "llmApiKey" | "hfToken">;

export const LEGACY_SECRET_FIELDS: readonly LegacySecretField[] = ["llmApiKey", "hfToken"];

// Fixed vault identities for the two legacy flat-config secrets. The Hugging Face token has
// no v2 service; it is filed under stt because the hub only serves ASR/translation models.
export const LEGACY_CREDENTIALS: Record<LegacySecretField, CredentialLocator> = {
  llmApiKey: { capability: "assistant", configId: "legacy-llm", slot: "api_key" },
  hfToken: { capability: "stt", configId: "legacy-hf-hub", slot: "hf_token" },
};

// Must equal LEGACY_PERSIST_STORAGE_NAME in store/engineStore.ts (asserted by tests); not
// imported to keep lib free of store runtime imports.
export const LEGACY_STORAGE_KEY = "echoflux-storage";

export type LegacyMigrationStatus = "absent" | "migrated" | "failed";

export interface LegacyMigrationOutcome {
  field: LegacySecretField;
  status: LegacyMigrationStatus;
  errorCode: string | null;
}

// Writes one raw value to the vault and confirms it with a status read. The caller removes
// the raw value only on "migrated"; on "failed" it must keep it so a later run can retry.
export async function migrateLegacySecret(
  field: LegacySecretField,
  raw: string,
): Promise<LegacyMigrationOutcome> {
  if (!raw) return { field, status: "absent", errorCode: null };
  const locator = LEGACY_CREDENTIALS[field];
  try {
    await writeCredential(locator, raw);
    const confirmed = await getCredentialStatus(locator);
    if (!confirmed.configured) {
      return { field, status: "failed", errorCode: "vault_unconfirmed" };
    }
    return { field, status: "migrated", errorCode: null };
  } catch (error) {
    return { field, status: "failed", errorCode: errorCodeOf(error, "vault_failure") };
  }
}

// Removes the given fields from the pre-slice blob, leaving everything else byte-for-byte
// meaningful. Idempotent; a blob that does not parse is left alone because nothing reads
// secrets out of it either.
export function scrubLegacyStorageSecrets(
  storage: StateStorage,
  fields: readonly LegacySecretField[],
): void {
  if (fields.length === 0) return;
  const raw = storage.getItem(LEGACY_STORAGE_KEY);
  if (typeof raw !== "string") return;
  let blob: { state?: { config?: Record<string, unknown> } };
  try {
    blob = JSON.parse(raw);
  } catch {
    return;
  }
  const config = blob?.state?.config;
  if (typeof config !== "object" || config === null) return;
  let changed = false;
  for (const field of fields) {
    if (field in config) {
      delete config[field];
      changed = true;
    }
  }
  if (changed) storage.setItem(LEGACY_STORAGE_KEY, JSON.stringify(blob));
}
