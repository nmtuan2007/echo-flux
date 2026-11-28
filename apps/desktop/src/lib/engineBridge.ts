import { invoke } from "@tauri-apps/api/tauri";
import type {
  AuthFrame,
  EngineMessage,
  ProviderCapability,
  ServerTransportFrame,
  ServiceConfig,
  StoredValidationStatus,
} from "../store/types";
import { TRANSPORT_PROTOCOL_VERSION } from "../store/types";

// Typed wrapper over the Tauri engine lifecycle, control forwarding, and Test Connection
// commands (src-tauri/src/engine_process.rs, control_bridge.rs). Nothing here ever receives
// the control token or a secret: Rust returns the port and event token only, and resolves
// credential references itself.

export interface EngineConnectionInfo {
  port: number;
  eventToken: string;
}

export type EngineProcessState = "stopped" | "running" | "exited";

export interface EngineProcessStatus {
  state: EngineProcessState;
  port: number | null;
}

export interface StartCredentialRef {
  capability: ProviderCapability;
  configId: string;
  slot: string;
}

export interface ControlForwardResult {
  messages: EngineMessage[];
}

export interface ProviderValidationResult {
  capability: ProviderCapability;
  providerId: string;
  status: StoredValidationStatus;
  latencyMs: number | null;
  testedAt: number;
  fingerprint: string | null;
  credentialVersion: number | null;
  errorCode: string | null;
}

const ERROR_CODE_RE = /^[a-z][a-z0-9_]{0,63}$/;

// Error text is built locally from the code alone, so whatever the rejection carried (a
// string from Tauri itself, or anything else) is never shown or logged.
export class BridgeFailure extends Error {
  constructor(
    readonly code: string,
    summary: string,
  ) {
    super(`${summary} (${code})`);
    this.name = "BridgeFailure";
  }
}

export function errorCodeOf(error: unknown, fallback = "bridge_failed"): string {
  if (error instanceof BridgeFailure) return error.code;
  if (typeof error === "object" && error !== null) {
    const code = (error as { code?: unknown }).code;
    if (typeof code === "string" && ERROR_CODE_RE.test(code)) return code;
  }
  return fallback;
}

async function call<T>(command: string, summary: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(command, args);
  } catch (error) {
    throw new BridgeFailure(errorCodeOf(error), summary);
  }
}

function invalidResponse(summary: string): BridgeFailure {
  return new BridgeFailure("invalid_response", summary);
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_IPC__" in window;
}

// ---------------------------------------------------------------------------
// Engine process lifecycle
// ---------------------------------------------------------------------------

function isPort(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0 && value <= 65535;
}

// Idempotent on the Rust side: returns the current run when one is already up.
export async function startOwnedEngine(): Promise<EngineConnectionInfo> {
  const summary = "The engine could not be started";
  const raw = await call<unknown>("engine_start", summary);
  const info = raw as Partial<EngineConnectionInfo> | null;
  if (!info || !isPort(info.port) || typeof info.eventToken !== "string" || !info.eventToken) {
    throw invalidResponse(summary);
  }
  // Rebuilt field by field so nothing else the command might return is kept.
  return { port: info.port, eventToken: info.eventToken };
}

export async function stopOwnedEngine(): Promise<EngineProcessStatus> {
  return normalizeStatus(await call<unknown>("engine_stop", "The engine could not be stopped"));
}

export async function getEngineStatus(): Promise<EngineProcessStatus> {
  return normalizeStatus(await call<unknown>("engine_status", "Engine status is unavailable"));
}

function normalizeStatus(raw: unknown): EngineProcessStatus {
  const status = raw as Partial<EngineProcessStatus> | null;
  const state = status?.state;
  if (state !== "stopped" && state !== "running" && state !== "exited") {
    throw invalidResponse("Engine status is unavailable");
  }
  return { state, port: isPort(status?.port) ? status.port : null };
}

// ---------------------------------------------------------------------------
// Control forwarding (legacy start/stop over the Rust-held control role)
// ---------------------------------------------------------------------------

function normalizeForward(raw: unknown, summary: string): ControlForwardResult {
  const messages = (raw as { messages?: unknown } | null)?.messages;
  if (!Array.isArray(messages)) throw invalidResponse(summary);
  // Phase B dispatcher replies carry structured fields without a legacy `type` key,
  // so keep any non-null plain object returned by the control bridge.
  return {
    messages: messages.filter(
      (m): m is EngineMessage => typeof m === "object" && m !== null && !Array.isArray(m),
    ),
  };
}

// `config` must already be public (dotted keys, no secret-named leaf); Rust rejects the whole
// request otherwise. `credentials` maps a secret config key to the vault entry Rust injects.
export async function forwardStart(
  config: Record<string, unknown>,
  credentials: Record<string, StartCredentialRef>,
): Promise<ControlForwardResult> {
  const summary = "The engine did not accept Start";
  const raw = await call<unknown>("engine_forward_start", summary, {
    request: { config, credentials },
  });
  return normalizeForward(raw, summary);
}

export async function forwardStop(): Promise<ControlForwardResult> {
  const summary = "The engine did not accept Stop";
  return normalizeForward(await call<unknown>("engine_forward_stop", summary), summary);
}

// ---------------------------------------------------------------------------
// Forwarded legacy commands (engine_forward_command)
// ---------------------------------------------------------------------------

// The non-lifecycle legacy commands Rust relays over its control role. start/stop keep their
// dedicated commands above.
export const FORWARDED_COMMAND_TYPES = [
  "list_devices",
  "request_suggestion",
  "request_assist",
  "configure_assistant",
  "request_summary",
  "summary_cancel",
  "conversation_create",
  "conversation_start",
  "conversation_list",
  "conversation_get",
  "conversation_rename",
  "conversation_delete",
  "conversation_export",
  "conversation_import",
  "message_update",
  "profile_create",
  "profile_list",
  "profile_get",
  "profile_update",
  "profile_delete",
  "profile_attach",
  "pack_status",
  "pack_install",
  "pack_cancel",
  "pack_retry",
  "pack_delete",
  "pack_check_readiness",
  "get_provider_manifests",
  "audio_probe",
  "test_connection",
  "assistant_ask",
  "assistant_ask_clear",
] as const;

export type ForwardedCommandType = (typeof FORWARDED_COMMAND_TYPES)[number];

export function isForwardedCommandType(type: unknown): type is ForwardedCommandType {
  return (FORWARDED_COMMAND_TYPES as readonly unknown[]).includes(type);
}

export interface ForwardedCommand extends Record<string, unknown> {
  type: ForwardedCommandType;
}

// `command` must already be free of raw secrets (Rust rejects hf_token and secret-named
// keys). `credentials` names vault entries Rust injects; it is omitted when empty.
export async function forwardCommand(
  command: ForwardedCommand,
  credentials: Record<string, StartCredentialRef> = {},
): Promise<ControlForwardResult> {
  const summary = "The engine did not accept the command";
  if (!isForwardedCommandType(command.type)) {
    throw new BridgeFailure("command_not_allowed", summary);
  }
  const args: Record<string, unknown> = { command };
  if (Object.keys(credentials).length > 0) args.credentials = credentials;
  return normalizeForward(await call<unknown>("engine_forward_command", summary, args), summary);
}

// Fixed toast text per forwarded-command error code; nothing from the rejection is echoed.
export function describeCommandError(code: string): string {
  switch (code) {
    case "command_not_allowed":
      return "The engine does not allow this action";
    case "invalid_command":
      return "The engine could not read this request";
    case "secret_in_command":
      return "The request was refused because it carried a secret";
    case "engine_not_running":
      return "The engine is not running";
    case "timeout":
      return "The engine did not answer in time";
    case "command_rejected":
      return "The engine rejected this request";
    case "vault_error":
      return "A saved credential could not be read from the system vault";
    default:
      return "The engine request failed";
  }
}

// ---------------------------------------------------------------------------
// Test Connection
// ---------------------------------------------------------------------------

// Public v2 fields only; validation metadata is an outcome, not an input.
export function testConnectionService(service: ServiceConfig): Record<string, unknown> {
  return {
    enabled: service.enabled,
    providerId: service.providerId,
    endpoint: service.endpoint,
    modelOverride: service.modelOverride,
    options: { ...service.options },
    credentialRefs: Object.fromEntries(
      Object.entries(service.credentialRefs).map(([slot, ref]) => [
        slot,
        { configId: ref.configId, slot: ref.slot, version: ref.version },
      ]),
    ),
  };
}

export async function testProviderConnection(
  capability: ProviderCapability,
  service: ServiceConfig,
): Promise<ProviderValidationResult> {
  const summary = "Test Connection did not complete";
  const raw = (await call<unknown>("provider_test_connection", summary, {
    request: { capability, service: testConnectionService(service) },
  })) as Partial<ProviderValidationResult> | null;

  const status = raw?.status;
  if (
    !raw ||
    raw.capability !== capability ||
    raw.providerId !== service.providerId ||
    (status !== "valid" && status !== "invalid" && status !== "unvalidated") ||
    typeof raw.testedAt !== "number"
  ) {
    throw invalidResponse(summary);
  }
  const errorCode =
    typeof raw.errorCode === "string" && ERROR_CODE_RE.test(raw.errorCode) ? raw.errorCode : null;
  return {
    capability,
    providerId: service.providerId,
    status,
    latencyMs: typeof raw.latencyMs === "number" ? raw.latencyMs : null,
    testedAt: raw.testedAt,
    fingerprint: typeof raw.fingerprint === "string" ? raw.fingerprint : null,
    credentialVersion: typeof raw.credentialVersion === "number" ? raw.credentialVersion : null,
    errorCode,
  };
}

// ---------------------------------------------------------------------------
// Authenticated event plane (engine/server/protocol.py)
// ---------------------------------------------------------------------------

export const CLOSE_BAD_HANDSHAKE = 4400;
export const CLOSE_AUTH_FAILED = 4401;
export const CLOSE_AUTH_TIMEOUT = 4408;

export function eventSocketUrl(info: EngineConnectionInfo): string {
  return `ws://127.0.0.1:${info.port}`;
}

export function eventAuthFrame(eventToken: string): string {
  const frame: AuthFrame = {
    op: "auth",
    role: "event",
    token: eventToken,
    protocol: TRANSPORT_PROTOCOL_VERSION,
  };
  return JSON.stringify(frame);
}

// Only the frame shapes an event client can legitimately receive; anything else is null.
export function parseEventPlaneFrame(data: unknown): ServerTransportFrame | null {
  if (typeof data !== "string") return null;
  let frame: unknown;
  try {
    frame = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof frame !== "object" || frame === null) return null;
  const f = frame as Record<string, unknown>;
  switch (f.op) {
    case "auth_ok":
      return f.role === "event" ? { op: "auth_ok", role: "event", protocol: TRANSPORT_PROTOCOL_VERSION } : null;
    case "event": {
      const event = f.event as EngineMessage | undefined;
      return typeof event === "object" && event !== null && typeof event.type === "string"
        ? { op: "event", event }
        : null;
    }
    case "error": {
      const code = typeof f.code === "string" && ERROR_CODE_RE.test(f.code) ? f.code : "error";
      return { op: "error", code, message: "" };
    }
    default:
      return null;
  }
}

export function describeTransportError(code: string): string {
  switch (code) {
    case "auth_failed":
    case "auth_required":
    case "auth_timeout":
    case "unsupported_protocol":
      return `The engine rejected the event connection (${code})`;
    case "forbidden":
      return "The engine event connection is receive-only (forbidden)";
    default:
      return `The engine event connection reported an error (${code})`;
  }
}
