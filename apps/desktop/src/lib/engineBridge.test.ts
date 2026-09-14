import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StateStorage } from "zustand/middleware";
import {
  createEngineStore,
  LEGACY_PERSIST_STORAGE_NAME,
  PERSIST_STORAGE_NAME,
  selectPersistedState,
  type EngineState,
} from "../store/engineStore";
import { stripCommandSecrets } from "../store/sessionSlice";
import { DEFAULT_CONFIG, publicConfigSnapshot } from "../store/settingsSlice";
import {
  BridgeFailure,
  forwardCommand,
  parseEventPlaneFrame,
  type ForwardedCommand,
  startOwnedEngine,
  testConnectionService,
} from "./engineBridge";
import { LEGACY_CREDENTIALS, LEGACY_STORAGE_KEY } from "./secureStore";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/tauri", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({ emit: vi.fn(async () => undefined) }));

// ---------------------------------------------------------------------------
// Fakes: Tauri invoke (engine manager, control bridge, vault) and WebSocket. No real engine,
// vault, or provider is involved; every secret below is a fake.
// ---------------------------------------------------------------------------

const SECRET_KEY = "sk-test-not-a-real-key-0000";
const SECRET_HF = "hf_test_not_a_real_token_0000";
const EVENT_TOKEN = "e".repeat(64);
const EVENT_TOKEN_2 = "f".repeat(64);
const CONTROL_TOKEN = "c0".repeat(32);

type Handler = ((event?: unknown) => void) | null;

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  onopen: Handler = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event?: { code: number }) => void) | null = null;
  onerror: Handler = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close(code = 1000) {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  receive(frame: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(frame) });
  }
}

type InvokeHandler = (args: Record<string, unknown>) => unknown;

let handlers: Record<string, InvokeHandler>;
let vault: Map<string, string>;
let engineInfo: Record<string, unknown>;

function vaultKey(args: Record<string, unknown>) {
  return `${args.capability}:${args.configId}:${args.slot}`;
}

function installHandlers() {
  vault = new Map();
  engineInfo = { port: 5101, eventToken: EVENT_TOKEN };
  handlers = {
    engine_start: () => ({ ...engineInfo }),
    engine_status: () => ({ state: "running", port: engineInfo.port }),
    engine_forward_start: () => ({ messages: [{ type: "status", status: "started" }] }),
    engine_forward_stop: () => ({ messages: [{ type: "status", status: "stopped" }] }),
    engine_forward_command: (args) => ({
      messages: forwardReplies((args.command as { type: string }).type, args.command),
    }),
    credential_save: (args) => {
      if (vault.has(vaultKey(args))) {
        throw { code: "already_configured", message: "fixed", capability: args.capability };
      }
      vault.set(vaultKey(args), args.secret as string);
      return { configured: true };
    },
    credential_replace: (args) => {
      if (!vault.has(vaultKey(args))) throw { code: "not_configured", message: "fixed" };
      vault.set(vaultKey(args), args.secret as string);
      return { configured: true };
    },
    credential_delete: (args) => {
      vault.delete(vaultKey(args));
      return { configured: false };
    },
    credential_status: (args) => ({ configured: vault.has(vaultKey(args)) }),
  };
  invokeMock.mockImplementation(async (command: string, args?: Record<string, unknown>) => {
    const handler = handlers[command];
    if (!handler) throw { code: "unknown_command", message: "fixed" };
    return handler(args ?? {});
  });
}

// Correlated replies the Rust bridge returns for each allowlisted command, up to its terminal.
function forwardReplies(type: string, command: unknown): Record<string, unknown>[] {
  const c = command as Record<string, unknown>;
  switch (type) {
    case "list_devices":
      return [{ type: "devices_list", microphones: [{ id: "m1", name: "Mic" }], speakers: [] }];
    case "request_models_list":
      return [{ type: "models_list", asr: [{ id: "tiny" }], translation: [] }];
    case "download_model":
      return [];
    case "delete_model":
      return [{ type: "model_action_result", action: "delete", model_id: c.model_id, success: true, error: null }];
    case "search_hub":
      return [{ type: "hub_search_results", results: [{ id: "openai/whisper-tiny" }] }];
    case "request_suggestion":
      return [{ type: "suggestion_result", entry_id: c.entry_id, options: [{ text: "Sure" }] }];
    case "request_summary":
      return [
        { type: "llm_chunk", text: "Part one. " },
        { type: "llm_chunk", text: "Part two." },
        { type: "llm_done" },
      ];
    default:
      throw { code: "command_not_allowed", message: "fixed" };
  }
}

function calls(command: string): Record<string, unknown>[] {
  return invokeMock.mock.calls
    .filter(([name]) => name === command)
    .map(([, args]) => (args ?? {}) as Record<string, unknown>);
}

function memoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  const storage: StateStorage = {
    getItem: (name) => data.get(name) ?? null,
    setItem: (name, value) => {
      data.set(name, value);
    },
    removeItem: (name) => {
      data.delete(name);
    },
  };
  return { data, storage };
}

async function flush() {
  for (let i = 0; i < 50; i += 1) await Promise.resolve();
}

function lastSocket(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

function authenticate(ws: FakeWebSocket) {
  ws.open();
  ws.receive({ op: "auth_ok", role: "event", protocol: 1 });
}

async function secureStore(initial: Record<string, string> = {}) {
  const mem = memoryStorage(initial);
  const store = createEngineStore({ storage: mem.storage, reconnectDelayMs: 1000 });
  await store.getState().bootstrapEngine({ storage: mem.storage });
  await flush();
  return { store, data: mem.data, storage: mem.storage };
}

async function authenticatedStore(initial: Record<string, string> = {}) {
  const ctx = await secureStore(initial);
  const ws = lastSocket();
  authenticate(ws);
  return { ...ctx, ws };
}

function legacyBlob(config: Record<string, unknown>) {
  return JSON.stringify({
    version: 0,
    state: {
      theme: "light",
      config: { modelSize: "medium", ...config },
      history: [{ id: "c-old", name: "Old", date: 2, entries: [] }],
    },
  });
}

// Everything observable about the store as data: functions dropped, sockets reduced to their
// enumerable fields.
function serializedState(state: EngineState): string {
  return JSON.stringify(state, (_key, value) => (typeof value === "function" ? undefined : value));
}

function final(ws: FakeWebSocket, id: string, text: string) {
  ws.receive({ op: "event", event: { type: "final", entry_id: id, text, timestamp: 1 } });
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__ = () => undefined;
  window.localStorage.clear();
  installHandlers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  invokeMock.mockReset();
  delete (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__;
});

// ---------------------------------------------------------------------------
// Bridge wrapper
// ---------------------------------------------------------------------------

describe("engine bridge wrapper", () => {
  it("keeps only the port and event token from engine_start", async () => {
    engineInfo = { port: 5101, eventToken: EVENT_TOKEN, controlToken: CONTROL_TOKEN };
    const info = await startOwnedEngine();
    expect(info).toEqual({ port: 5101, eventToken: EVENT_TOKEN });
    expect(JSON.stringify(info)).not.toContain(CONTROL_TOKEN);

    engineInfo = { port: 0, eventToken: EVENT_TOKEN };
    await expect(startOwnedEngine()).rejects.toMatchObject({ code: "invalid_response" });
  });

  it("builds error text from the code only", async () => {
    handlers.engine_start = () => {
      throw { code: "spawn_failed", message: `C:\\private\\path ${SECRET_KEY}` };
    };
    const error = (await startOwnedEngine().catch((e) => e)) as BridgeFailure;
    expect(error).toBeInstanceOf(BridgeFailure);
    expect(error.code).toBe("spawn_failed");
    expect(error.message).not.toContain(SECRET_KEY);
    expect(error.message).not.toContain("private");

    handlers.engine_start = () => {
      throw `free text ${SECRET_KEY}`;
    };
    const stringError = (await startOwnedEngine().catch((e) => e)) as BridgeFailure;
    expect(stringError.code).toBe("bridge_failed");
    expect(stringError.message).not.toContain(SECRET_KEY);
  });

  it("accepts only event-plane frames an event client can receive", () => {
    expect(parseEventPlaneFrame('{"op":"auth_ok","role":"control","protocol":1}')).toBeNull();
    expect(parseEventPlaneFrame('{"op":"event","event":{"no":"type"}}')).toBeNull();
    expect(parseEventPlaneFrame('{"type":"final","text":"legacy shape"}')).toBeNull();
    expect(parseEventPlaneFrame("{not json")).toBeNull();
    expect(parseEventPlaneFrame('{"op":"error","code":"Bad Code!","message":"x"}')).toEqual({
      op: "error",
      code: "error",
      message: "",
    });
  });

  it("names the same legacy storage key as the store", () => {
    expect(LEGACY_STORAGE_KEY).toBe(LEGACY_PERSIST_STORAGE_NAME);
  });
});

// ---------------------------------------------------------------------------
// Authenticated event plane
// ---------------------------------------------------------------------------

describe("authenticated event plane", () => {
  it("starts the owned engine and authenticates before processing any event", async () => {
    const { store } = await secureStore();
    expect(store.getState().transport).toBe("secure");
    expect(calls("engine_start")).toHaveLength(1);

    const ws = lastSocket();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(ws.url).toBe("ws://127.0.0.1:5101");

    ws.open();
    expect(ws.sent.map((s) => JSON.parse(s))).toEqual([
      { op: "auth", role: "event", token: EVENT_TOKEN, protocol: 1 },
    ]);

    final(ws, "e-early", "before auth");
    ws.receive({ type: "final", entry_id: "e-raw", text: "legacy frame", timestamp: 1 });
    expect(store.getState()).toMatchObject({ connected: false, entries: [] });

    ws.receive({ op: "auth_ok", role: "event", protocol: 1 });
    expect(store.getState().connected).toBe(true);

    final(ws, "e-1", "after auth");
    ws.receive({ type: "final", entry_id: "e-raw", text: "legacy frame", timestamp: 1 });
    expect(store.getState().entries.map((e) => e.id)).toEqual(["e-1"]);

    ws.receive({ op: "error", code: "forbidden", message: `echo ${SECRET_KEY}` });
    expect(store.getState().appError).toContain("forbidden");
    expect(store.getState().appError).not.toContain(SECRET_KEY);
  });

  it("never exposes the event or control token through store state or persistence", async () => {
    engineInfo = { port: 5101, eventToken: EVENT_TOKEN, controlToken: CONTROL_TOKEN };
    const { store, data } = await authenticatedStore();
    const state = store.getState();
    for (const token of [EVENT_TOKEN, CONTROL_TOKEN]) {
      expect(serializedState(state)).not.toContain(token);
      expect(JSON.stringify(selectPersistedState(state))).not.toContain(token);
      expect(data.get(PERSIST_STORAGE_NAME) ?? "").not.toContain(token);
    }
    for (const socket of FakeWebSocket.instances) {
      expect(socket.sent.join("")).not.toContain(CONTROL_TOKEN);
    }
  });

});

// ---------------------------------------------------------------------------
// Forwarded legacy commands (engine_forward_command)
// ---------------------------------------------------------------------------

describe("forwarded commands through the control bridge", () => {
  function forwardedTypes(): string[] {
    return calls("engine_forward_command").map((a) => (a.command as { type: string }).type);
  }

  it("routes all seven allowlisted actions through invoke and applies the replies", async () => {
    const { store, ws } = await authenticatedStore();
    final(ws, "e-1", "question?");

    store.getState().requestDeviceList();
    await flush();
    expect(store.getState().availableMics).toEqual([{ id: "m1", name: "Mic" }]);

    store.getState().requestModelsList();
    await flush();
    expect(store.getState().modelsList.asr).toEqual([{ id: "tiny" }]);

    store.getState().searchHub("whisper", "asr");
    expect(store.getState().searchingHub).toBe(true);
    await flush();
    expect(store.getState()).toMatchObject({
      searchingHub: false,
      hubSearchResults: [{ id: "openai/whisper-tiny" }],
    });

    store.getState().downloadModelById("tiny", "asr");
    await flush();
    // Progress and the result arrive later on the event socket.
    expect(store.getState()).toMatchObject({ downloading: true, downloadModel: "tiny" });
    ws.receive({ op: "event", event: { type: "download_progress", model: "tiny", percent: 100 } });
    await flush();
    expect(store.getState().downloading).toBe(false);

    store.getState().deleteModelById("tiny", "asr");
    await flush();

    await store.getState().requestSuggestion("e-1", "question?", ["question?"]);
    await flush();
    expect(store.getState().suggestionLoading).toEqual({});
    expect(store.getState().entries[0].suggestions?.options).toEqual([{ text: "Sure" }]);

    store.getState().requestSummary();
    await flush();

    expect(new Set(forwardedTypes())).toEqual(
      new Set([
        "list_devices",
        "request_models_list",
        "search_hub",
        "download_model",
        "delete_model",
        "request_suggestion",
        "request_summary",
      ]),
    );
    // download_progress 100 and a successful delete each refresh the model list the same way.
    expect(forwardedTypes().filter((t) => t === "request_models_list").length).toBeGreaterThan(1);
    expect(calls("engine_forward_start")).toHaveLength(0);
    expect(store.getState().appError).toBeNull();

    // The event socket carried the auth frame and nothing else.
    expect(ws.sent).toHaveLength(1);
    expect(JSON.parse(ws.sent[0]).op).toBe("auth");
  });

  it("passes the hf_token vault reference only when configured and never a raw secret", async () => {
    const { store, ws } = await authenticatedStore();

    store.getState().searchHub("whisper", "asr");
    await flush();
    expect(calls("engine_forward_command")[0]).not.toHaveProperty("credentials");

    // A draft typed in Settings is committed to the vault before the command leaves.
    store.getState().updateConfig({ hfToken: SECRET_HF, llmApiKey: SECRET_KEY });
    store.getState().downloadModelById("tiny", "asr");
    await flush();
    store.getState().searchHub("whisper", "asr");
    await flush();
    store.getState().requestDeviceList();
    await flush();

    expect(vault.get("stt:legacy-hf-hub:hf_token")).toBe(SECRET_HF);
    expect(store.getState().config.hfToken).toBe("");

    const [, download, search, devices] = calls("engine_forward_command");
    const ref = { hf_token: { capability: "stt", configId: "legacy-hf-hub", slot: "hf_token" } };
    expect(download.credentials).toEqual(ref);
    expect(search.credentials).toEqual(ref);
    expect(devices).not.toHaveProperty("credentials");
    expect(download.command).toEqual({ type: "download_model", model_id: "tiny", model_type: "asr" });
    expect(search.command).toEqual({ type: "search_hub", query: "whisper", task: "asr" });

    final(ws, "e-1", "question?");
    await store.getState().requestSuggestion("e-1", "question?", ["question?"]);
    store.getState().requestSummary();
    await flush();
    const assistant = calls("engine_forward_command").slice(-2);
    for (const args of assistant) {
      const llmConfig = (args.command as { llm_config: Record<string, unknown> }).llm_config;
      expect(llmConfig).not.toHaveProperty("api_key");
      expect(llmConfig.model).toBe(DEFAULT_CONFIG.llmModel);
      expect(args).not.toHaveProperty("credentials");
    }

    const everything = JSON.stringify(calls("engine_forward_command"));
    expect(everything).not.toContain(SECRET_HF);
    expect(everything).not.toContain(SECRET_KEY);
    expect(everything).not.toMatch(/"(hf_token|api_key|token)":\s*"/);
    expect(ws.sent.join("")).not.toContain(SECRET_HF);
    expect(ws.sent).toHaveLength(1);
  });

  it("delivers a batched summary stream in order and ends on llm_done", async () => {
    const { store, ws } = await authenticatedStore();
    final(ws, "e-1", "we agreed");
    store.getState().requestSummary();
    expect(store.getState().summaryLoading).toBe(true);
    await flush();

    expect(store.getState()).toMatchObject({
      summaryText: "Part one. Part two.",
      summaryLoading: false,
      summaryVisible: true,
    });
    const [args] = calls("engine_forward_command");
    expect(args.command).toMatchObject({
      type: "request_summary",
      entries: [{ text: "we agreed", source: "speaker" }],
    });
    expect(ws.sent).toHaveLength(1);
  });

  it("ignores replies that resolve after the engine generation changed", async () => {
    const { store, ws } = await authenticatedStore();
    final(ws, "e-1", "question?");
    const pending: ((value: unknown) => void)[] = [];
    handlers.engine_forward_command = () => new Promise((resolve) => pending.push(resolve));

    store.getState().requestSummary();
    store.getState().requestDeviceList();
    await flush();
    expect(pending).toHaveLength(2);

    ws.close(1006);
    engineInfo = { port: 5202, eventToken: EVENT_TOKEN_2 };
    store.getState().connect();
    await flush();
    authenticate(lastSocket());
    const generation = store.getState().engineGeneration;

    pending[0]({ messages: [{ type: "llm_chunk", text: "stale" }, { type: "llm_done" }] });
    pending[1]({ messages: [{ type: "devices_list", microphones: [{ id: "old", name: "Old" }], speakers: [] }] });
    await flush();

    expect(store.getState().engineGeneration).toBe(generation);
    expect(store.getState().summaryText).toBe("");
    expect(store.getState().availableMics).toEqual([]);
    expect(store.getState().appError).toBeNull();
  });

  it("clears loading and shows fixed-text toasts on bridge errors", async () => {
    const { store, ws } = await authenticatedStore();
    final(ws, "e-1", "question?");

    const failWith = (code: string) => {
      handlers.engine_forward_command = () => {
        throw { code, message: `echo ${SECRET_KEY} C:\\private\\path` };
      };
    };

    failWith("timeout");
    await store.getState().requestSuggestion("e-1", "question?", ["question?"]);
    await flush();
    expect(store.getState().suggestionLoading).toEqual({});
    expect(store.getState().appError).toBe("Suggestion: The engine did not answer in time (timeout)");

    failWith("command_rejected");
    store.getState().requestSummary();
    await flush();
    expect(store.getState().summaryLoading).toBe(false);
    expect(store.getState().appError).toBe("Summary: The engine rejected this request (command_rejected)");

    failWith("engine_not_running");
    store.getState().searchHub("whisper", "asr");
    await flush();
    expect(store.getState().searchingHub).toBe(false);
    expect(store.getState().appError).toBe("Hub search: The engine is not running (engine_not_running)");

    failWith("vault_error");
    store.getState().downloadModelById("tiny", "asr");
    await flush();
    expect(store.getState()).toMatchObject({ downloading: false, downloadModel: "" });
    expect(store.getState().appError).toContain("(vault_error)");

    for (const code of ["secret_in_command", "invalid_command", "command_not_allowed"]) {
      failWith(code);
      store.getState().requestDeviceList();
      await flush();
      expect(store.getState().appError).toContain(`Device list: `);
      expect(store.getState().appError).toContain(`(${code})`);
    }

    handlers.engine_forward_command = () => {
      throw `free text ${SECRET_KEY}`;
    };
    store.getState().deleteModelById("tiny", "asr");
    await flush();
    expect(store.getState().appError).toBe("Model delete: The engine request failed (bridge_failed)");

    expect(serializedState(store.getState())).not.toContain(SECRET_KEY);
    expect(serializedState(store.getState())).not.toContain("private");
    expect(ws.sent).toHaveLength(1);
  });

  it("clears loading when the replies end without their terminal message", async () => {
    const { store, ws } = await authenticatedStore();
    final(ws, "e-1", "question?");
    handlers.engine_forward_command = () => ({ messages: [{ type: "error", message: "LLM is not configured" }] });

    await store.getState().requestSuggestion("e-1", "question?", ["question?"]);
    store.getState().requestSummary();
    store.getState().downloadModelById("tiny", "asr");
    await flush();
    expect(store.getState()).toMatchObject({
      suggestionLoading: {},
      summaryLoading: false,
      downloading: false,
      appError: "LLM is not configured",
    });
  });

  it("refuses non-allowlisted types before invoke and in the wrapper", async () => {
    const { store } = await authenticatedStore();
    store.getState().socket?.send(JSON.stringify({ type: "start", config: {} }));
    await flush();
    expect(calls("engine_forward_command")).toHaveLength(0);
    expect(store.getState().appError).toBe(
      "This action is not available on the secure engine connection",
    );

    await expect(
      forwardCommand({ type: "stop" } as unknown as ForwardedCommand),
    ).rejects.toMatchObject({ code: "command_not_allowed" });
    expect(calls("engine_forward_command")).toHaveLength(0);
  });

  it("strips secret-named keys at every depth", () => {
    expect(
      stripCommandSecrets({
        type: "request_summary",
        hf_token: SECRET_HF,
        llm_config: { api_key: SECRET_KEY, model: "m", nested: [{ Authorization: "x", keep: 1 }] },
      }),
    ).toEqual({ type: "request_summary", llm_config: { model: "m", nested: [{ keep: 1 }] } });
  });
});

describe("browser legacy path for the same actions", () => {
  it("still sends commands on the plain socket without invoke", async () => {
    delete (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__;
    const store = createEngineStore({ storage: memoryStorage().storage, reconnectDelayMs: 1000 });
    await store.getState().bootstrapEngine();
    const ws = lastSocket();
    ws.open();
    store.getState().requestDeviceList();
    store.getState().searchHub("whisper", "asr");
    expect(ws.sent.map((s) => JSON.parse(s).type)).toEqual(["list_devices", "search_hub"]);
    expect(invokeMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Start / Stop through Tauri
// ---------------------------------------------------------------------------

describe("start and stop through the control bridge", () => {
  it("forwards public config plus credential references and routes the replies", async () => {
    vault.set("assistant:legacy-llm:api_key", SECRET_KEY);
    const { store, ws } = await authenticatedStore();
    expect(store.getState().secretStatus.llmApiKey).toBe(true);

    store.getState().updateConfig({ modelSize: "tiny", llmEnabled: true, micDeviceId: "" });
    store.getState().togglePipeline();
    expect(store.getState().isToggling).toBe(true);
    await flush();

    const [args] = calls("engine_forward_start");
    const request = args.request as {
      config: Record<string, unknown>;
      credentials: Record<string, unknown>;
    };
    expect(request.config["asr.model_size"]).toBe("tiny");
    expect(request.config["llm.enabled"]).toBe(true);
    expect("audio.mic_device_id" in request.config).toBe(false);
    expect(Object.keys(request.config).some((k) => /api_key|token/.test(k))).toBe(false);
    expect(request.credentials).toEqual({
      "llm.api_key": { capability: "assistant", configId: "legacy-llm", slot: "api_key" },
    });
    expect(JSON.stringify(args)).not.toContain(SECRET_KEY);

    expect(store.getState()).toMatchObject({ running: true, isToggling: false });

    store.getState().togglePipeline();
    await flush();
    expect(calls("engine_forward_stop")).toHaveLength(1);
    expect(store.getState()).toMatchObject({ running: false, isToggling: false });

    expect(ws.sent).toHaveLength(1); // the auth frame only
  });

  it("sends no credential reference when the vault has no key", async () => {
    const { store } = await authenticatedStore();
    store.getState().startPipeline();
    await flush();
    const request = calls("engine_forward_start")[0].request as { credentials: unknown };
    expect(request.credentials).toEqual({});
  });

  it("commits a typed key draft to the vault before Start and forwards only its reference", async () => {
    const { store } = await authenticatedStore();
    store.getState().updateConfig({ llmApiKey: SECRET_KEY });
    store.getState().startPipeline();
    await flush();

    const order = invokeMock.mock.calls.map(([name]) => name);
    expect(order.indexOf("credential_save")).toBeLessThan(order.indexOf("engine_forward_start"));
    expect(vault.get("assistant:legacy-llm:api_key")).toBe(SECRET_KEY);
    expect(store.getState().config.llmApiKey).toBe("");
    const [args] = calls("engine_forward_start");
    expect(JSON.stringify(args)).not.toContain(SECRET_KEY);
    expect((args.request as { credentials: object }).credentials).toHaveProperty("llm.api_key");
  });

  it("does not start while disconnected and reports bridge failures by code", async () => {
    const { store } = await secureStore();
    store.getState().startPipeline();
    await flush();
    expect(calls("engine_forward_start")).toHaveLength(0);

    authenticate(lastSocket());
    handlers.engine_forward_start = () => {
      throw { code: "command_rejected", message: `raw ${SECRET_KEY}` };
    };
    store.getState().startPipeline();
    await flush();
    expect(store.getState()).toMatchObject({ running: false, isToggling: false });
    expect(store.getState().appError).toContain("command_rejected");
    expect(store.getState().appError).not.toContain(SECRET_KEY);
  });
});

// ---------------------------------------------------------------------------
// Reconnect and engine generations
// ---------------------------------------------------------------------------

describe("reconnect and engine generations", () => {
  it("reconnects to the same run after the delay and re-authenticates", async () => {
    vi.useFakeTimers();
    const { store, ws } = await authenticatedStore();
    const generation = store.getState().engineGeneration;

    ws.close(1006);
    expect(store.getState()).toMatchObject({ connected: false, socket: null });
    await vi.advanceTimersByTimeAsync(999);
    expect(calls("engine_start")).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await flush();

    expect(calls("engine_start")).toHaveLength(2);
    expect(FakeWebSocket.instances).toHaveLength(2);
    const second = lastSocket();
    expect(second.url).toBe("ws://127.0.0.1:5101");
    second.open();
    expect(JSON.parse(second.sent[0])).toMatchObject({ op: "auth", token: EVENT_TOKEN });
    second.receive({ op: "auth_ok", role: "event", protocol: 1 });
    expect(store.getState().connected).toBe(true);
    expect(store.getState().engineGeneration).toBe(generation);
  });

  it("cancels a stale reconnect when the owned engine generation changes", async () => {
    vi.useFakeTimers();
    const { store, ws: first } = await authenticatedStore();
    const generation = store.getState().engineGeneration;

    first.close(1006); // arms a reconnect for the first run
    engineInfo = { port: 5202, eventToken: EVENT_TOKEN_2 };
    store.getState().connect(); // Rust now reports a different run
    await flush();

    expect(store.getState().engineGeneration).toBe(generation + 1);
    const second = lastSocket();
    expect(second.url).toBe("ws://127.0.0.1:5202");

    await vi.advanceTimersByTimeAsync(10_000);
    await flush();
    expect(calls("engine_start")).toHaveLength(2);
    expect(FakeWebSocket.instances).toHaveLength(2);

    second.open();
    expect(JSON.parse(second.sent[0]).token).toBe(EVENT_TOKEN_2);
    second.receive({ op: "auth_ok", role: "event", protocol: 1 });

    // Frames and closes from the previous run's socket are ignored.
    final(first, "e-stale", "from the old run");
    first.onclose?.({ code: 1006 });
    expect(store.getState()).toMatchObject({ connected: true, entries: [] });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("drops an engine_start that resolves after disconnect", async () => {
    const { store, ws } = await authenticatedStore();
    let resolveStart: (value: unknown) => void = () => undefined;
    handlers.engine_start = () => new Promise((resolve) => (resolveStart = resolve));

    ws.close(1006);
    store.getState().connect();
    await flush();
    store.getState().disconnect();
    resolveStart({ port: 5303, eventToken: EVENT_TOKEN_2 });
    await flush();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(store.getState()).toMatchObject({ connected: false, socket: null });
  });

  it("does not reconnect after an explicit disconnect", async () => {
    vi.useFakeTimers();
    const { store } = await authenticatedStore();
    store.getState().disconnect();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls("engine_start")).toHaveLength(1);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("recovers from a rejected event token by taking the run Rust reports next", async () => {
    vi.useFakeTimers();
    const { store } = await secureStore();
    const ws = lastSocket();
    ws.open();
    ws.receive({ op: "error", code: "auth_failed", message: "Authentication failed" });
    ws.close(4401);
    expect(store.getState().connected).toBe(false);
    expect(store.getState().appError).toContain("auth_failed");

    engineInfo = { port: 5101, eventToken: EVENT_TOKEN_2 };
    await vi.advanceTimersByTimeAsync(1000);
    await flush();
    const retry = lastSocket();
    retry.open();
    expect(JSON.parse(retry.sent[0]).token).toBe(EVENT_TOKEN_2);
  });

  it("reports an engine start failure and retries on the same generation", async () => {
    vi.useFakeTimers();
    handlers.engine_start = () => {
      throw { code: "engine_not_found", message: "C:\\path\\to\\engine" };
    };
    const { store } = await secureStore();
    expect(FakeWebSocket.instances).toHaveLength(0);
    expect(store.getState().appError).toContain("engine_not_found");
    expect(store.getState().appError).not.toContain("path");

    handlers.engine_start = () => ({ ...engineInfo });
    await vi.advanceTimersByTimeAsync(1000);
    await flush();
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Legacy key migration
// ---------------------------------------------------------------------------

describe("legacy key migration", () => {
  it("moves legacy secrets to the vault, confirms, and removes every raw copy", async () => {
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY, hfToken: SECRET_HF });
    const { store, data } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    const state = store.getState();

    expect(calls("credential_save").map((a) => [a.capability, a.configId, a.slot])).toEqual([
      ["assistant", "legacy-llm", "api_key"],
      ["stt", "legacy-hf-hub", "hf_token"],
    ]);
    expect(calls("credential_status").length).toBeGreaterThanOrEqual(2);
    expect(vault.get("assistant:legacy-llm:api_key")).toBe(SECRET_KEY);
    expect(vault.get("stt:legacy-hf-hub:hf_token")).toBe(SECRET_HF);

    expect(state.config.llmApiKey).toBe("");
    expect(state.config.hfToken).toBe("");
    expect(state.secretStatus).toEqual({ llmApiKey: true, hfToken: true });
    expect(state.services.assistant.credentialRefs).toEqual({
      api_key: { configId: "legacy-llm", slot: "api_key", version: 1 },
    });
    expect(state.appError).toBeNull();

    const scrubbed = data.get(LEGACY_PERSIST_STORAGE_NAME) as string;
    expect(scrubbed).not.toContain(SECRET_KEY);
    expect(scrubbed).not.toContain(SECRET_HF);
    expect(JSON.parse(scrubbed).state.history[0].id).toBe("c-old");
    expect(data.get(PERSIST_STORAGE_NAME)).not.toContain(SECRET_KEY);
    expect(serializedState(state)).not.toContain(SECRET_KEY);
    expect(serializedState(state)).not.toContain(SECRET_HF);
  });

  it("keeps the raw value when the vault write fails and never echoes it", async () => {
    handlers.credential_save = () => {
      throw { code: "vault_unavailable", message: `locked while saving ${SECRET_KEY}` };
    };
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY });
    const { store, data } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    const state = store.getState();

    expect(state.config.llmApiKey).toBe(SECRET_KEY);
    expect(data.get(LEGACY_PERSIST_STORAGE_NAME)).toBe(legacy);
    expect(state.secretStatus.llmApiKey).toBe(false);
    expect(state.services.assistant.credentialRefs).toEqual({});
    expect(state.appError).toContain("vault_unavailable");
    expect(state.appError).not.toContain(SECRET_KEY);
    // The engine still comes up; only the migration is deferred.
    expect(calls("engine_start")).toHaveLength(1);
  });

  it("treats an unconfirmed write as a failure", async () => {
    handlers.credential_status = () => ({ configured: false });
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY });
    const { store, data } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    expect(store.getState().config.llmApiKey).toBe(SECRET_KEY);
    expect(data.get(LEGACY_PERSIST_STORAGE_NAME)).toBe(legacy);
    expect(store.getState().appError).toContain("vault_unconfirmed");
  });

  it("retries after a failure and is idempotent once migrated", async () => {
    const failingSave = () => {
      throw { code: "vault_failure", message: "fixed" };
    };
    const workingSave = handlers.credential_save;
    handlers.credential_save = failingSave;
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY });
    const { store, data, storage } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    expect(store.getState().config.llmApiKey).toBe(SECRET_KEY);

    handlers.credential_save = workingSave;
    const retry = await store.getState().commitLegacySecrets(storage);
    expect(retry.find((o) => o.field === "llmApiKey")?.status).toBe("migrated");
    expect(store.getState().config.llmApiKey).toBe("");
    expect(data.get(LEGACY_PERSIST_STORAGE_NAME)).not.toContain(SECRET_KEY);

    const saves = calls("credential_save").length;
    const again = await store.getState().commitLegacySecrets(storage);
    expect(again.map((o) => o.status)).toEqual(["absent", "absent"]);
    expect(calls("credential_save")).toHaveLength(saves);

    // A later launch over the scrubbed storage has nothing to migrate.
    const relaunch = createEngineStore({ storage, reconnectDelayMs: 1000 });
    expect(relaunch.getState().config.llmApiKey).toBe("");
    await relaunch.getState().bootstrapEngine({ storage });
    await flush();
    expect(calls("credential_save")).toHaveLength(saves);
    expect(relaunch.getState().secretStatus.llmApiKey).toBe(true);
  });

  it("replaces an already-configured entry and bumps the reference version", async () => {
    vault.set("assistant:legacy-llm:api_key", "sk-older-fake-value-1111");
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY });
    const { store } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    expect(calls("credential_replace")).toHaveLength(1);
    expect(vault.get("assistant:legacy-llm:api_key")).toBe(SECRET_KEY);
    expect(store.getState().services.assistant.credentialRefs.api_key.version).toBe(1);

    store.getState().setActiveView("settings");
    store.getState().updateConfig({ llmApiKey: "sk-newer-fake-value-2222" });
    store.getState().setActiveView("transcript"); // leaving Settings commits the draft
    await flush();
    expect(store.getState().config.llmApiKey).toBe("");
    expect(store.getState().services.assistant.credentialRefs.api_key.version).toBe(2);
  });

  it("joins concurrent commits into one vault write per secret", async () => {
    const { store, storage } = await secureStore();
    store.getState().updateConfig({ llmApiKey: SECRET_KEY });
    const a = store.getState().commitLegacySecrets(storage);
    const b = store.getState().commitLegacySecrets(storage);
    expect(a).toBe(b);
    await a;
    expect(calls("credential_save")).toHaveLength(1);
  });

  it("does not migrate outside the Tauri runtime", async () => {
    delete (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__;
    const legacy = legacyBlob({ llmApiKey: SECRET_KEY });
    const { store, data } = await secureStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    expect(store.getState().transport).toBe("legacy");
    expect(invokeMock).not.toHaveBeenCalled();
    expect(data.get(LEGACY_PERSIST_STORAGE_NAME)).toBe(legacy);
    expect(lastSocket().url).toBe("ws://127.0.0.1:8765");
  });
});

// ---------------------------------------------------------------------------
// Validation and Test Connection
// ---------------------------------------------------------------------------

describe("validation and Test Connection", () => {
  const VALID = { status: "valid" as const, fingerprint: "a".repeat(64), checkedAt: 1, errorCode: null };

  function validationResult(overrides: Record<string, unknown> = {}) {
    return {
      capability: "assistant",
      providerId: "openai",
      status: "valid",
      latencyMs: 12,
      testedAt: 1_700_000_000,
      fingerprint: "b".repeat(64),
      credentialVersion: 1,
      errorCode: null,
      ...overrides,
    };
  }

  it("sends public config with references only and applies the result", async () => {
    vault.set("assistant:legacy-llm:api_key", SECRET_KEY);
    handlers.provider_test_connection = () => validationResult();
    const { store } = await authenticatedStore();
    store.getState().updateServiceConfig("assistant", { modelOverride: "gpt-x" });

    const validation = await store.getState().testServiceConnection("assistant");
    const [args] = calls("provider_test_connection");
    const request = args.request as { capability: string; service: Record<string, unknown> };
    expect(request.capability).toBe("assistant");
    expect(request.service).toEqual({
      enabled: false,
      providerId: "openai",
      endpoint: "",
      modelOverride: "gpt-x",
      options: {},
      credentialRefs: { api_key: { configId: "legacy-llm", slot: "api_key", version: 1 } },
    });
    expect(JSON.stringify(args)).not.toContain(SECRET_KEY);
    expect(validation).toEqual({
      status: "valid",
      fingerprint: "b".repeat(64),
      checkedAt: 1_700_000_000,
      errorCode: null,
    });
    expect(store.getState().services.assistant.validation).toEqual(validation);
  });

  it("discards a result whose config changed while the test ran", async () => {
    let resolveTest: (value: unknown) => void = () => undefined;
    handlers.provider_test_connection = () => new Promise((resolve) => (resolveTest = resolve));
    const { store } = await authenticatedStore();

    const pending = store.getState().testServiceConnection("assistant");
    await flush();
    store.getState().updateServiceConfig("assistant", { endpoint: "https://example.invalid/v1" });
    resolveTest(validationResult());
    expect(await pending).toBeNull();
    expect(store.getState().services.assistant.validation.status).toBe("unvalidated");
  });

  it("records a bridge failure as unvalidated with its code", async () => {
    handlers.provider_test_connection = () => {
      throw { code: "validator_unavailable", message: `detail ${SECRET_KEY}` };
    };
    const { store } = await authenticatedStore();
    const validation = await store.getState().testServiceConnection("assistant");
    expect(validation).toEqual({
      status: "unvalidated",
      fingerprint: null,
      checkedAt: null,
      errorCode: "validator_unavailable",
    });
    expect(serializedState(store.getState())).not.toContain(SECRET_KEY);
  });

  it("rejects a result for a different provider", async () => {
    handlers.provider_test_connection = () => validationResult({ providerId: "other" });
    const { store } = await authenticatedStore();
    const validation = await store.getState().testServiceConnection("assistant");
    expect(validation?.errorCode).toBe("invalid_response");
  });

  it("invalidates on public assistant edits and on credential changes", async () => {
    const { store } = await authenticatedStore();

    store.getState().setServiceValidation("assistant", VALID);
    store.getState().updateConfig({ targetLang: "de" });
    expect(store.getState().services.assistant.validation).toEqual(VALID);

    for (const change of [{ llmModel: "other/model" }, { llmProviderUrl: "http://localhost:11434/v1" }]) {
      store.getState().setServiceValidation("assistant", VALID);
      store.getState().updateConfig(change);
      expect(store.getState().services.assistant.validation.status).toBe("unvalidated");
    }

    store.getState().setServiceValidation("assistant", VALID);
    store.getState().updateConfig({ llmApiKey: SECRET_KEY });
    expect(store.getState().services.assistant.validation.status).toBe("unvalidated");
    store.getState().setServiceValidation("assistant", VALID);
    await store.getState().commitLegacySecrets();
    expect(store.getState().services.assistant.validation.status).toBe("unvalidated");
    expect(store.getState().services.assistant.credentialRefs.api_key.version).toBe(1);
  });

  it("builds the Test Connection payload without validation metadata", () => {
    const payload = testConnectionService({
      enabled: true,
      providerId: "openai",
      endpoint: "",
      modelOverride: "",
      options: { temperature: 0.2 },
      credentialRefs: {},
      validation: VALID,
    });
    expect(payload).not.toHaveProperty("validation");
  });
});

// ---------------------------------------------------------------------------
// Overlay sync
// ---------------------------------------------------------------------------

describe("overlay sync payload", () => {
  it("blanks legacy secrets and keeps public settings", () => {
    const snapshot = publicConfigSnapshot({
      ...DEFAULT_CONFIG,
      llmApiKey: SECRET_KEY,
      hfToken: SECRET_HF,
      stealthMode: true,
    });
    expect(snapshot).toMatchObject({ llmApiKey: "", hfToken: "", stealthMode: true });
    expect(JSON.stringify(snapshot)).not.toContain(SECRET_KEY);
    expect(JSON.stringify(snapshot)).not.toContain(SECRET_HF);
    expect(LEGACY_CREDENTIALS.llmApiKey.capability).toBe("assistant");
  });
});
