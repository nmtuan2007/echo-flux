import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import type { StateStorage } from "zustand/middleware";
import {
  createEngineStore,
  LEGACY_PERSIST_STORAGE_NAME,
  PERSIST_STORAGE_NAME,
  PERSIST_VERSION,
  selectPersistedState,
  useEngineStore,
  type EngineState,
} from "./engineStore";
import { LEGACY_SECRET_CONFIG_KEYS, PERSISTED_CONFIG_KEYS } from "./settingsSlice";
import { PROTOCOL_FAMILIES, type ProtocolFamily, type ProviderManifest } from "./types";

const emitMock = vi.hoisted(() => vi.fn(async () => undefined));
vi.mock("@tauri-apps/api/event", () => ({ emit: emitMock }));

// ---------------------------------------------------------------------------
// Fakes: WebSocket and storage boundaries. No engine or provider is involved.
// ---------------------------------------------------------------------------

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
  onclose: Handler = null;
  onerror: Handler = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  receive(message: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  messages(): Record<string, unknown>[] {
    return this.sent.map((s) => JSON.parse(s));
  }

  lastMessage(): Record<string, unknown> | undefined {
    const all = this.messages();
    return all[all.length - 1];
  }
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

function newStore(initial: Record<string, string> = {}) {
  const mem = memoryStorage(initial);
  const store = createEngineStore({ storage: mem.storage, reconnectDelayMs: 3000 });
  return { store, data: mem.data };
}

function connectedStore(initial: Record<string, string> = {}) {
  const ctx = newStore(initial);
  ctx.store.getState().connect();
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  ws.open();
  return { ...ctx, ws };
}

const SECRET_KEY = "sk-test-not-a-real-key-0000";
const SECRET_HF = "hf_test_not_a_real_token_0000";

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  emitMock.mockClear();
  delete (window as unknown as { __TAURI__?: unknown }).__TAURI__;
});

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

describe("persistence whitelist", () => {
  it("writes only theme, public config, and history", () => {
    const { store, ws, data } = connectedStore();
    store.getState().updateConfig({ llmApiKey: SECRET_KEY, hfToken: SECRET_HF, modelSize: "tiny" });
    ws.receive({ type: "final", entry_id: "e-1", text: "hello", timestamp: 1 });
    ws.receive({ type: "error", message: "transient failure" });
    store.getState().requestSuggestion("e-1", "hello", []);

    const raw = data.get(PERSIST_STORAGE_NAME) as string;
    expect(raw).toBeTypeOf("string");
    const parsed = JSON.parse(raw);

    expect(parsed.version).toBe(PERSIST_VERSION);
    expect(Object.keys(parsed.state).sort()).toEqual(["config", "history", "services", "theme"]);
    expect(Object.keys(parsed.state.config).sort()).toEqual([...PERSISTED_CONFIG_KEYS].sort());
    expect(parsed.state.config.modelSize).toBe("tiny");

    for (const forbidden of [
      SECRET_KEY,
      SECRET_HF,
      "llmApiKey",
      "hfToken",
      "socket",
      "suggestionLoading",
      "appError",
      "transient failure",
      "partials",
      "connected",
    ]) {
      expect(raw).not.toContain(forbidden);
    }
    // The secrets are still usable in memory by the legacy runtime.
    expect(store.getState().config.llmApiKey).toBe(SECRET_KEY);
  });

  it("classifies every legacy secret as non-persisted", () => {
    for (const key of LEGACY_SECRET_CONFIG_KEYS) {
      expect(PERSISTED_CONFIG_KEYS).not.toContain(key);
    }
  });

  it("drops injected secret-shaped and runtime fields during serialization", () => {
    const state = {
      ...useEngineStore.getState(),
      socket: { token: "socket-token" },
      config: {
        ...useEngineStore.getState().config,
        llmApiKey: SECRET_KEY,
        apiToken: "tok-injected",
        authorization: "Bearer injected",
      },
      history: [
        {
          id: "c1",
          name: "n",
          date: 1,
          headers: { Authorization: "Bearer injected" },
          entries: [{ id: "e1", text: "hi", translation: null, isFinal: true, timestamp: 1, token: "x" }],
        },
      ],
    } as unknown as EngineState;

    const serialized = JSON.stringify(selectPersistedState(state));
    for (const forbidden of [SECRET_KEY, "tok-injected", "Bearer", "socket-token", '"token"']) {
      expect(serialized).not.toContain(forbidden);
    }
    expect(serialized).toContain('"text":"hi"');
  });

  it("hydrates only whitelisted keys from versioned storage", () => {
    const blob = JSON.stringify({
      version: PERSIST_VERSION,
      state: {
        theme: "light",
        connected: true,
        socket: "not-a-socket",
        appError: "stale error",
        suggestionLoading: { e1: true },
        config: { host: "10.0.0.2", port: "not-a-number", llmApiKey: "sk-injected-into-v1" },
        history: [{ id: "c1", name: "Past", date: 5, entries: [] }],
      },
    });
    const { store } = newStore({ [PERSIST_STORAGE_NAME]: blob });
    const state = store.getState();

    expect(state.theme).toBe("light");
    expect(state.connected).toBe(false);
    expect(state.socket).toBeNull();
    expect(state.appError).toBeNull();
    expect(state.suggestionLoading).toEqual({});
    expect(state.config.host).toBe("10.0.0.2");
    expect(state.config.port).toBe(8765);
    expect(state.config.llmApiKey).toBe("");
    expect(state.history.map((c) => c.id)).toEqual(["c1"]);
  });

  it("migrates the legacy v0 blob without rewriting or copying its secrets", () => {
    const legacy = JSON.stringify({
      version: 0,
      state: {
        theme: "light",
        config: { modelSize: "medium", llmApiKey: SECRET_KEY, hfToken: SECRET_HF },
        history: [{ id: "c-old", name: "Old", date: 2, entries: [] }],
      },
    });
    const { store, data } = newStore({ [LEGACY_PERSIST_STORAGE_NAME]: legacy });
    const state = store.getState();

    expect(state.config.modelSize).toBe("medium");
    expect(state.history.map((c) => c.id)).toEqual(["c-old"]);
    // Read into memory for the legacy runtime...
    expect(state.config.llmApiKey).toBe(SECRET_KEY);
    expect(state.config.hfToken).toBe(SECRET_HF);
    // ...never written to the versioned blob, and the legacy blob is left untouched.
    const written = data.get(PERSIST_STORAGE_NAME) as string;
    expect(written).toContain('"medium"');
    expect(written).not.toContain(SECRET_KEY);
    expect(written).not.toContain(SECRET_HF);
    expect(data.get(LEGACY_PERSIST_STORAGE_NAME)).toBe(legacy);
  });

  it("persists schema-v2 services configuration across store instances", () => {
    const { store, data } = newStore();
    store.getState().updateServiceConfig("stt", {
      providerId: "deepgram",
      endpoint: "wss://custom.deepgram.com",
      modelOverride: "nova-2",
      options: { punctuate: true, tier: 2 },
      credentialRefs: { api_key: { configId: "deepgram", slot: "api_key", version: 1 } },
    });
    store.getState().updateServiceConfig("assistant", {
      enabled: true,
      providerId: "openai",
      modelOverride: "gpt-4o",
      options: { temperature: 0.7 },
      credentialRefs: { api_key: { configId: "openai", slot: "api_key", version: 2 } },
    });

    const raw = data.get(PERSIST_STORAGE_NAME) as string;
    expect(raw).toBeTypeOf("string");

    const { store: hydrated } = newStore({ [PERSIST_STORAGE_NAME]: raw });
    const s = hydrated.getState().services;
    expect(s.stt.providerId).toBe("deepgram");
    expect(s.stt.endpoint).toBe("wss://custom.deepgram.com");
    expect(s.stt.modelOverride).toBe("nova-2");
    expect(s.stt.options).toEqual({ punctuate: true, tier: 2 });
    expect(s.stt.credentialRefs.api_key).toEqual({ configId: "deepgram", slot: "api_key", version: 1 });

    expect(s.assistant.enabled).toBe(true);
    expect(s.assistant.providerId).toBe("openai");
    expect(s.assistant.modelOverride).toBe("gpt-4o");
    expect(s.assistant.options).toEqual({ temperature: 0.7 });
    expect(s.assistant.credentialRefs.api_key).toEqual({ configId: "openai", slot: "api_key", version: 2 });
  });

  it("enforces zero secrets in services persistence even when injected", () => {
    const rawServices = {
      stt: {
        enabled: true,
        providerId: "deepgram",
        endpoint: "wss://api.deepgram.com",
        modelOverride: "nova-3",
        options: {
          api_key: "sk-injected-secret",
          token: "secret-token",
          authorization: "Bearer secret-bearer",
          punctuate: true,
        },
        credentialRefs: {
          api_key: { configId: "deepgram", slot: "api_key", version: 1, secret: "leaked-secret" },
        },
      },
      translation: {
        enabled: true,
        providerId: "local",
        endpoint: "",
        modelOverride: "",
        options: { apiKey: "injected" },
        credentialRefs: {},
      },
      assistant: {
        enabled: false,
        providerId: "openai",
        endpoint: "",
        modelOverride: "",
        options: { secret_key: "leaked" },
        credentialRefs: {},
      },
    };

    const state = {
      ...useEngineStore.getState(),
      services: rawServices,
    } as unknown as EngineState;

    const persisted = selectPersistedState(state);
    const serialized = JSON.stringify(persisted);

    for (const forbidden of [
      "sk-injected-secret",
      "secret-token",
      "secret-bearer",
      "leaked-secret",
      "api_key",
      "secret_key",
    ]) {
      // In credentialRefs, slot is "api_key", so ensure raw secret strings are never present
      if (forbidden === "api_key") {
        expect(persisted.services.stt.options.api_key).toBeUndefined();
      } else {
        expect(serialized).not.toContain(forbidden);
      }
    }
    expect(persisted.services.stt.options.punctuate).toBe(true);
    expect((persisted.services.stt.credentialRefs.api_key as any).secret).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

describe("engine events", () => {
  it("applies partials with the anti-flicker heuristic", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "partial", text: "hello world", source: "mic", translation_backend: "marian" });
    expect(store.getState().partials.mic).toEqual({ text: "hello world", translation: null, source: "mic" });
    expect(store.getState().activeTranslationBackend).toBe("marian");

    ws.receive({ type: "partial", text: "hello", source: "mic", translation_backend: "online" });
    expect(store.getState().partials.mic.text).toBe("hello world");
    expect(store.getState().activeTranslationBackend).toBe("marian");

    ws.receive({ type: "partial", text: "hello worl", source: "mic" });
    expect(store.getState().partials.mic.text).toBe("hello worl");
  });

  it("turns finals into entries and clears that stream's partial", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "partial", text: "abc", source: "system" });
    ws.receive({ type: "partial", text: "mine", source: "mic" });
    ws.receive({ type: "final", entry_id: "e-1", text: "abc def", translation: null, timestamp: 12, source: "system" });

    const state = store.getState();
    expect(state.entries).toEqual([
      { id: "e-1", text: "abc def", translation: null, isFinal: true, timestamp: 12, source: "system" },
    ]);
    expect(state.partials.system).toBeUndefined();
    expect(state.partials.mic.text).toBe("mine");

    ws.receive({ type: "final", text: "   ", source: "mic" });
    expect(store.getState().entries).toHaveLength(1);
    expect(store.getState().partials.mic).toBeUndefined();

    ws.receive({ type: "final", text: "no id", timestamp: 13 });
    expect(store.getState().entries[1].id).toMatch(/^t-/);
  });

  it("matches translation updates by entry id, then text, then prefix, else discards", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "final", entry_id: "e-1", text: "first sentence", timestamp: 1 });
    ws.receive({ type: "final", entry_id: "e-2", text: "second sentence here", timestamp: 2 });

    ws.receive({ type: "translation_update", entry_id: "e-1", source_text: "ignored", translation: "T1" });
    ws.receive({ type: "translation_update", source_text: "second sentence here", translation: "T2" });
    expect(store.getState().entries.map((e) => e.translation)).toEqual(["T1", "T2"]);

    ws.receive({ type: "final", entry_id: "e-3", text: "a third sentence that is long", timestamp: 3 });
    ws.receive({ type: "translation_update", source_text: "third sentence", translation: "T3" });
    expect(store.getState().entries[2].translation).toBe("T3");

    const before = store.getState().entries;
    ws.receive({ type: "translation_update", entry_id: "nope", source_text: "unrelated", translation: "X" });
    ws.receive({ type: "translation_update", entry_id: "e-1", translation: "" });
    expect(store.getState().entries).toEqual(before);
  });

  it("respects revisions when the engine sends them", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "final", entry_id: "e-1", text: "draft", revision: 1, timestamp: 1 });
    ws.receive({ type: "final", entry_id: "e-1", text: "revised", revision: 2, timestamp: 1 });
    expect(store.getState().entries).toHaveLength(1);
    expect(store.getState().entries[0]).toMatchObject({ text: "revised", revision: 2 });

    ws.receive({ type: "final", entry_id: "e-1", text: "stale", revision: 1, timestamp: 1 });
    ws.receive({ type: "translation_update", entry_id: "e-1", translation: "old", revision: 1 });
    expect(store.getState().entries[0]).toMatchObject({ text: "revised", translation: null });

    ws.receive({ type: "translation_update", entry_id: "e-1", translation: "new", revision: 2 });
    expect(store.getState().entries[0].translation).toBe("new");
  });

  it("stores suggestion results on the entry and saves the session", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "final", entry_id: "e-1", text: "question?", timestamp: 1 });
    store.getState().requestSuggestion("e-1", "question?", ["question?"]);

    expect(store.getState().suggestionLoading).toEqual({ "e-1": true });
    expect(ws.lastMessage()).toMatchObject({
      type: "request_suggestion",
      entry_id: "e-1",
      target_text: "question?",
      context: ["question?"],
    });

    const options = [{ strategy: "direct", text: "Answer" }];
    ws.receive({ type: "suggestion_result", entry_id: "e-1", options });
    const state = store.getState();
    expect(state.suggestionLoading).toEqual({});
    expect(state.entries[0].suggestions).toEqual({ options, error: undefined });
    expect(state.history).toHaveLength(1);
    expect(state.history[0].entries[0].suggestions?.options).toEqual(options);
  });

  it("relays suggestions through the Tauri event boundary when not connected", async () => {
    const { store } = newStore();
    (window as unknown as { __TAURI__?: unknown }).__TAURI__ = {};
    await store.getState().requestSuggestion("e-9", "hi", ["hi"]);
    expect(emitMock).toHaveBeenCalledWith("relay_request_suggestion", {
      entryId: "e-9",
      targetText: "hi",
      context: ["hi"],
    });
    expect(store.getState().suggestionLoading).toEqual({});
  });

  it("streams a summary and saves it with the session", () => {
    const { store, ws } = connectedStore();
    store.getState().requestSummary();
    expect(ws.sent).toHaveLength(0); // no entries yet

    ws.receive({ type: "final", entry_id: "e-1", text: "we agreed", source: "mic", timestamp: 1 });
    ws.receive({ type: "final", entry_id: "e-2", text: "on a plan", timestamp: 2 });
    store.getState().requestSummary();
    expect(ws.lastMessage()).toMatchObject({
      type: "request_summary",
      entries: [
        { text: "we agreed", source: "mic" },
        { text: "on a plan", source: "speaker" },
      ],
    });
    expect(store.getState()).toMatchObject({ summaryLoading: true, summaryVisible: true, summaryText: "" });

    ws.receive({ type: "llm_chunk", text: "Plan " });
    ws.receive({ type: "llm_chunk", text: "agreed." });
    expect(store.getState().summaryText).toBe("Plan agreed.");
    ws.receive({ type: "llm_done" });
    expect(store.getState().summaryLoading).toBe(false);
    expect(store.getState().history[0].summary).toBe("Plan agreed.");

    store.getState().closeSummary();
    expect(store.getState().summaryVisible).toBe(false);
  });

  it("handles pipeline status, errors, devices, and model events", () => {
    const { store, ws } = connectedStore();
    store.getState().updateConfig({ modelSize: "tiny", micDeviceId: "" });
    store.getState().togglePipeline();
    const start = ws.lastMessage() as { type: string; config: Record<string, unknown> };
    expect(start.type).toBe("start");
    expect(start.config["asr.model_size"]).toBe("tiny");
    expect("audio.mic_device_id" in start.config).toBe(false);
    expect(store.getState().isToggling).toBe(true);

    ws.receive({ type: "status", status: "started" });
    expect(store.getState()).toMatchObject({ running: true, isToggling: false, entries: [] });

    ws.receive({ type: "final", entry_id: "e-1", text: "spoken", timestamp: 1 });
    store.getState().togglePipeline();
    expect(ws.lastMessage()).toEqual({ type: "stop" });
    ws.receive({ type: "status", status: "stopped" });
    expect(store.getState()).toMatchObject({ running: false, isToggling: false });
    expect(store.getState().history).toHaveLength(1);

    ws.receive({ type: "error", message: "boom" });
    expect(store.getState().appError).toBe("boom");
    store.getState().clearAppError();
    expect(store.getState().appError).toBeNull();

    ws.receive({ type: "devices_list", microphones: [{ id: "1", name: "Mic" }], speakers: [] });
    expect(store.getState().availableMics).toEqual([{ id: "1", name: "Mic" }]);
  });

  it("ignores malformed and unknown messages", () => {
    const { store, ws } = connectedStore();
    const before = store.getState();
    ws.onmessage?.({ data: "{not json" });
    ws.receive({ type: "not_a_real_type" });
    expect(store.getState()).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// Reconnect and reset
// ---------------------------------------------------------------------------

describe("reconnect and reset", () => {
  it("resets connection state and reconnects after exactly the delay", () => {
    vi.useFakeTimers();
    const { store, ws } = connectedStore();
    ws.receive({ type: "status", status: "started" });
    ws.receive({ type: "download_progress", model: "m", percent: 10 });

    ws.close(); // engine went away
    expect(store.getState()).toMatchObject({
      connected: false,
      socket: null,
      running: false,
      isToggling: false,
      downloading: false,
      activeTranslationBackend: null,
    });

    vi.advanceTimersByTime(2999);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(FakeWebSocket.instances[1].url).toBe("ws://127.0.0.1:8765");

    FakeWebSocket.instances[1].open();
    expect(store.getState().connected).toBe(true);
  });

  it("does not reconnect after an explicit disconnect", () => {
    vi.useFakeTimers();
    const { store } = connectedStore();
    store.getState().disconnect();
    expect(store.getState()).toMatchObject({ connected: false, socket: null });
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("ignores events from a replaced socket", () => {
    vi.useFakeTimers();
    const { store } = newStore();
    store.getState().connect();
    const first = FakeWebSocket.instances[0];
    store.getState().disconnect();
    store.getState().connect();
    const second = FakeWebSocket.instances[1];
    second.open();

    first.onclose?.();
    first.receive({ type: "error", message: "from stale socket" });
    expect(store.getState()).toMatchObject({ connected: true, socket: second, appError: null });
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("clears transcript state deterministically and blocks loads while running", () => {
    const { store, ws } = connectedStore();
    ws.receive({ type: "final", entry_id: "e-1", text: "keep me", timestamp: 1 });
    store.getState().saveCurrentSession();
    const conversationId = store.getState().history[0].id;
    ws.receive({ type: "partial", text: "pending", source: "mic" });
    store.getState().openSummary();

    store.getState().clearTranscript();
    expect(store.getState()).toMatchObject({
      entries: [],
      partials: {},
      suggestionLoading: {},
      summaryText: "",
      summaryVisible: false,
      activeConversationId: null,
    });

    ws.receive({ type: "status", status: "started" });
    store.getState().loadConversation(conversationId);
    expect(store.getState().entries).toEqual([]);

    ws.receive({ type: "status", status: "stopped" });
    store.getState().setActiveView("history");
    store.getState().loadConversation(conversationId);
    expect(store.getState()).toMatchObject({ activeView: "transcript", activeConversationId: conversationId });
    expect(store.getState().entries.map((e) => e.text)).toEqual(["keep me"]);

    store.getState().renameConversation(conversationId, "Renamed");
    expect(store.getState().history[0].name).toBe("Renamed");
    store.getState().deleteConversation(conversationId);
    expect(store.getState()).toMatchObject({ history: [], activeConversationId: null });
  });

  it("preserves activeConversationId when session starts with existing conversation", () => {
    const { store, ws } = connectedStore();
    store.setState({ activeConversationId: "conv-123" });
    ws.receive({ type: "status", status: "started" });
    expect(store.getState().running).toBe(true);
    expect(store.getState().activeConversationId).toBe("conv-123");
  });

  it("loads conversation from selectedConversationDetail when not found in in-memory history", () => {
    const { store } = newStore();
    store.setState({
      selectedConversationDetail: {
        conversation: { conversation_id: "conv-sqlite-1" },
        messages: [
          { message_id: "msg-1", stt_text: "Hello from SQLite", timestamp_ms: 1000, source: "mic" },
        ],
        summary: { summary_markdown: "Meeting summary from SQLite" },
      },
    });

    store.getState().loadConversation("conv-sqlite-1");
    const state = store.getState();
    expect(state.activeConversationId).toBe("conv-sqlite-1");
    expect(state.activeView).toBe("transcript");
    expect(state.entries).toHaveLength(1);
    expect(state.entries[0]).toMatchObject({
      id: "msg-1",
      text: "Hello from SQLite",
      timestamp: 1,
      source: "mic",
      isFinal: true,
    });
    expect(state.summaryText).toBe("Meeting summary from SQLite");
  });

  it("keeps separate store instances isolated", () => {
    const a = connectedStore();
    const b = newStore();
    a.ws.receive({ type: "final", entry_id: "e-1", text: "only in a", timestamp: 1 });
    expect(b.store.getState().entries).toEqual([]);
    expect(b.store.getState().connected).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Compatibility and v2 settings
// ---------------------------------------------------------------------------

describe("useEngineStore compatibility", () => {
  const STATE_KEYS = [
    "connected", "socket", "running", "isToggling", "entries", "partials",
    "downloading", "downloadModel", "downloadPercent",
    "history", "activeConversationId",
    "activeView", "theme", "appError", "config", "availableMics", "availableSpeakers",
    "suggestionLoading", "summaryText", "summaryLoading", "summaryVisible",
  ];
  const ACTION_KEYS = [
    "connect", "disconnect", "startPipeline", "stopPipeline", "togglePipeline",
    "clearTranscript", "setActiveView", "setTheme", "updateConfig", "requestDeviceList",
    "clearAppError",
    "requestSuggestion", "requestSummary", "openSummary", "closeSummary",
    "renameConversation", "deleteConversation", "saveCurrentSession", "loadConversation",
  ];

  it("exposes every legacy selector and action", () => {
    const state = useEngineStore.getState() as unknown as Record<string, unknown>;
    for (const key of STATE_KEYS) expect(state, key).toHaveProperty(key);
    for (const key of ACTION_KEYS) expect(typeof state[key], key).toBe("function");
    expect(typeof useEngineStore.setState).toBe("function");
    expect(typeof useEngineStore.subscribe).toBe("function");
  });

  it("does not send commands while disconnected", () => {
    const { store } = newStore();
    const s = store.getState();
    s.startPipeline();
    s.stopPipeline();
    s.requestDeviceList();
    s.requestSummary();
    expect(store.getState()).toMatchObject({ isToggling: false, downloading: false });
  });

  it("defaults v2 services to match the engine and resets validation on fingerprinted edits", () => {
    const { store } = newStore();
    const services = store.getState().services;
    expect(services.stt).toMatchObject({ enabled: true, providerId: "local" });
    expect(services.translation).toMatchObject({ enabled: true, providerId: "local" });
    expect(services.assistant).toMatchObject({ enabled: false, providerId: "openai" });

    const valid = { status: "valid" as const, fingerprint: "a".repeat(64), checkedAt: 1, errorCode: null };
    store.getState().setServiceValidation("assistant", valid);
    store.getState().updateServiceConfig("assistant", { enabled: true });
    expect(store.getState().services.assistant.validation).toEqual(valid);

    store.getState().updateServiceConfig("assistant", {
      modelOverride: "gpt-x",
      options: { temperature: 0.2, api_key: SECRET_KEY, Authorization: "Bearer y" },
    });
    const assistant = store.getState().services.assistant;
    expect(assistant.validation.status).toBe("unvalidated");
    expect(assistant.options).toEqual({ temperature: 0.2 });
    expect(JSON.stringify(assistant)).not.toContain(SECRET_KEY);
  });
});

// ---------------------------------------------------------------------------
// Provider manifest types (engine/providers/manifests.py)
// ---------------------------------------------------------------------------

describe("provider manifest types", () => {
  // Mirrors the Python ProtocolFamily enum; the annotation fails tsc if any value is dropped.
  const PYTHON_FAMILIES: ProtocolFamily[] = [
    "local",
    "openai_realtime",
    "gemini_live",
    "openai_compatible_text",
    "gemini_text",
    "deepgram",
    "assemblyai_streaming",
    "azure_speech",
  ];

  it("accepts every Python protocol family", () => {
    expectTypeOf<(typeof PYTHON_FAMILIES)[number]>().toEqualTypeOf<ProtocolFamily>();
    expect([...PROTOCOL_FAMILIES].sort()).toEqual([...PYTHON_FAMILIES].sort());
  });

  it("describes a profiled Custom STT manifest", () => {
    const custom: ProviderManifest = {
      providerId: "custom",
      displayName: "Custom (realtime-compatible)",
      capabilities: ["stt"],
      protocolFamily: "openai_realtime",
      recommendedEndpoint: "",
      recommendedModel: "",
      endpointEditable: true,
      requiresEndpoint: true,
      modelSelectable: true,
      fields: [],
      protocolProfiles: { "openai-realtime": "openai_realtime", "gemini-live": "gemini_live" },
      profileField: "protocol_profile",
    };
    expectTypeOf(custom.protocolProfiles).toEqualTypeOf<
      Record<string, ProtocolFamily> | undefined
    >();
    expect(custom.protocolProfiles?.["gemini-live"]).toBe("gemini_live");
  });
});
