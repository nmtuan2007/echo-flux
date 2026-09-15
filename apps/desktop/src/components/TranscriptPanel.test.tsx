import { act, fireEvent, render, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEngineStore } from "../store/engineStore";
import { DEFAULT_CONFIG, defaultServices } from "../store/settingsSlice";
import { sttProviderLabel, transcriptResetState } from "../store/transcriptSlice";
import { StatusBar } from "./StatusBar";
import { TranscriptPanel } from "./TranscriptPanel";

vi.mock("@tauri-apps/api/tauri", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ emit: vi.fn(async () => undefined) }));

// Mocked engine messages only: a fake socket stands in for the engine and no provider is
// contacted.

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

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
    act(() => {
      this.onmessage?.({ data: JSON.stringify(message) });
    });
  }

  sentMessages(): Record<string, unknown>[] {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const CONVERSATION = "conv-1";
const CLOUD = "openai-realtime";

function connect(overrides: Partial<typeof DEFAULT_CONFIG> = {}): FakeWebSocket {
  useEngineStore.setState({
    ...transcriptResetState(),
    running: false,
    history: [],
    activeConversationId: null,
    config: { ...DEFAULT_CONFIG, llmEnabled: true, ...overrides },
    services: defaultServices(),
  });
  useEngineStore.getState().connect();
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  act(() => ws.open());
  ws.receive({ type: "status", status: "started" });
  return ws;
}

function partial(ws: FakeWebSocket, source: string, text: string, item: string, revision: number, extra = {}) {
  ws.receive({
    type: "partial",
    text,
    translation: null,
    is_final: false,
    timestamp: 1,
    source,
    item_id: item,
    revision,
    stt_provider: "local",
    ...extra,
  });
}

function final(ws: FakeWebSocket, source: string, text: string, item: string, entryId: string, extra = {}) {
  ws.receive({
    type: "final",
    entry_id: entryId,
    text,
    translation: null,
    is_final: true,
    timestamp: 2,
    source,
    item_id: item,
    revision: 9,
    stt_provider: "local",
    ...extra,
  });
}

function partialNode(stream: string): HTMLElement | null {
  return document.querySelector(`[data-stream="${stream}"]`);
}

function entryNode(entryId: string): HTMLElement {
  const node = document.querySelector(`[data-entry-id="${entryId}"]`);
  if (!node) throw new Error(`entry ${entryId} not rendered`);
  return node as HTMLElement;
}

beforeEach(() => {
  window.localStorage.clear();
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  act(() => useEngineStore.getState().disconnect());
  vi.unstubAllGlobals();
});

describe("revision-safe transcript rendering", () => {
  it("keeps rapid alternating mic/system partials on their own streams without remounting", () => {
    const ws = connect();
    render(<TranscriptPanel />);

    partial(ws, "system", "they", "system:1", 1);
    partial(ws, "mic", "I", "mic:1", 1);
    const systemNode = partialNode("system");
    const micNode = partialNode("mic");

    for (let i = 2; i <= 30; i++) {
      partial(ws, i % 2 ? "mic" : "system", `${i % 2 ? "mic" : "system"} rev ${i}`, i % 2 ? "mic:1" : "system:1", i);
    }

    // Same DOM nodes (keyed by stream), fixed order, and no text crossed streams.
    expect(partialNode("system")).toBe(systemNode);
    expect(partialNode("mic")).toBe(micNode);
    const order = Array.from(document.querySelectorAll("[data-stream]")).map((n) => n.getAttribute("data-stream"));
    expect(order).toEqual(["mic", "system"]);
    expect(within(micNode!).getByText(/mic rev 29/)).toBeTruthy();
    expect(within(systemNode!).getByText(/system rev 30/)).toBeTruthy();

    // Stale revision is ignored; a newer revision may legitimately shorten the text.
    partial(ws, "mic", "stale", "mic:1", 3);
    expect(micNode!.textContent).toContain("mic rev 29");
    partial(ws, "mic", "Hi", "mic:1", 31);
    expect(micNode!.textContent).toContain("Hi");
    expect(systemNode!.textContent).toContain("system rev 30");
  });

  it("does not let a final or late partial for one item clobber another", () => {
    const ws = connect();
    render(<TranscriptPanel />);

    partial(ws, "mic", "first utterance", "mic:1", 1);
    final(ws, "mic", "first utterance.", "mic:1", "e-1");
    expect(partialNode("mic")).toBeNull();
    expect(entryNode("e-1").textContent).toContain("first utterance.");

    // A late partial for the finalized item cannot reopen it.
    partial(ws, "mic", "first utter", "mic:1", 2);
    expect(partialNode("mic")).toBeNull();

    // A retraction for an older item leaves the newer partial alone.
    partial(ws, "mic", "second", "mic:2", 1);
    ws.receive({ type: "final", text: "", is_final: true, timestamp: 3, source: "mic", item_id: "mic:0", revision: 4, retracted: true });
    expect(partialNode("mic")!.textContent).toContain("second");

    // Redelivered final for a committed item does not duplicate the entry.
    final(ws, "mic", "first utterance.", "mic:1", "e-1b");
    expect(useEngineStore.getState().entries.map((e) => e.id)).toEqual(["e-1"]);

    // System stream is independent.
    partial(ws, "system", "other side", "system:1", 1);
    final(ws, "mic", "second.", "mic:2", "e-2");
    expect(partialNode("system")!.textContent).toContain("other side");
  });

  it("treats a final translation as authoritative over drafts and stale revisions", () => {
    const ws = connect();
    render(<TranscriptPanel />);
    final(ws, "system", "hello", "system:1", "e-1");
    // Translation revisions never fall below the transcript revision (9 from final()).

    ws.receive({ type: "translation_update", entry_id: "e-1", source_text: "hello", translation: "draft", is_final: false, revision: 9 });
    expect(entryNode("e-1").querySelector("[data-translation-state='draft']")?.textContent).toBe("draft");

    ws.receive({ type: "translation_update", entry_id: "e-1", source_text: "hello", translation: "xin chào", is_final: true, revision: 10 });
    ws.receive({ type: "translation_update", entry_id: "e-1", source_text: "hello", translation: "late draft", is_final: false, revision: 11 });
    ws.receive({ type: "translation_update", entry_id: "e-1", source_text: "hello", translation: "old final", is_final: true, revision: 9 });

    const node = entryNode("e-1").querySelector(".transcript-translation");
    expect(node?.textContent).toBe("xin chào");
    expect(node?.getAttribute("data-translation-state")).toBe("final");

    // Legacy translation_update without revision still applies.
    final(ws, "mic", "legacy", "mic:1", "e-2");
    ws.receive({ type: "translation_update", entry_id: "e-2", source_text: "legacy", translation: "cũ", is_final: true });
    expect(entryNode("e-2").textContent).toContain("cũ");
  });
});

describe("provider fallback and continuity", () => {
  it("shows sticky Local fallback and continuity, and keeps them after Stop", () => {
    const ws = connect({ language: "auto", targetLang: "vi", modelSize: "large-v3" });
    render(
      <>
        <TranscriptPanel />
        <StatusBar />
      </>,
    );

    partial(ws, "system", "cloud words", "system:1:openai", 1, { stt_provider: CLOUD });
    expect(document.querySelector(".stt-notice")).toBeNull();

    // The abandoned cloud item is retracted, then Local takes over.
    ws.receive({ type: "final", text: "", is_final: true, timestamp: 3, source: "system", item_id: "system:1:openai", revision: 2, retracted: true, stt_provider: CLOUD });
    partial(ws, "system", "local words", "system:2:local", 1);
    const notice = document.querySelector(".stt-notice") as HTMLElement;
    expect(notice.textContent).toContain(`Others: switched to Local from ${sttProviderLabel(CLOUD)}`);

    // A later cloud-tagged event cannot move the stream off Local.
    partial(ws, "system", "local words again", "system:2:local", 2, { stt_provider: CLOUD });
    expect(useEngineStore.getState().sttStreams.system).toMatchObject({ provider: "local", fallbackFrom: CLOUD });

    final(ws, "system", "local words again.", "system:2:local", "e-1", {
      continuity: { reason: "audio_gap", start_ms: 12300, end_ms: 14000, epoch: 2 },
    });
    expect(notice.textContent).toContain("Continuity uncertain in 1 interval (latest 0:12.3–0:14.0)");
    expect(entryNode("e-1").textContent).toContain("Continuity uncertain 0:12.3–0:14.0");

    const status = document.querySelector(".status-bar") as HTMLElement;
    expect(status.textContent).toContain("Auto → VI");
    expect(status.textContent).toContain("(fallback)");
    expect(status.textContent).toContain("1 gap");
    expect(status.textContent).not.toContain("large-v3");

    ws.receive({ type: "status", status: "stopped" });
    expect(document.querySelector(".stt-notice")?.textContent).toContain("switched to Local");
    expect(useEngineStore.getState().history[0].entries[0]).toMatchObject({ sttFallbackFrom: CLOUD });

    // A new session starts clean.
    ws.receive({ type: "status", status: "started" });
    expect(document.querySelector(".stt-notice")).toBeNull();
  });
});

describe("manual Assist", () => {
  const replies = [
    { strategy: "Cooperative", text: "Sounds good, let's do Friday.", reasoning: "Agrees" },
    { strategy: "Assertive", text: "Friday does not work for us." },
  ];
  const questions = [{ question: "Which Friday do you mean?", purpose: "Date" }];

  function assistButton(entryId: string): HTMLButtonElement | null {
    return within(entryNode(entryId)).queryByRole("button", { name: "Assist" }) as HTMLButtonElement | null;
  }

  function requestsSent(ws: FakeWebSocket) {
    return ws.sentMessages().filter((m) => m.type === "request_suggestion");
  }

  it("is hidden without an Assistant and never offered on partials or sent automatically", () => {
    const ws = connect({ llmEnabled: false });
    render(<TranscriptPanel />);
    final(ws, "system", "Can we meet Friday?", "system:1", "e-1", { conversation_id: CONVERSATION });
    expect(assistButton("e-1")).toBeNull();

    act(() => useEngineStore.getState().updateConfig({ llmEnabled: true }));
    partial(ws, "mic", "typing", "mic:1", 1);
    expect(within(partialNode("mic")!).queryByRole("button")).toBeNull();
    expect(assistButton("e-1")).not.toBeNull();
    final(ws, "system", "Another one", "system:2", "e-2", { conversation_id: CONVERSATION });
    expect(requestsSent(ws)).toHaveLength(0);
  });

  it("stays unavailable with a reason when the engine has not provided a conversation id", () => {
    const ws = connect();
    render(<TranscriptPanel />);
    final(ws, "system", "Can we meet Friday?", "system:1", "e-1");

    const button = assistButton("e-1")!;
    expect(button.disabled).toBe(true);
    expect(entryNode("e-1").textContent).toContain("not saved yet");
    fireEvent.click(button);
    act(() => useEngineStore.getState().requestAssist("e-1"));
    expect(requestsSent(ws)).toHaveLength(0);
    expect(useEngineStore.getState().suggestionLoading).toEqual({});
  });

  it("sends only ids, ignores stale replies, and renders replies and questions separately", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    const ws = connect();
    render(<TranscriptPanel />);
    final(ws, "mic", "I can do Friday", "mic:1", "e-0", { conversation_id: CONVERSATION });
    final(ws, "system", "Can we meet Friday?", "system:1", "e-1", { conversation_id: CONVERSATION });

    fireEvent.click(assistButton("e-1")!);
    expect(requestsSent(ws)).toEqual([
      { type: "request_suggestion", conversation_id: CONVERSATION, message_id: "e-1", entry_id: "e-1" },
    ]);
    expect(within(entryNode("e-1")).getByRole("status").textContent).toContain("thinking");
    expect(assistButton("e-1")).toBeNull();
    act(() => useEngineStore.getState().requestAssist("e-1"));
    expect(requestsSent(ws)).toHaveLength(1);
    expect(useEngineStore.getState().assistRequests["e-1"].requestId).toMatch(/^assist-\d+$/);

    // Reply for another conversation, and a reply for a message nobody asked about.
    ws.receive({ type: "suggestion_result", entry_id: "e-1", conversation_id: "conv-other", message_id: "e-1", assist_id: "a-x", options: [], suggested_replies: [{ strategy: "Cooperative", text: "wrong" }], clarifying_questions: [] });
    ws.receive({ type: "suggestion_result", entry_id: "e-0", conversation_id: CONVERSATION, message_id: "e-0", assist_id: "a-y", options: [], suggested_replies: [{ strategy: "Cooperative", text: "unrequested" }], clarifying_questions: [] });
    expect(useEngineStore.getState().suggestionLoading).toEqual({ "e-1": true });
    expect(document.body.textContent).not.toContain("wrong");
    expect(document.body.textContent).not.toContain("unrequested");

    ws.receive({
      type: "suggestion_result",
      entry_id: "e-1",
      options: replies.map(({ strategy, text }) => ({ strategy, text })),
      conversation_id: CONVERSATION,
      message_id: "e-1",
      assist_id: "a-1",
      suggested_replies: replies,
      clarifying_questions: questions,
      provider: { provider_id: "openai" },
      context_window: { message_count: 1 },
    });

    const node = entryNode("e-1");
    const replyGroup = within(node).getByRole("group", { name: "Suggested replies" });
    const questionGroup = within(node).getByRole("group", { name: "Clarifying questions" });
    expect(replyGroup.textContent).toContain("Sounds good, let's do Friday.");
    expect(replyGroup.textContent).toContain("Friday does not work for us.");
    expect(replyGroup.textContent).not.toContain("Which Friday");
    expect(questionGroup.textContent).toContain("Which Friday do you mean?");
    expect(useEngineStore.getState().entries[1].assist).toMatchObject({ requestId: "assist-1", assistId: "a-1", providerId: "openai" });

    await act(async () => {
      fireEvent.click(within(questionGroup).getByRole("button", { name: /Copy question/ }));
    });
    expect(writeText).toHaveBeenCalledWith("Which Friday do you mean?");

    // A redelivered reply after completion is ignored.
    ws.receive({ type: "suggestion_result", entry_id: "e-1", conversation_id: CONVERSATION, message_id: "e-1", assist_id: "a-2", options: [], suggested_replies: [], clarifying_questions: [] });
    expect(within(entryNode("e-1")).getByRole("group", { name: "Suggested replies" }).textContent).toContain("Sounds good");

    // Nothing but ids ever left for Assist.
    for (const message of requestsSent(ws)) {
      expect(Object.keys(message).sort()).toEqual(["conversation_id", "entry_id", "message_id", "type"]);
    }
  });

  it("renders provider, structured, empty, and interrupted states with retry where allowed", () => {
    const ws = connect();
    render(<TranscriptPanel />);
    final(ws, "system", "Question one?", "system:1", "e-1", { conversation_id: CONVERSATION });
    final(ws, "system", "Question two?", "system:2", "e-2", { conversation_id: CONVERSATION });

    fireEvent.click(assistButton("e-1")!);
    const firstRequestId = useEngineStore.getState().assistRequests["e-1"].requestId;
    ws.receive({ type: "suggestion_result", entry_id: "e-1", conversation_id: CONVERSATION, message_id: "e-1", error: "The Assistant provider could not complete the request.", error_code: "provider_error", retryable: true, provider_code: "rate_limited" });
    let alert = within(entryNode("e-1")).getByRole("alert");
    expect(alert.getAttribute("data-kind")).toBe("provider");
    expect(alert.textContent).toContain("rate limiting");

    fireEvent.click(within(alert).getByRole("button", { name: "Retry" }));
    expect(requestsSent(ws)).toHaveLength(2);
    expect(useEngineStore.getState().assistRequests["e-1"].requestId).not.toBe(firstRequestId);
    ws.receive({ type: "suggestion_result", entry_id: "e-1", conversation_id: CONVERSATION, message_id: "e-1", assist_id: "a-3", options: [], suggested_replies: [], clarifying_questions: [] });
    expect(entryNode("e-1").textContent).toContain("No suggestions for this message.");

    fireEvent.click(assistButton("e-2")!);
    ws.receive({ type: "suggestion_result", entry_id: "e-2", conversation_id: CONVERSATION, message_id: "e-2", error: "The message to assist with was not found.", error_code: "target_not_found", retryable: false });
    alert = within(entryNode("e-2")).getByRole("alert");
    expect(alert.getAttribute("data-kind")).toBe("structured");
    expect(within(alert).queryByRole("button", { name: "Retry" })).toBeNull();

    // The bridge ends a request without its reply: interrupted, retryable, late reply ignored.
    act(() => useEngineStore.setState({ entries: useEngineStore.getState().entries.map((e) => (e.id === "e-2" ? { ...e, assist: undefined, suggestions: undefined } : e)) }));
    fireEvent.click(assistButton("e-2")!);
    act(() => useEngineStore.setState({ suggestionLoading: {} }));
    alert = within(entryNode("e-2")).getByRole("alert");
    expect(alert.getAttribute("data-kind")).toBe("interrupted");
    ws.receive({ type: "suggestion_result", entry_id: "e-2", conversation_id: CONVERSATION, message_id: "e-2", assist_id: "late", options: [], suggested_replies: [{ strategy: "x", text: "late reply" }], clarifying_questions: [] });
    expect(document.body.textContent).not.toContain("late reply");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeTruthy();

    // A pending request does not survive a new session.
    fireEvent.click(within(alert).getByRole("button", { name: "Retry" }));
    ws.receive({ type: "status", status: "stopped" });
    ws.receive({ type: "status", status: "started" });
    expect(useEngineStore.getState().assistRequests).toEqual({});
  });

  it("still renders legacy shortcut suggestions as suggested replies", async () => {
    const ws = connect();
    render(<TranscriptPanel />);
    final(ws, "system", "Legacy question?", "system:1", "e-1");
    await act(async () => {
      await useEngineStore.getState().requestSuggestion("e-1", "Legacy question?", ["Legacy question?"]);
    });
    ws.receive({ type: "suggestion_result", entry_id: "e-1", options: [{ strategy: "Clarifying", text: "Legacy reply" }] });
    expect(useEngineStore.getState().entries[0].suggestions?.options?.[0]?.text).toBe("Legacy reply");
    expect(within(entryNode("e-1")).getByRole("group", { name: "Suggested replies" }).textContent).toContain("Legacy reply");
  });
});
