import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createEngineStore, useEngineStore } from "../store/engineStore";
import { DEFAULT_CONFIG, defaultServices } from "../store/settingsSlice";
import { transcriptResetState } from "../store/transcriptSlice";
import { LiveSessionBar } from "./LiveSessionBar";
import { OverlayPanel } from "./OverlayPanel";
import { ServiceSettingsCard } from "./ServiceSettingsCard";
import { SummaryModal } from "./SummaryModal";
import { TranscriptPanel } from "./TranscriptPanel";
import { CloudConsent } from "./CloudConsent";
import { ConversationSetupPanel } from "./ConversationSetupPanel";
import { ExportMenu } from "./ExportMenu";
import { exportAsMarkdown, exportAsSRT } from "../utils/export";

vi.mock("@tauri-apps/api/tauri", () => ({ invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/event", () => ({ emit: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/globalShortcut", () => ({
  isRegistered: vi.fn(async () => false),
  register: vi.fn(async () => undefined),
  unregister: vi.fn(async () => undefined),
}));

describe("Phase 1-4 UI enhancements", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useEngineStore.setState({
      ...transcriptResetState(),
      connected: true,
      running: false,
      isToggling: false,
      config: { ...DEFAULT_CONFIG },
      services: defaultServices(),
      sourceLabels: { mic: "Tuan", system: "Partner" },
      pendingTitle: "Roadmap Sync",
      attachedProfileIds: ["p1"],
      hasConsentedToCloud: false,
      summaryLoading: false,
      summaryText: "",
      summaryVisible: false,
    });
  });

  it("allows toggling translation on/off in ServiceSettingsCard", () => {
    render(<ServiceSettingsCard service="translation" />);
    const checkbox = screen.getByLabelText("Enable Translation") as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    fireEvent.click(checkbox);
    expect(useEngineStore.getState().services.translation.enabled).toBe(false);
  });

  it("renders custom speaker labels and filters transcript entries", () => {
    useEngineStore.setState({
      entries: [
        {
          id: "e-1",
          text: "Hello from me",
          translation: "Xin chào",
          isFinal: true,
          timestamp: 1000,
          source: "mic",
        },
        {
          id: "e-2",
          text: "Budget update",
          translation: "Cập nhật ngân sách",
          isFinal: true,
          timestamp: 1005,
          source: "system",
        },
      ],
    });

    render(<TranscriptPanel />);
    expect(screen.getAllByText(/Tuan/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Partner/).length).toBeGreaterThan(0);

    const search = screen.getByRole("searchbox", { name: /Search transcript/i });
    fireEvent.change(search, { target: { value: "Budget" } });
    expect(screen.queryByText("Hello from me")).toBeNull();
    expect(screen.getByText("Budget update")).toBeTruthy();
  });

  it("updates target language from LiveSessionBar", () => {
    render(<LiveSessionBar />);
    const targetSelect = screen.getByLabelText("Target language") as HTMLSelectElement;
    fireEvent.change(targetSelect, { target: { value: "ja" } });
    expect(useEngineStore.getState().services.translation.options.target_lang).toBe("ja");
  });

  it("renders lightweight caption mode in OverlayPanel without Assist buttons", () => {
    useEngineStore.setState({
      entries: [
        {
          id: "e-1",
          text: "Overlay caption line",
          translation: "Phụ đề",
          isFinal: true,
          timestamp: 1000,
          source: "mic",
        },
      ],
    });

    render(<OverlayPanel />);
    expect(screen.getByText("Overlay caption line")).toBeTruthy();
    expect(screen.queryByText("Assist")).toBeNull();
  });

  it("formats markdown exports with title, summary, and speaker labels", () => {
    const md = exportAsMarkdown(
      [{ text: "Hi", translation: "Chào", timestamp: 0, speaker: "Tuan" }],
      "Weekly Sync",
      "- Key decision",
    );
    expect(md).toContain("# Weekly Sync");
    expect(md).toContain("## Summary");
    expect(md).toContain("**Tuan** Hi");
  });

  it("formats SRT exports with relative elapsed cue timestamps starting at 00:00:00,000", () => {
    const srt = exportAsSRT([
      { text: "Hello world", translation: null, timestamp: 1_700_000_000, speaker: "Tuan" },
      { text: "Second line", translation: "Dòng hai", timestamp: 1_700_000_005.5, speaker: "Partner" },
    ]);
    expect(srt).toContain("1\n00:00:00,000 --> 00:00:01,000\nTuan: Hello world");
    expect(srt).toContain("2\n00:00:05,500 --> 00:00:06,500\nPartner: Second line\nDòng hai");
  });

  it("enforces granular cloud consent per capability in CloudConsent and ServiceSettingsCard", () => {
    useEngineStore.getState().acceptCloudConsent({ stt: true, translation: false, assistant: false, summary: false });
    useEngineStore.getState().selectServiceProvider("translation", "openai");

    render(<ServiceSettingsCard service="translation" />);
    expect(screen.getByText(/Cloud permission is not granted for Translation/i)).toBeTruthy();

    render(<CloudConsent mode="settings" />);
    const translationToggle = screen.getByLabelText(/Translation \(Transcript Streaming\)/i) as HTMLInputElement;
    expect(translationToggle.checked).toBe(false);
    fireEvent.click(translationToggle);
    const saveBtn = screen.getByRole("button", { name: /Save Permissions/i });
    fireEvent.click(saveBtn);
    expect(useEngineStore.getState().consent.translation).toBe(true);
  });

  it("preserves pendingTitle in ConversationSetupPanel and supports Save Setup without starting", () => {
    const onClose = vi.fn();
    const onStart = vi.fn();
    useEngineStore.setState({ pendingTitle: "Architecture Sync" });
    useEngineStore.getState().openSetup();
    expect(useEngineStore.getState().pendingTitle).toBe("Architecture Sync");

    render(<ConversationSetupPanel isOpen={true} onClose={onClose} onStart={onStart} />);
    const titleInput = screen.getByPlaceholderText(/Q4 Strategy Review/i) as HTMLInputElement;
    expect(titleInput.value).toBe("Architecture Sync");
    fireEvent.change(titleInput, { target: { value: "Updated Sync" } });

    fireEvent.click(screen.getByRole("button", { name: /Save Setup/i }));
    expect(useEngineStore.getState().pendingTitle).toBe("Updated Sync");
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onStart).not.toHaveBeenCalled();
  });

  it("shows quick Assist controls in OverlayPanel when Assistant is enabled", () => {
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    useEngineStore.setState({
      engineConversationId: "c-overlay",
      entries: [
        {
          id: "e-1",
          messageId: "e-1",
          conversationId: "c-overlay",
          text: "Can we deploy tomorrow?",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "system",
          assist: {
            requestId: "r-1",
            assistId: "a-1",
            conversationId: "c-overlay",
            suggestedReplies: [{ strategy: "Cooperative", text: "Yes, after QA sign-off." }],
            clarifyingQuestions: [],
            providerId: "openai",
            error: null,
          },
        },
      ],
    });

    render(<OverlayPanel />);
    expect(screen.getByRole("button", { name: /Assist/i })).toBeTruthy();
    expect(screen.getByText("Yes, after QA sign-off.")).toBeTruthy();
  });

  it("renders unified ExportMenu options", () => {
    render(
      <ExportMenu
        entries={[{ text: "Sample line", translation: null, timestamp: 10, speaker: "Me" }]}
        title="Demo"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Export/i }));
    expect(screen.getByRole("menuitem", { name: /Export Subtitles \(\.srt\)/i })).toBeTruthy();
    expect(screen.getByRole("menuitem", { name: /Export Markdown \(\.md\)/i })).toBeTruthy();
  });

  it("persists granular per-capability cloud consent across store reloads", () => {
    useEngineStore.getState().acceptCloudConsent({
      stt: true,
      translation: false,
      assistant: true,
      summary: false,
    });
    const reloaded = createEngineStore();
    expect(reloaded.getState().consent).toMatchObject({
      accepted: true,
      stt: true,
      translation: false,
      assistant: true,
      summary: false,
    });
  });

  it("clears archived history conversation id when starting a new recording", () => {
    useEngineStore.setState({
      history: [
        {
          id: "hist-1",
          name: "Old Meeting",
          date: 1000,
          entries: [{ id: "m1", text: "Old text", translation: null, isFinal: true, timestamp: 1 }],
        },
      ],
    });
    useEngineStore.getState().loadConversation("hist-1");
    expect(useEngineStore.getState().isViewingHistory).toBe(true);
    expect(useEngineStore.getState().activeConversationId).toBe("hist-1");

    const sent: string[] = [];
    useEngineStore.setState({
      connected: true,
      socket: { readyState: 1, send: (d: string) => sent.push(d) } as unknown as WebSocket,
    });
    useEngineStore.getState().startPipeline();
    expect(useEngineStore.getState().isViewingHistory).toBe(false);
    expect(useEngineStore.getState().activeConversationId).toBeNull();
    expect(sent).toHaveLength(1);
  });

  it("renders Start/Stop control and error banner inside OverlayPanel", () => {
    useEngineStore.setState({ connected: true, running: false, appError: "Mic disconnected" });
    render(<OverlayPanel />);
    expect(screen.getByRole("button", { name: /Start/i })).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("Mic disconnected");
  });

  it("uses exact startMs/endMs spans in exportAsSRT when available", () => {
    const srt = exportAsSRT([
      {
        text: "Exact timing",
        translation: null,
        timestamp: 0,
        startMs: 1000,
        endMs: 3250,
      },
      {
        text: "Second cue",
        translation: null,
        timestamp: 0,
        startMs: 4500,
        endMs: 6000,
      },
    ]);
    expect(srt).toContain("1\n00:00:00,000 --> 00:00:02,250\nExact timing");
    expect(srt).toContain("2\n00:00:03,500 --> 00:00:05,000\nSecond cue");
  });

  it("resets previous quick-session conversation ids when starting a new session without setup", () => {
    const sent: string[] = [];
    useEngineStore.setState({
      connected: true,
      running: false,
      activeConversationId: "conv-old",
      engineConversationId: "conv-old",
      preparedConversationId: null,
      socket: { readyState: 1, send: (d: string) => sent.push(d) } as unknown as WebSocket,
    });

    useEngineStore.getState().startPipeline();
    expect(useEngineStore.getState().activeConversationId).toBeNull();
    expect(useEngineStore.getState().engineConversationId).toBeNull();
    expect(sent).toHaveLength(1);
  });

  it("shows listening state in TranscriptPanel when running with no entries yet", () => {
    useEngineStore.setState({ running: true, entries: [], partials: {} });
    render(<TranscriptPanel />);
    expect(screen.getByText("Listening for speech...")).toBeTruthy();
    expect(screen.queryByText(/Press Start/i)).toBeNull();
  });

  it("supports inline segment editing, speaker switching, and spoken language selection", async () => {
    useEngineStore.setState({
      entries: [
        {
          id: "e-edit",
          text: "Initial ASR text",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "mic",
        },
      ],
    });

    render(
      <>
        <LiveSessionBar />
        <TranscriptPanel />
      </>,
    );

    const spokenSelect = screen.getByLabelText("Spoken language") as HTMLSelectElement;
    fireEvent.change(spokenSelect, { target: { value: "vi" } });
    expect(useEngineStore.getState().services.stt.options.language).toBe("vi");
    expect(useEngineStore.getState().config.language).toBe("vi");

    fireEvent.click(screen.getByRole("button", { name: "Edit segment" }));
    const editor = screen.getByLabelText("Edit transcript text") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "Corrected transcript text" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => {
      expect(useEngineStore.getState().entries[0].text).toBe("Corrected transcript text");
    });

    fireEvent.click(screen.getByRole("button", { name: /Switch speaker from Tuan/i }));
    await vi.waitFor(() => {
      expect(useEngineStore.getState().entries[0].source).toBe("system");
    });
  });

  it("renders clarifying questions and retry action inside OverlayPanel Quick Assist", () => {
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    useEngineStore.setState({
      engineConversationId: "c-overlay-q",
      entries: [
        {
          id: "e-q1",
          messageId: "e-q1",
          conversationId: "c-overlay-q",
          text: "Let's finalize the budget.",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "system",
          assist: {
            requestId: "r-q1",
            assistId: "a-q1",
            conversationId: "c-overlay-q",
            suggestedReplies: [{ strategy: "Direct", text: "Looks good." }],
            clarifyingQuestions: [{ question: "What is the Q4 cap?" }],
            providerId: "openai",
            error: null,
          },
        },
      ],
    });

    const { rerender } = render(<OverlayPanel />);
    expect(screen.getByText("What is the Q4 cap?")).toBeTruthy();

    useEngineStore.setState({
      entries: [
        {
          id: "e-q2",
          messageId: "e-q2",
          conversationId: "c-overlay-q",
          text: "Can you confirm?",
          translation: null,
          isFinal: true,
          timestamp: 1001,
          source: "system",
          assist: {
            requestId: "r-q2",
            assistId: null,
            conversationId: "c-overlay-q",
            suggestedReplies: [],
            clarifyingQuestions: [],
            providerId: "openai",
            error: {
              kind: "provider",
              code: "provider_error",
              providerCode: "timeout",
              message: "Timed out.",
              retryable: true,
            },
          },
        },
      ],
    });
    rerender(<OverlayPanel />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });

  it("hides SummaryModal without cancelling an active streaming summary", () => {
    const cancelSpy = vi.fn(async () => undefined);
    useEngineStore.setState({
      summaryVisible: true,
      summaryLoading: true,
      summaryText: "Partial summary...",
      cancelSummary: cancelSpy,
    });

    useEngineStore.getState().closeSummary();
    expect(useEngineStore.getState().summaryVisible).toBe(false);
    expect(useEngineStore.getState().summaryLoading).toBe(true);
    expect(cancelSpy).not.toHaveBeenCalled();
  });

  it("updates translation immediately when message_update returns a retranslation", async () => {
    (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__ = () => undefined;
    const tauri = await import("@tauri-apps/api/tauri");
    vi.mocked(tauri.invoke).mockResolvedValueOnce({
      messages: [{ text: "Edited English", translation: "Tiếng Việt mới" }],
    });

    useEngineStore.setState({
      engineConversationId: "conv-retrans",
      entries: [
        {
          id: "m-1",
          messageId: "m-1",
          conversationId: "conv-retrans",
          text: "Old English",
          translation: "Bản dịch cũ",
          isFinal: true,
          timestamp: 1000,
          source: "system",
        },
      ],
    });

    const ok = await useEngineStore.getState().updateEntry("m-1", { text: "Edited English" });
    expect(ok).toBe(true);
    expect(useEngineStore.getState().entries[0].translation).toBe("Tiếng Việt mới");
    delete (window as unknown as { __TAURI_IPC__?: unknown }).__TAURI_IPC__;
  });

  it("persists sourceLabels and attachedProfileIds across store reloads", () => {
    useEngineStore.getState().setSourceLabels({ mic: "Alice", system: "Client" });
    useEngineStore.getState().setAttachedProfileIds(["prof-a", "prof-b"]);

    const reloaded = createEngineStore();
    expect(reloaded.getState().sourceLabels).toEqual({ mic: "Alice", system: "Client" });
    expect(reloaded.getState().attachedProfileIds).toEqual(["prof-a", "prof-b"]);
  });

  it("renders consolidated Summary & Ask button and places entry timestamp inside actions", () => {
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    useEngineStore.setState({
      engineConversationId: "conv-ui",
      entries: [
        {
          id: "e-time",
          text: "Segment with non-overlapping timestamp",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "mic",
        },
      ],
    });

    const { container } = render(
      <>
        <LiveSessionBar />
        <TranscriptPanel />
      </>,
    );

    expect(screen.getByRole("button", { name: /Summary & Ask/i })).toBeTruthy();
    const actionsEl = container.querySelector(".transcript-entry-actions");
    expect(actionsEl?.querySelector(".transcript-time")).toBeTruthy();
  });

  it("shows silence warning hint in TranscriptPanel after 15s without audio signal", () => {
    vi.useFakeTimers();
    try {
      useEngineStore.setState({
        connected: true,
        running: true,
        entries: [],
        partials: {},
        audioLevels: {},
      });
      render(<TranscriptPanel />);
      expect(screen.queryByText(/No audio signal detected yet/i)).toBeNull();
      act(() => {
        vi.advanceTimersByTime(15100);
      });
      expect(screen.getByText(/No audio signal detected yet/i)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it("supports continueSession mode when loading a conversation from History", () => {
    useEngineStore.setState({
      history: [
        {
          id: "hist-cont",
          name: "Part 1",
          date: 1000,
          entries: [{ id: "m1", text: "Old text", translation: null, isFinal: true, timestamp: 1 }],
        },
      ],
    });
    useEngineStore.getState().loadConversation("hist-cont", { continueSession: true });
    expect(useEngineStore.getState().isViewingHistory).toBe(false);
    expect(useEngineStore.getState().preparedConversationId).toBe("hist-cont");

    const sent: string[] = [];
    useEngineStore.setState({
      connected: true,
      socket: { readyState: 1, send: (d: string) => sent.push(d) } as unknown as WebSocket,
    });
    useEngineStore.getState().startPipeline();
    expect(useEngineStore.getState().activeConversationId).toBe("hist-cont");
    expect(sent).toHaveLength(1);
  });

  it("disables Overlay Lock button when global shortcuts are disabled", () => {
    useEngineStore.setState({ shortcutsEnabled: false });
    render(<OverlayPanel />);
    const lockBtn = screen.getByRole("button", { name: /^Lock$/i }) as HTMLButtonElement;
    expect(lockBtn.disabled).toBe(true);
  });

  it("renders translation fallback and translation error notices in TranscriptPanel", () => {
    useEngineStore.setState({
      config: { ...DEFAULT_CONFIG, translationEnabled: true },
      entries: [
        {
          id: "e-tr-err",
          text: "Bonjour tout le monde",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "system",
          translationFallback: true,
          translationError: { code: "rate_limit", message: "Translation quota exceeded." },
        },
      ],
    });
    render(<TranscriptPanel />);
    expect(screen.getByText(/Translation switched to Local fallback/i)).toBeTruthy();
    expect(screen.getByText(/Translation quota exceeded/i)).toBeTruthy();
  });

  it("supports multi-turn Quick Ask history in SummaryModal", async () => {
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    useEngineStore.setState({
      summaryVisible: true,
      engineConversationId: "conv-ask",
      askHistory: [
        { id: "t1", question: "What was decided?", answer: "Ship v2 on Friday." },
        { id: "t2", question: "Who owns QA?", answer: "Linh." },
      ],
    });
    render(<SummaryModal />);
    expect(screen.getByText("What was decided?")).toBeTruthy();
    expect(screen.getByText("Ship v2 on Friday.")).toBeTruthy();
    expect(screen.getByText("Who owns QA?")).toBeTruthy();
    expect(screen.getByText("Linh.")).toBeTruthy();
  });
});
