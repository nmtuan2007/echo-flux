import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useEngineStore } from "../store/engineStore";
import { DEFAULT_CONFIG, defaultServices } from "../store/settingsSlice";
import { transcriptResetState } from "../store/transcriptSlice";
import { LiveSessionBar } from "./LiveSessionBar";
import { OverlayPanel } from "./OverlayPanel";
import { ServiceSettingsCard } from "./ServiceSettingsCard";
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
});
