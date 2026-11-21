import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useEngineStore } from "../store/engineStore";
import { DEFAULT_CONFIG, defaultServices } from "../store/settingsSlice";
import { transcriptResetState } from "../store/transcriptSlice";
import { LiveSessionBar } from "./LiveSessionBar";
import { OverlayPanel } from "./OverlayPanel";
import { ServiceSettingsCard } from "./ServiceSettingsCard";
import { TranscriptPanel } from "./TranscriptPanel";
import { exportAsMarkdown } from "../utils/export";

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
});
