import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Header } from "../components/Header";
import { LiveSessionBar } from "../components/LiveSessionBar";
import { OnboardingDialog } from "../components/OnboardingDialog";
import { OverlayPanel } from "../components/OverlayPanel";
import { TranscriptPanel } from "../components/TranscriptPanel";
import { FORWARDED_COMMAND_TYPES } from "../lib/engineBridge";
import { createEngineStore, useEngineStore } from "../store/engineStore";
import { PROVIDER_CATALOG } from "../store/providerCatalog";
import { DEFAULT_CONFIG, defaultServices } from "../store/settingsSlice";
import { transcriptResetState } from "../store/transcriptSlice";

const invokeMock = vi.hoisted(() => vi.fn());
const emitMock = vi.hoisted(() => vi.fn(async () => undefined));
vi.mock("@tauri-apps/api/tauri", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({ emit: emitMock }));

describe("live translation, defaults, consent, and UI polish", () => {
  beforeEach(() => {
    window.localStorage.clear();
    invokeMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
    emitMock.mockClear();
    useEngineStore.setState({
      ...transcriptResetState(),
      connected: true,
      running: false,
      isToggling: false,
      overlayLocked: false,
      onboardingCompleted: false,
      audioProbeLoading: false,
      audioProbeResult: null,
      appError: null,
      config: { ...DEFAULT_CONFIG },
      services: defaultServices(),
      consent: {
        version: 1,
        accepted: false,
        acceptedAt: null,
        revoked: false,
        revokedAt: null,
        stt: false,
        translation: false,
        assistant: false,
        summary: false,
      },
      hasConsentedToCloud: false,
    });
  });

  it("exposes quality_preset outside Advanced in Local STT catalog", () => {
    const local = PROVIDER_CATALOG.find((p) => p.providerId === "local");
    const preset = local?.fields.find((f) => f.key === "quality_preset");
    expect(preset?.advanced).toBe(false);
  });

  it("defaults audioSource to 'both' and keeps initial consent false until a key is saved", async () => {
    const fresh = createEngineStore();
    expect(fresh.getState().config.audioSource).toBe("both");
    expect(fresh.getState().consent.translation).toBe(false);

    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "credential_status") return { configured: false };
      if (cmd === "credential_save") return { configured: true };
      return undefined;
    });
    useEngineStore.getState().selectServiceProvider("translation", "openai");
    const saved = await useEngineStore
      .getState()
      .saveServiceSecret("translation", "api_key", "sk-test-fake-0001");
    expect(saved).toBe(true);
    expect(useEngineStore.getState().consent.translation).toBe(true);
    expect(useEngineStore.getState().hasConsentedToCloud).toBe(true);
  });

  it("includes translation_configure in FORWARDED_COMMAND_TYPES and dispatches it when running", async () => {
    expect(FORWARDED_COMMAND_TYPES).toContain("translation_configure");

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "translation_configure_result", ok: true }],
    });

    useEngineStore.setState({ running: true });
    await useEngineStore.getState().configureLiveTranslation({
      enabled: true,
      targetLang: "ja",
      sourceLang: "en",
    });

    expect(invokeMock).toHaveBeenCalledWith("engine_forward_command", {
      command: expect.objectContaining({
        type: "translation_configure",
        enabled: true,
        target_lang: "ja",
        source_lang: "en",
      }),
    });
    expect(useEngineStore.getState().config.targetLang).toBe("ja");
    expect(useEngineStore.getState().services.translation.options.target_lang).toBe("ja");
  });

  it("sets appError if translation_configure_result reports ok: false", async () => {
    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "translation_configure_result", ok: false, error: "Pack missing" }],
    });

    useEngineStore.setState({ running: true });
    await useEngineStore.getState().configureLiveTranslation({ targetLang: "ko" });
    expect(useEngineStore.getState().appError).toBe("Pack missing");
  });

  it("renders Header Unlock Overlay tooltip with Ctrl+Shift+U", () => {
    useEngineStore.setState({ overlayLocked: true });
    render(<Header />);
    const unlockBtn = screen.getByRole("button", { name: "Unlock Overlay" });
    expect(unlockBtn.getAttribute("title")).toBe(
      "Unlock Overlay (make interactive) - Ctrl+Shift+U",
    );
  });

  it("renders OnboardingDialog language pair selectors in Step 1 and Ctrl+Shift+U in Step 2", () => {
    render(<OnboardingDialog />);
    const sourceSelect = screen.getByLabelText("Spoken Language") as HTMLSelectElement;
    const targetSelect = screen.getByLabelText("Translate to") as HTMLSelectElement;

    fireEvent.change(sourceSelect, { target: { value: "en" } });
    fireEvent.change(targetSelect, { target: { value: "ja" } });
    expect(useEngineStore.getState().config.sourceLang).toBe("en");
    expect(useEngineStore.getState().config.targetLang).toBe("ja");
    expect(useEngineStore.getState().services.translation.options.target_lang).toBe("ja");

    fireEvent.click(screen.getByRole("button", { name: /Continue|Next/i }));
    expect(screen.getByText("Ctrl+Shift+U")).toBeTruthy();
    expect(screen.getByText(/Unlock click-through Overlay/i)).toBeTruthy();
  });

  it("supports live translation hot-toggle and inline audio signal check in LiveSessionBar", async () => {
    invokeMock.mockResolvedValue({
      messages: [{ type: "translation_configure_result", ok: true }],
    });

    const { rerender } = render(<LiveSessionBar />);

    // Open Devices popover and trigger Check Signal
    fireEvent.click(screen.getByRole("button", { name: /Devices/i }));
    const checkBtn = screen.getByRole("button", { name: "Check Signal" });
    expect(checkBtn).toBeTruthy();

    useEngineStore.setState({
      audioProbeResult: { mic: { ok: true, level: 0.42 } },
    });
    rerender(<LiveSessionBar />);
    expect(screen.getByText("Signal OK (42%)")).toBeTruthy();

    // Hot-toggle translation while running === true
    useEngineStore.setState({ running: true });
    rerender(<LiveSessionBar />);

    const targetSelect = screen.getByLabelText("Target language") as HTMLSelectElement;
    expect(targetSelect.disabled).toBe(false);
    fireEvent.change(targetSelect, { target: { value: "fr" } });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith("engine_forward_command", {
        command: expect.objectContaining({
          type: "translation_configure",
          target_lang: "fr",
        }),
      });
    });

    const toggleCheckbox = screen.getByLabelText("Translate") as HTMLInputElement;
    expect(toggleCheckbox.disabled).toBe(false);
    fireEvent.click(toggleCheckbox);
    await waitFor(() => {
      expect(useEngineStore.getState().config.translationEnabled).toBe(false);
    });
  });

  it("allows hot-toggling translation and renders fallback badge in OverlayPanel while running", () => {
    useEngineStore.setState({ running: true });
    useEngineStore.setState({
      sttStreams: {
        mic: {
          source: "mic",
          provider: "local",
          fallbackFrom: "deepgram",
          continuity: [],
          continuityCount: 0,
        },
      },
    });
    render(<OverlayPanel />);

    const toggleBtn = screen.getByRole("button", { name: "Toggle translation" }) as HTMLButtonElement;
    expect(toggleBtn.disabled).toBe(false);
    const targetSelect = screen.getByLabelText("Target language") as HTMLSelectElement;
    expect(targetSelect.disabled).toBe(false);
    expect(document.querySelector(".overlay-fallback-badge")?.textContent).toBe("Local Fallback");
  });

  it("shows Saving... and disables controls during async TranscriptPanel inline edit", async () => {
    let resolveUpdate: (ok: boolean) => void = () => undefined;
    const updateSpy = vi.fn(
      () => new Promise<boolean>((resolve) => (resolveUpdate = resolve)),
    );
    useEngineStore.setState({
      updateEntry: updateSpy,
      entries: [
        {
          id: "e-save",
          text: "Original line",
          translation: null,
          isFinal: true,
          timestamp: 1000,
          source: "mic",
        },
      ],
    });

    render(<TranscriptPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Edit segment" }));
    const editor = screen.getByLabelText("Edit transcript text") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "Edited line" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    const savingBtn = screen.getByRole("button", { name: "Saving..." }) as HTMLButtonElement;
    const cancelBtn = screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement;
    expect(savingBtn.disabled).toBe(true);
    expect(cancelBtn.disabled).toBe(true);
    expect(editor.disabled).toBe(true);

    await act(async () => {
      resolveUpdate(true);
    });
    expect(screen.queryByRole("button", { name: "Saving..." })).toBeNull();
  });
});
