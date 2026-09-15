import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { selectPersistedState, useEngineStore, type EngineState } from "../store/engineStore";
import {
  defaultServices,
  initialServiceUiState,
  PROVIDER_CATALOG,
  providersFor,
} from "../store/settingsSlice";
import type { ServiceName } from "../store/types";
import { ServiceSettingsCard } from "./ServiceSettingsCard";
import { SettingsPanel } from "./SettingsPanel";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/tauri", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({ emit: vi.fn(async () => undefined) }));

// Fakes only: an in-memory vault and a scripted Test Connection. No provider is contacted and
// every secret below is fake.

const SECRET = "sk-test-not-a-real-key-ef025-0000";

type Handler = (args: Record<string, unknown>) => unknown;
let handlers: Record<string, Handler>;
let vault: Map<string, string>;

function vaultKey(args: Record<string, unknown>): string {
  return `${args.capability}:${args.configId}:${args.slot}`;
}

function validationResult(service: ServiceName, overrides: Record<string, unknown> = {}) {
  return {
    capability: service,
    providerId: useEngineStore.getState().services[service].providerId,
    status: "valid",
    latencyMs: 5,
    testedAt: 1_700_000_000,
    fingerprint: "a".repeat(64),
    credentialVersion: null,
    errorCode: null,
    ...overrides,
  };
}

function serializedState(state: EngineState): string {
  return JSON.stringify(state, (_key, value) => (typeof value === "function" ? undefined : value));
}

function cardFor(service: ServiceName): HTMLElement {
  render(<ServiceSettingsCard service={service} />);
  return screen.getByRole("region");
}

function phaseOf(card: HTMLElement): string | null {
  return card.querySelector("[data-phase]")?.getAttribute("data-phase") ?? null;
}

function testButton(card: HTMLElement): HTMLButtonElement {
  return within(card).getByRole("button", {
    name: /^(Test Connection|Test again|Retry|Testing\.\.\.|Retrying\.\.\.)$/,
  }) as HTMLButtonElement;
}

function providerOptions(card: HTMLElement): string[] {
  const select = within(card).getByLabelText("Provider") as HTMLSelectElement;
  return Array.from(select.options).map((o) => o.value);
}

function chooseProvider(card: HTMLElement, providerId: string) {
  fireEvent.change(within(card).getByLabelText("Provider"), { target: { value: providerId } });
}

beforeEach(() => {
  window.localStorage.clear();
  vault = new Map();
  handlers = {
    credential_status: (args) => ({ configured: vault.has(vaultKey(args)) }),
    credential_save: (args) => {
      if (vault.has(vaultKey(args))) throw { code: "already_configured" };
      vault.set(vaultKey(args), String(args.secret));
      return { configured: true };
    },
    credential_replace: (args) => {
      vault.set(vaultKey(args), String(args.secret));
      return { configured: true };
    },
    credential_delete: (args) => {
      vault.delete(vaultKey(args));
      return { configured: false };
    },
  };
  invokeMock.mockImplementation(async (command: string, args: Record<string, unknown>) => {
    const handler = handlers[command];
    if (!handler) throw { code: "bridge_failed" };
    return handler(args);
  });
  useEngineStore.setState({
    services: defaultServices(),
    running: false,
    secretStatus: { llmApiKey: false, hfToken: false },
    ...initialServiceUiState(),
  });
});

// ---------------------------------------------------------------------------
// Capability / provider matrix
// ---------------------------------------------------------------------------

const MATRIX: Record<ServiceName, string[]> = {
  stt: ["local", "openai-realtime", "gemini-live", "deepgram", "azure-speech", "assemblyai", "custom"],
  translation: [
    "local",
    "openai",
    "gemini",
    "azure-openai",
    "assemblyai-llm-gateway",
    "custom-openai-compatible",
  ],
  assistant: [
    "openai",
    "gemini",
    "azure-openai",
    "assemblyai-llm-gateway",
    "local-openai-compatible",
    "custom-openai-compatible",
  ],
};

const STREAMING_ONLY = ["deepgram", "azure-speech", "assemblyai", "openai-realtime", "gemini-live", "custom"];

describe("capability/provider matrix", () => {
  it.each(Object.keys(MATRIX) as ServiceName[])("offers exactly the %s providers", (service) => {
    if (service === "assistant") {
      useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    }
    const card = cardFor(service);
    expect(providerOptions(card)).toEqual(MATRIX[service]);
    expect(providersFor(service).map((m) => m.providerId)).toEqual(MATRIX[service]);
  });

  it("never lists streaming STT providers under text services", () => {
    for (const service of ["translation", "assistant"] as const) {
      const ids = providersFor(service).map((m) => m.providerId);
      for (const id of STREAMING_ONLY) expect(ids).not.toContain(id);
    }
  });

  it("mirrors the 13 Python built-in manifests", () => {
    expect(PROVIDER_CATALOG.map((m) => m.providerId).sort()).toEqual(
      [...new Set([...MATRIX.stt, ...MATRIX.translation, ...MATRIX.assistant])].sort(),
    );
    const custom = PROVIDER_CATALOG.find((m) => m.providerId === "custom");
    expect(custom?.protocolProfiles).toEqual({
      "openai-realtime": "openai_realtime",
      "gemini-live": "gemini_live",
    });
    const keys = custom?.fields.map((f) => f.key) ?? [];
    expect(keys).toEqual(
      expect.arrayContaining([
        "protocol_profile",
        "header_1_name",
        "header_2_value",
        "secret_header_1_name",
        "secret_header_2",
      ]),
    );
    expect(custom?.fields.find((f) => f.key === "secret_header_1")?.secret).toBe(true);
  });

  it("renders exactly three service cards in Settings", () => {
    render(<SettingsPanel />);
    expect(document.querySelectorAll("section.service-card")).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// Layout rules
// ---------------------------------------------------------------------------

describe("card layout", () => {
  it("keeps the API URL visible and editable for every provider", () => {
    const card = cardFor("stt");
    for (const id of MATRIX.stt) {
      chooseProvider(card, id);
      const url = within(card).getByLabelText(/^API URL/) as HTMLInputElement;
      expect(url.disabled).toBe(false);
      fireEvent.change(url, { target: { value: "wss://example.invalid/v1" } });
      expect(useEngineStore.getState().services.stt.endpoint).toBe("wss://example.invalid/v1");
    }
  });

  it("has no main-page source language selector; hints stay under Advanced", () => {
    const card = cardFor("stt");
    expect(within(card).getByText(/Source language: auto-detect/)).toBeTruthy();
    const hint = within(card).getByLabelText("Language hint");
    expect(hint.closest("details")).not.toBeNull();
  });

  it("shows the recommended model and keeps the override under Advanced", () => {
    const card = cardFor("translation");
    chooseProvider(card, "openai");
    expect(within(card).getByText("gpt-5.6-terra")).toBeTruthy();
    expect(within(card).getByLabelText("Model override").closest("details")).not.toBeNull();
  });

  it("requires a model for custom providers and links the error", () => {
    const card = cardFor("translation");
    chooseProvider(card, "custom-openai-compatible");
    const model = within(card).getByLabelText(/^Model/) as HTMLInputElement;
    expect(model.closest("details")).toBeNull();
    expect(model.getAttribute("aria-invalid")).toBe("true");
    const errorId = model.getAttribute("aria-describedby") ?? "";
    expect(document.getElementById(errorId)?.textContent).toMatch(/model is required/i);
    expect(testButton(card).disabled).toBe(true);

    fireEvent.change(model, { target: { value: "my-model" } });
    fireEvent.change(within(card).getByLabelText(/^API URL/), {
      target: { value: "http://localhost:8000/v1" },
    });
    expect(model.getAttribute("aria-invalid")).toBeNull();
    expect(testButton(card).disabled).toBe(false);
  });

  it("offers the Custom STT protocol profile choice", () => {
    const card = cardFor("stt");
    chooseProvider(card, "custom");
    const protocol = within(card).getByLabelText(/^Protocol/) as HTMLSelectElement;
    expect(Array.from(protocol.options).map((o) => o.value)).toEqual([
      "",
      "openai-realtime",
      "gemini-live",
    ]);
    fireEvent.change(protocol, { target: { value: "gemini-live" } });
    expect(useEngineStore.getState().services.stt.options.protocol_profile).toBe("gemini-live");
  });

  it("uses typed inputs with accessible errors and never a JSON textarea", () => {
    const card = cardFor("translation");
    chooseProvider(card, "openai");
    expect(card.querySelector("textarea")).toBeNull();
    const temperature = within(card).getByLabelText("Temperature") as HTMLInputElement;
    fireEvent.change(temperature, { target: { value: "abc" } });
    expect(temperature.value).toBe("abc");
    expect(temperature.getAttribute("aria-invalid")).toBe("true");
    const described = (temperature.getAttribute("aria-describedby") ?? "").split(" ");
    const messages = described.map((id) => document.getElementById(id)?.textContent);
    expect(messages).toContain("Enter a number.");
    expect(useEngineStore.getState().services.translation.options.temperature).toBeUndefined();

    fireEvent.change(temperature, { target: { value: "3" } });
    expect(within(card).getByText("Must be at most 2.")).toBeTruthy();
    fireEvent.change(temperature, { target: { value: "0.7" } });
    expect(temperature.getAttribute("aria-invalid")).toBeNull();
    expect(useEngineStore.getState().services.translation.options.temperature).toBe(0.7);
  });

  it("lets the Assistant be disabled", () => {
    const card = cardFor("assistant");
    expect(within(card).queryByLabelText("Provider")).toBeNull();
    fireEvent.click(within(card).getByLabelText("Enable Assistant"));
    expect(useEngineStore.getState().services.assistant.enabled).toBe(true);
    expect(within(card).getByLabelText("Provider")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Test Connection states
// ---------------------------------------------------------------------------

describe("Test Connection states", () => {
  it("moves through idle, loading, success, and stale", async () => {
    let resolveTest: (value: unknown) => void = () => undefined;
    handlers.provider_test_connection = () => new Promise((resolve) => (resolveTest = resolve));
    const card = cardFor("stt");
    expect(phaseOf(card)).toBe("idle");

    fireEvent.click(testButton(card));
    await waitFor(() => expect(phaseOf(card)).toBe("loading"));
    expect(testButton(card).disabled).toBe(true);

    resolveTest(validationResult("stt"));
    await waitFor(() => expect(phaseOf(card)).toBe("success"));
    expect(within(card).getByRole("status").textContent).toMatch(/succeeded/);

    fireEvent.change(within(card).getByLabelText(/^API URL/), { target: { value: "x" } });
    expect(phaseOf(card)).toBe("stale");
    expect(useEngineStore.getState().services.stt.validation.status).toBe("unvalidated");
    expect(within(card).getByRole("status").textContent).toMatch(/changed since the last test/);
  });

  it("shows a safe message for the error code, then retries", async () => {
    handlers.provider_test_connection = () =>
      validationResult("stt", { status: "invalid", errorCode: "auth_failed" });
    const card = cardFor("stt");

    fireEvent.click(testButton(card));
    await waitFor(() => expect(phaseOf(card)).toBe("error"));
    expect(within(card).getByRole("status").textContent).toBe(
      "The provider rejected the saved credential. (auth_failed)",
    );
    expect(testButton(card).textContent).toBe("Retry");

    let resolveTest: (value: unknown) => void = () => undefined;
    handlers.provider_test_connection = () => new Promise((resolve) => (resolveTest = resolve));
    fireEvent.click(testButton(card));
    await waitFor(() => expect(phaseOf(card)).toBe("retry"));
    resolveTest(validationResult("stt"));
    await waitFor(() => expect(phaseOf(card)).toBe("success"));
  });

  it("never echoes bridge error text", async () => {
    handlers.provider_test_connection = () => {
      throw { code: "validator_unavailable", message: `leak ${SECRET}` };
    };
    const card = cardFor("stt");
    fireEvent.click(testButton(card));
    await waitFor(() => expect(phaseOf(card)).toBe("error"));
    expect(card.textContent).not.toContain(SECRET);
    expect(within(card).getByRole("status").textContent).toContain("(validator_unavailable)");
  });
});

// ---------------------------------------------------------------------------
// Write-only secrets
// ---------------------------------------------------------------------------

describe("secret fields", () => {
  it("saves write-only, shows configured, replaces, and deletes", async () => {
    const card = cardFor("translation");
    chooseProvider(card, "openai");
    expect(within(card).getByText("Not configured")).toBeTruthy();

    const input = within(card).getByLabelText(/^API key/) as HTMLInputElement;
    expect(input.type).toBe("password");
    fireEvent.change(input, { target: { value: SECRET } });
    expect(serializedState(useEngineStore.getState())).not.toContain(SECRET);
    fireEvent.click(within(card).getByRole("button", { name: "Save" }));
    expect(input.value).toBe("");

    await waitFor(() => expect(within(card).getByText("Configured")).toBeTruthy());
    expect(vault.get("translation:openai:api_key")).toBe(SECRET);
    const ref = useEngineStore.getState().services.translation.credentialRefs.api_key;
    expect(ref).toEqual({ configId: "openai", slot: "api_key", version: 1 });

    // Nothing the store holds or persists can read the value back.
    const state = useEngineStore.getState();
    expect(serializedState(state)).not.toContain(SECRET);
    expect(JSON.stringify(selectPersistedState(state))).not.toContain(SECRET);
    for (let i = 0; i < window.localStorage.length; i++) {
      const key = window.localStorage.key(i) ?? "";
      expect(window.localStorage.getItem(key) ?? "").not.toContain(SECRET);
    }
    expect(card.textContent).not.toContain(SECRET);

    fireEvent.click(within(card).getByRole("button", { name: "Replace API key" }));
    const replacement = within(card).getByLabelText(/^API key/) as HTMLInputElement;
    expect(document.activeElement).toBe(replacement);
    fireEvent.change(replacement, { target: { value: `${SECRET}-2` } });
    fireEvent.keyDown(replacement, { key: "Enter" });
    await waitFor(() =>
      expect(useEngineStore.getState().services.translation.credentialRefs.api_key?.version).toBe(2),
    );
    expect(serializedState(useEngineStore.getState())).not.toContain(SECRET);

    fireEvent.click(within(card).getByRole("button", { name: "Delete API key" }));
    await waitFor(() => expect(within(card).getByText("Not configured")).toBeTruthy());
    expect(vault.has("translation:openai:api_key")).toBe(false);
    expect(useEngineStore.getState().services.translation.credentialRefs.api_key).toBeUndefined();
  });

  it("marks a prior successful test stale when the credential changes", async () => {
    handlers.provider_test_connection = () => validationResult("assistant");
    vault.set("assistant:openai:api_key", SECRET);
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    // SettingsPanel dispatches this on mount; the card alone does not.
    await useEngineStore.getState().refreshServiceCredentials("assistant");
    const card = cardFor("assistant");
    expect(within(card).getByText("Configured")).toBeTruthy();

    fireEvent.click(testButton(card));
    await waitFor(() => expect(phaseOf(card)).toBe("success"));

    fireEvent.click(within(card).getByRole("button", { name: "Delete API key" }));
    await waitFor(() => expect(phaseOf(card)).toBe("stale"));
  });

  it("reports a vault failure without keeping the typed value", async () => {
    handlers.credential_save = () => {
      throw { code: "vault_unavailable", message: SECRET };
    };
    const card = cardFor("translation");
    chooseProvider(card, "gemini");
    const input = within(card).getByLabelText(/^API key/) as HTMLInputElement;
    fireEvent.change(input, { target: { value: SECRET } });
    fireEvent.click(within(card).getByRole("button", { name: "Save" }));

    const alert = await within(card).findByRole("alert");
    expect(alert.textContent).toContain("(vault_unavailable)");
    expect(input.getAttribute("aria-describedby")).toContain(alert.id);
    expect(input.value).toBe("");
    expect(card.textContent).not.toContain(SECRET);
    expect(serializedState(useEngineStore.getState())).not.toContain(SECRET);
    expect(useEngineStore.getState().services.translation.credentialRefs).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// Session lock
// ---------------------------------------------------------------------------

describe("session lock", () => {
  it("disables every control while a session runs", () => {
    useEngineStore.getState().updateServiceConfig("assistant", { enabled: true });
    useEngineStore.setState({ running: true });
    render(<SettingsPanel />);
    const cards = Array.from(document.querySelectorAll<HTMLElement>("section.service-card"));
    expect(cards).toHaveLength(3);
    for (const card of cards) {
      const controls = card.querySelectorAll<HTMLInputElement>("input, select, button");
      expect(controls.length).toBeGreaterThan(0);
      for (const control of Array.from(controls)) expect(control.disabled).toBe(true);
    }

    useEngineStore.getState().selectServiceProvider("stt", "deepgram");
    expect(useEngineStore.getState().services.stt.providerId).toBe("local");
  });
});
