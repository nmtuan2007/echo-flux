import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEngineStore } from "../store/engineStore";
import { ContextProfilesPanel } from "./ContextProfilesPanel";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/tauri", () => ({ invoke: invokeMock }));

describe("ContextProfilesPanel Component (EF-027)", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    window.localStorage.clear();
    useEngineStore.setState({
      profiles: [],
      profilesLoading: false,
      profilesError: null,
      activeProfile: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state when no profiles exist", async () => {
    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "profile_list_result", profiles: [] }],
    });

    render(<ContextProfilesPanel />);

    expect(screen.getByRole("heading", { name: /^Context Profiles$/i })).toBeTruthy();
    await waitFor(() => {
      expect(
        screen.getByText(/No context profiles yet\. Create a profile to provide background/i),
      ).toBeTruthy();
    });
  });

  it("renders profile cards from profile_list", async () => {
    const mockProfiles = [
      {
        profile_id: "p1",
        name: "Medical Terms",
        content_format: "markdown",
        revision: 2,
        content_bytes: 1024,
        updated_at_ms: Date.now(),
      },
      {
        profile_id: "p2",
        name: "Acronyms Glossary",
        content_format: "text",
        revision: 1,
        content_bytes: 2048,
        updated_at_ms: Date.now(),
      },
    ];

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "profile_list_result", profiles: mockProfiles }],
    });

    render(<ContextProfilesPanel />);

    await waitFor(() => {
      expect(screen.getByText("Medical Terms")).toBeTruthy();
      expect(screen.getByText("Acronyms Glossary")).toBeTruthy();
      expect(screen.getByText("v2")).toBeTruthy();
      expect(screen.getByText("v1")).toBeTruthy();
    });
  });

  it("validates name and content byte limits on creation", async () => {
    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "profile_list_result", profiles: [] }],
    });

    render(<ContextProfilesPanel />);

    const newBtn = screen.getByRole("button", { name: /\+ New Profile/i });
    fireEvent.click(newBtn);

    expect(screen.getByText(/Create Context Profile/i)).toBeTruthy();

    const nameInput = screen.getByPlaceholderText(/e\.g\. Healthcare Acronyms/i) as HTMLInputElement;
    const contentInput = screen.getByPlaceholderText(/Enter glossary definitions/i) as HTMLTextAreaElement;

    // Check byte counter initial value (65536 bytes)
    expect(screen.getByText(/65536 bytes remaining/i)).toBeTruthy();

    // Type content and check remaining bytes decrement
    fireEvent.change(contentInput, { target: { value: "Hello World" } });
    expect(screen.getByText(/65525 bytes remaining/i)).toBeTruthy();

    // Fill valid data and submit
    fireEvent.change(nameInput, { target: { value: "New Profile 1" } });

    invokeMock.mockResolvedValueOnce({
      messages: [
        {
          type: "profile_create_result",
          profile: {
            profile_id: "p-new-1",
            name: "New Profile 1",
            content: "Hello World",
            content_format: "markdown",
            revision: 1,
          },
        },
      ],
    });
    // Second call for reloadProfiles
    invokeMock.mockResolvedValueOnce({
      messages: [
        {
          type: "profile_list_result",
          profiles: [
            {
              profile_id: "p-new-1",
              name: "New Profile 1",
              content_format: "markdown",
              revision: 1,
              content_bytes: 11,
              updated_at_ms: Date.now(),
            },
          ],
        },
      ],
    });

    const submitBtn = screen.getByRole("button", { name: /Create Profile/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith("engine_forward_command", {
        command: expect.objectContaining({
          type: "profile_create",
          name: "New Profile 1",
          content: "Hello World",
        }),
      });
    });
  });

  it("handles profile deletion with confirmation modal and snapshot notice", async () => {
    const mockProfiles = [
      {
        profile_id: "p-del",
        name: "Profile to Delete",
        content_format: "markdown",
        revision: 1,
        content_bytes: 50,
        updated_at_ms: Date.now(),
      },
    ];

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "profile_list_result", profiles: mockProfiles }],
    });

    render(<ContextProfilesPanel />);

    await waitFor(() => {
      expect(screen.getByText("Profile to Delete")).toBeTruthy();
    });

    const deleteBtn = screen.getByRole("button", { name: /🗑 Delete/i });
    fireEvent.click(deleteBtn);

    // Confirmation dialog should be visible
    expect(screen.getByText(/Delete Context Profile\?/i)).toBeTruthy();
    expect(
      screen.getByText(/Deleting this profile removes it from future sessions/i),
    ).toBeTruthy();

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "profile_delete_result", deleted: true, profile_id: "p-del" }],
    });

    const confirmBtn = screen.getByRole("button", { name: /^Delete Profile$/i });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith("engine_forward_command", {
        command: expect.objectContaining({
          type: "profile_delete",
          profile_id: "p-del",
        }),
      });
    });
  });
});
