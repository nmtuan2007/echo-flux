import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEngineStore } from "../store/engineStore";
import { SummaryModal } from "./SummaryModal";
import { HistoryPanel } from "./HistoryPanel";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/tauri", () => ({ invoke: invokeMock }));

describe("SummaryModal Component (EF-029)", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    window.localStorage.clear();
    useEngineStore.setState({
      summaryVisible: false,
      summaryLoading: false,
      summaryText: "",
      summaryWarning: null,
      summaryError: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not render when summaryVisible is false", () => {
    const { container } = render(<SummaryModal />);
    expect(container.firstChild).toBeNull();
  });

  it("renders summary content when visible", () => {
    useEngineStore.setState({
      summaryVisible: true,
      summaryLoading: false,
      summaryText: "### Action Items\n- Follow up with team\n- Prepare release notes",
      summaryWarning: null,
      summaryError: null,
    });

    render(<SummaryModal />);

    expect(screen.getByText(/Meeting Summary/i)).toBeTruthy();
    expect(screen.getByText("Action Items")).toBeTruthy();
    expect(screen.getByText("Follow up with team")).toBeTruthy();
  });

  it("displays cancel button and calls cancelSummary while streaming", () => {
    const cancelSpy = vi.fn();
    useEngineStore.setState({
      summaryVisible: true,
      summaryLoading: true,
      summaryText: "Analyzing meeting notes...",
      cancelSummary: cancelSpy,
    });

    render(<SummaryModal />);

    const stopBtn = screen.getByRole("button", { name: /stop/i });
    expect(stopBtn).toBeTruthy();
    fireEvent.click(stopBtn);
    expect(cancelSpy).toHaveBeenCalledTimes(1);
  });

  it("renders truncation warning banner when older messages are omitted", () => {
    useEngineStore.setState({
      summaryVisible: true,
      summaryLoading: false,
      summaryText: "Summary covering latest messages.",
      summaryWarning: {
        reason: "max_tokens_exceeded",
        omitted_message_count: 14,
        used_tokens: 4000,
        max_tokens: 4096,
        included_message_count: 20,
      },
    });

    render(<SummaryModal />);

    const warningBanner = screen.getByRole("status");
    expect(warningBanner).toBeTruthy();
    expect(
      screen.getByText(/Transcript was truncated: 14 earlier messages omitted to fit model context/i),
    ).toBeTruthy();
    expect(screen.getByText(/Summary covers the most recent 20 messages/i)).toBeTruthy();
  });

  it("renders error banner when summaryError is set", () => {
    useEngineStore.setState({
      summaryVisible: true,
      summaryLoading: false,
      summaryText: "",
      summaryError: "Model rate limit exceeded. Please try again later.",
    });

    render(<SummaryModal />);

    const errorAlert = screen.getByRole("alert");
    expect(errorAlert).toBeTruthy();
    expect(screen.getByText(/Model rate limit exceeded/i)).toBeTruthy();
  });
});

describe("HistoryPanel Component (EF-029 & SQLite Integration)", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    window.localStorage.clear();
    useEngineStore.setState({
      history: [],
      savedConversations: [],
      historyLoading: false,
      historySearchQuery: "",
      selectedConversationDetail: null,
      running: false,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders saved conversations from SQLite repository", async () => {
    const mockConversations = [
      {
        conversation_id: "c-101",
        title: "Sprint Planning Q4",
        created_at_ms: Date.now() - 10000,
        updated_at_ms: Date.now() - 5000,
        message_count: 42,
        has_summary: true,
      },
      {
        conversation_id: "c-102",
        title: "Design Sync with Marketing",
        created_at_ms: Date.now() - 20000,
        updated_at_ms: Date.now() - 15000,
        message_count: 18,
        has_summary: false,
      },
    ];

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "conversation_list_result", conversations: mockConversations }],
    });

    render(<HistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText("Sprint Planning Q4")).toBeTruthy();
      expect(screen.getByText("Design Sync with Marketing")).toBeTruthy();
      expect(screen.getByText(/42 entries/i)).toBeTruthy();
      expect(screen.getByText(/18 entries/i)).toBeTruthy();
      expect(screen.getByText(/📝 Summary/i)).toBeTruthy();
    });
  });

  it("filters conversations via real-time search input", async () => {
    const mockConversations = [
      {
        conversation_id: "c-1",
        title: "Quarterly Financial Review",
        created_at_ms: Date.now(),
        updated_at_ms: Date.now(),
        message_count: 10,
      },
      {
        conversation_id: "c-2",
        title: "Architectural Deep Dive",
        created_at_ms: Date.now() - 5000,
        updated_at_ms: Date.now() - 5000,
        message_count: 5,
      },
    ];

    invokeMock.mockResolvedValueOnce({
      messages: [{ type: "conversation_list_result", conversations: mockConversations }],
    });

    render(<HistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText("Quarterly Financial Review")).toBeTruthy();
      expect(screen.getByText("Architectural Deep Dive")).toBeTruthy();
    });

    const searchInput = screen.getByRole("textbox", { name: /search conversations/i });
    fireEvent.change(searchInput, { target: { value: "Architectural" } });

    await waitFor(() => {
      expect(screen.queryByText("Quarterly Financial Review")).toBeNull();
      expect(screen.getByText("Architectural Deep Dive")).toBeTruthy();
    });
  });

  it("opens delete confirmation modal and permanently deletes conversation", async () => {
    const mockConversations = [
      {
        conversation_id: "c-del",
        title: "Temporary Demo Session",
        created_at_ms: Date.now(),
        updated_at_ms: Date.now(),
        message_count: 3,
      },
    ];

    invokeMock
      .mockResolvedValueOnce({
        messages: [{ type: "conversation_list_result", conversations: mockConversations }],
      })
      .mockResolvedValueOnce({
        messages: [{ type: "conversation_deleted", conversation_id: "c-del" }],
      });

    render(<HistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText("Temporary Demo Session")).toBeTruthy();
    });

    // Click delete icon button
    const deleteBtn = screen.getByRole("button", { name: /delete conversation/i });
    fireEvent.click(deleteBtn);

    // Modal dialog appears
    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(screen.getByText(/Are you sure you want to permanently delete/i)).toBeTruthy();

    // Confirm deletion
    const confirmBtn = screen.getByRole("button", { name: /^delete$/i });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(screen.queryByText("Temporary Demo Session")).toBeNull();
      expect(screen.getByText(/No saved conversations/i)).toBeTruthy();
    });
  });

  it("navigates to conversation detail view and returns back", async () => {
    const mockConversations = [
      {
        conversation_id: "c-view",
        title: "Product Roadmap Discussion",
        created_at_ms: Date.now(),
        updated_at_ms: Date.now(),
        message_count: 2,
      },
    ];

    const mockDetail = {
      type: "conversation_get_result",
      conversation: {
        conversation_id: "c-view",
        title: "Product Roadmap Discussion",
        created_at_ms: Date.now(),
      },
      messages: [
        {
          message_id: "m-1",
          source: "mic",
          speaker_label: "Me",
          stt_text: "Welcome everyone to the roadmap sync.",
          translation_text: null,
          timestamp_ms: Date.now(),
        },
        {
          message_id: "m-2",
          source: "system",
          speaker_label: "Others",
          stt_text: "Thanks! We are ready to present the timeline.",
          translation_text: "Cảm ơn! Chúng tôi đã sẵn sàng trình bày kế hoạch.",
          timestamp_ms: Date.now() + 1000,
        },
      ],
      summary: {
        summary_markdown: "## Key Takeaways\n- Timeline is on track for Q4.",
      },
    };

    invokeMock
      .mockResolvedValueOnce({
        messages: [{ type: "conversation_list_result", conversations: mockConversations }],
      })
      .mockResolvedValueOnce({
        messages: [mockDetail],
      });

    render(<HistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText("Product Roadmap Discussion")).toBeTruthy();
    });

    const itemTitle = screen.getByText("Product Roadmap Discussion");
    fireEvent.click(itemTitle);

    await waitFor(() => {
      expect(screen.getByText("Welcome everyone to the roadmap sync.")).toBeTruthy();
      expect(screen.getByText("Thanks! We are ready to present the timeline.")).toBeTruthy();
      expect(screen.getByText(/Cảm ơn! Chúng tôi đã sẵn sàng/i)).toBeTruthy();
      expect(screen.getByText(/Timeline is on track for Q4/i)).toBeTruthy();
    });

    // Click Back to History
    const backBtn = screen.getByRole("button", { name: /back to history/i });
    fireEvent.click(backBtn);

    await waitFor(() => {
      expect(screen.getByText("Conversation History")).toBeTruthy();
    });
  });
});
