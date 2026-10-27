import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ContextMeter } from "./ContextMeter";

const contextData = {
  measured: true,
  categories: [
    {
      id: "system",
      label: "System prompt",
      hint: "",
      chars: 4000,
      tokens: 1000,
      share: 0.1,
    },
    {
      id: "tools",
      label: "Tool schemas",
      hint: "",
      chars: 28000,
      tokens: 7000,
      share: 0.7,
    },
    {
      id: "history",
      label: "Conversation history",
      hint: "",
      chars: 8000,
      tokens: 2000,
      share: 0.2,
    },
    { id: "prompt", label: "Current message", hint: "", chars: 0, tokens: 0, share: 0 },
  ],
  total_chars: 40000,
  total_tokens: 10000,
  context_window: 131072,
  used_percent: 7.6,
  remaining_tokens: 121072,
  model_name: "test-model",
  messages_included: 6,
  messages_total: 8,
  measured_at: 1,
};

const mockContext = vi.fn();
vi.mock("@/hooks/useSessionContext", () => ({
  useSessionContext: () => mockContext(),
}));

describe("ContextMeter", () => {
  it("renders nothing without a session", () => {
    mockContext.mockReturnValue({ data: undefined, isLoading: false });
    const { container } = render(<ContextMeter />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows total against the window", () => {
    mockContext.mockReturnValue({ data: contextData, isLoading: false });
    render(<ContextMeter sessionId="s1" />);
    expect(screen.getByTestId("context-meter")).toHaveTextContent("10K");
    expect(screen.getByTestId("context-meter")).toHaveTextContent("/ 131K");
    expect(screen.getByTestId("context-meter")).toHaveTextContent("7.6%");
    expect(screen.getByTestId("context-meter")).not.toHaveTextContent("estimate");
  });

  it("marks unmeasured numbers as an estimate", () => {
    mockContext.mockReturnValue({
      data: { ...contextData, measured: false },
      isLoading: false,
    });
    render(<ContextMeter sessionId="s1" />);
    expect(screen.getByTestId("context-meter")).toHaveTextContent("estimate");
  });

  it("breaks the total down by category in the popover", async () => {
    mockContext.mockReturnValue({ data: contextData, isLoading: false });
    const user = userEvent.setup();
    render(<ContextMeter sessionId="s1" />);

    await user.click(screen.getByTestId("context-meter"));

    expect(screen.getByText("Context window")).toBeInTheDocument();
    expect(screen.getByText("System prompt")).toBeInTheDocument();
    expect(screen.getByText("Tool schemas")).toBeInTheDocument();
    expect(screen.getByText("Conversation history")).toBeInTheDocument();
    // zero-sized categories are hidden
    expect(screen.queryByText("Current message")).toBeNull();
    expect(screen.getByText("6 of 8 messages included")).toBeInTheDocument();
  });

  it("offers compaction and reports the click", async () => {
    mockContext.mockReturnValue({ data: contextData, isLoading: false });
    const onCompact = vi.fn();
    const user = userEvent.setup();
    render(<ContextMeter sessionId="s1" onCompact={onCompact} />);

    await user.click(screen.getByTestId("context-meter"));
    await user.click(screen.getByRole("button", { name: /Compact conversation/i }));
    expect(onCompact).toHaveBeenCalledTimes(1);
  });

  it("disables compaction while a request is pending", async () => {
    mockContext.mockReturnValue({ data: contextData, isLoading: false });
    const user = userEvent.setup();
    render(<ContextMeter sessionId="s1" onCompact={() => {}} compactPending />);

    await user.click(screen.getByTestId("context-meter"));
    expect(screen.getByRole("button", { name: /Compact conversation/i })).toBeDisabled();
  });
});
