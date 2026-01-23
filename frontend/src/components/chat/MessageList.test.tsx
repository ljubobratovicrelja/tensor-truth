import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageList } from "./MessageList";

// Mock the hooks that MessageList uses
vi.mock("@/hooks", () => ({
  useScrollDirection: () => ({
    direction: null,
    isAtTop: true,
    isNearTop: true,
    isScrollable: false,
    scrollRef: vi.fn(),
  }),
  useIsMobile: () => false,
}));

vi.mock("@/stores", () => ({
  useUIStore: () => vi.fn(),
}));

describe("MessageList", () => {
  const defaultProps = {
    sessionId: "test-session",
    messages: [],
    isLoading: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("loading state", () => {
    it("shows skeleton when loading with no content", () => {
      render(<MessageList {...defaultProps} isLoading={true} />);

      // Should show skeletons (they have specific classes)
      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it("shows pending message instead of skeleton when loading", () => {
      render(
        <MessageList
          {...defaultProps}
          isLoading={true}
          pendingUserMessage="Hello, this is my message"
        />
      );

      // Should NOT show skeletons
      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBe(0);

      // Should show the pending message
      expect(screen.getByText("Hello, this is my message")).toBeInTheDocument();
    });

    it("shows streaming UI instead of skeleton when loading and streaming", () => {
      const { container } = render(
        <MessageList
          {...defaultProps}
          isLoading={true}
          isStreaming={true}
          streamingContent="Streaming response"
        />
      );

      // Should NOT show the loading skeleton pattern (3 skeletons stacked)
      // The skeleton loading state has a specific structure we can detect
      const hasLoadingSkeletonPattern =
        container.querySelectorAll(".space-y-4 > div").length === 3 &&
        container.querySelector(".space-y-4 .h-16") !== null;
      expect(hasLoadingSkeletonPattern).toBe(false);

      // Should show the message list structure (space-y-2 for messages)
      expect(container.querySelector(".space-y-2")).toBeInTheDocument();

      // Should render a MessageItem for the streaming response (has bot icon)
      expect(container.querySelector(".lucide-bot")).toBeInTheDocument();
    });

    it("shows existing messages instead of skeleton when loading", () => {
      render(
        <MessageList
          {...defaultProps}
          isLoading={true}
          messages={[{ role: "user", content: "Previous message" }]}
        />
      );

      // Should NOT show skeletons
      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBe(0);

      // Should show the existing message
      expect(screen.getByText("Previous message")).toBeInTheDocument();
    });
  });

  describe("pending message display", () => {
    it("shows pending message when not in messages list", () => {
      render(
        <MessageList
          {...defaultProps}
          messages={[]}
          pendingUserMessage="New pending message"
        />
      );

      expect(screen.getByText("New pending message")).toBeInTheDocument();
    });

    it("deduplicates pending message when it exists in messages", () => {
      render(
        <MessageList
          {...defaultProps}
          messages={[{ role: "user", content: "Same message" }]}
          pendingUserMessage="Same message"
        />
      );

      // Should only show the message once (from messages array, not pending)
      const messageElements = screen.getAllByText("Same message");
      expect(messageElements.length).toBe(1);
    });

    it("shows pending message alongside streaming indicator", () => {
      const { container } = render(
        <MessageList
          {...defaultProps}
          pendingUserMessage="User question"
          isStreaming={true}
          pipelineStatus="generating"
        />
      );

      // Should show the user's pending message
      expect(screen.getByText("User question")).toBeInTheDocument();

      // StreamingIndicator should be present (it shows random labels, so check for the icon)
      // The generating status uses Sparkles icon with animate-pulse
      const sparklesIcon = container.querySelector(".lucide-sparkles");
      expect(sparklesIcon).toBeInTheDocument();
    });
  });

  describe("empty state", () => {
    it("shows empty state message when no content", () => {
      render(<MessageList {...defaultProps} />);

      expect(
        screen.getByText("Send a message to start the conversation")
      ).toBeInTheDocument();
    });

    it("does not show empty state when there are messages", () => {
      render(
        <MessageList {...defaultProps} messages={[{ role: "user", content: "Hello" }]} />
      );

      expect(
        screen.queryByText("Send a message to start the conversation")
      ).not.toBeInTheDocument();
    });

    it("does not show empty state when pending message exists", () => {
      render(<MessageList {...defaultProps} pendingUserMessage="Pending message" />);

      expect(
        screen.queryByText("Send a message to start the conversation")
      ).not.toBeInTheDocument();
    });

    it("does not show empty state when streaming", () => {
      render(<MessageList {...defaultProps} isStreaming={true} />);

      expect(
        screen.queryByText("Send a message to start the conversation")
      ).not.toBeInTheDocument();
    });
  });
});
