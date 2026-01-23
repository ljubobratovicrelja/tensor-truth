import { describe, it, expect, beforeEach } from "vitest";
import { useChatStore } from "./chatStore";

describe("chatStore", () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useChatStore.setState({
      isStreaming: false,
      streamingContent: "",
      streamingThinking: "",
      streamingSources: [],
      streamingMetrics: null,
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      pendingUserMessage: null,
    });
  });

  describe("pendingUserMessage management", () => {
    it("should set pending user message when streaming starts", () => {
      useChatStore.getState().startStreaming("Hello world");

      expect(useChatStore.getState().pendingUserMessage).toBe("Hello world");
      expect(useChatStore.getState().isStreaming).toBe(true);
    });

    it("should clear pending user message when streaming finishes", () => {
      useChatStore.getState().startStreaming("Hello world");
      expect(useChatStore.getState().pendingUserMessage).toBe("Hello world");

      useChatStore.getState().finishStreaming("Response content", "normal");

      expect(useChatStore.getState().pendingUserMessage).toBe(null);
      expect(useChatStore.getState().isStreaming).toBe(false);
    });

    it("should clear pending user message on error", () => {
      useChatStore.getState().startStreaming("Hello world");
      expect(useChatStore.getState().pendingUserMessage).toBe("Hello world");

      useChatStore.getState().setError("Something went wrong");

      expect(useChatStore.getState().pendingUserMessage).toBe(null);
      expect(useChatStore.getState().error).toBe("Something went wrong");
    });

    it("should NOT clear pending user message on reset (for auto-send feature)", () => {
      useChatStore.getState().setPendingUserMessage("Preserved message");

      useChatStore.getState().reset();

      // pendingUserMessage is intentionally NOT cleared by reset()
      // This is needed for the auto-send feature from welcome page
      expect(useChatStore.getState().pendingUserMessage).toBe("Preserved message");
    });

    it("should clear pending user message with clearPendingUserMessage", () => {
      useChatStore.getState().setPendingUserMessage("Message to clear");
      expect(useChatStore.getState().pendingUserMessage).toBe("Message to clear");

      useChatStore.getState().clearPendingUserMessage();

      expect(useChatStore.getState().pendingUserMessage).toBe(null);
    });

    it("should allow explicit setting and clearing of pending message", () => {
      // Set
      useChatStore.getState().setPendingUserMessage("First message");
      expect(useChatStore.getState().pendingUserMessage).toBe("First message");

      // Update
      useChatStore.getState().setPendingUserMessage("Second message");
      expect(useChatStore.getState().pendingUserMessage).toBe("Second message");

      // Clear
      useChatStore.getState().clearPendingUserMessage();
      expect(useChatStore.getState().pendingUserMessage).toBe(null);
    });
  });

  describe("streaming state", () => {
    it("should accumulate streaming content with appendToken", () => {
      useChatStore.getState().startStreaming("User message");

      useChatStore.getState().appendToken("Hello ");
      useChatStore.getState().appendToken("world");

      expect(useChatStore.getState().streamingContent).toBe("Hello world");
    });

    it("should accumulate thinking content with appendThinking", () => {
      useChatStore.getState().startStreaming("User message");

      useChatStore.getState().appendThinking("Let me ");
      useChatStore.getState().appendThinking("think...");

      expect(useChatStore.getState().streamingThinking).toBe("Let me think...");
    });

    it("should clear all streaming state except pendingUserMessage on reset", () => {
      // Set up state manually to avoid startStreaming setting pendingUserMessage
      useChatStore.setState({
        isStreaming: true,
        streamingContent: "Some content",
        streamingThinking: "Some thinking",
        pipelineStatus: "generating",
        error: "Some error",
        pendingUserMessage: "User message",
      });

      useChatStore.getState().reset();

      const state = useChatStore.getState();
      expect(state.isStreaming).toBe(false);
      expect(state.streamingContent).toBe("");
      expect(state.streamingThinking).toBe("");
      expect(state.pipelineStatus).toBe(null);
      expect(state.error).toBe(null);
      // pendingUserMessage is preserved by reset()
      expect(state.pendingUserMessage).toBe("User message");
    });
  });
});
