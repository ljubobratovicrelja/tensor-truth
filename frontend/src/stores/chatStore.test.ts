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

  describe("confirmation requests", () => {
    const make = (id: string) => ({
      confirmation_id: id,
      tool_name: "add_arxiv_paper",
      action_type: "add_document",
      title: `Add arXiv paper ${id}`,
      summary: "",
      details: {},
    });

    beforeEach(() => {
      useChatStore.setState({ confirmationRequests: [] });
    });

    it("appends distinct confirmation requests", () => {
      useChatStore.getState().addConfirmationRequest(make("a"));
      useChatStore.getState().addConfirmationRequest(make("b"));
      expect(useChatStore.getState().confirmationRequests).toHaveLength(2);
    });

    it("ignores a duplicate frame for the same confirmation_id", () => {
      useChatStore.getState().addConfirmationRequest(make("a"));
      useChatStore.getState().addConfirmationRequest(make("a"));
      const requests = useChatStore.getState().confirmationRequests;
      expect(requests).toHaveLength(1);
      expect(requests[0].confirmation_id).toBe("a");
    });
  });

  describe("tool phase progress", () => {
    const calling = (tool: string, params: Record<string, unknown> = {}) => ({
      type: "tool_progress" as const,
      tool,
      action: "calling" as const,
      params,
      tool_id: tool,
    });

    const phase = (
      tool: string,
      percent: number,
      metadata: Record<string, unknown> = {}
    ) => ({
      type: "tool_phase" as const,
      tool_id: tool,
      phase: "converting",
      message: `Converting 1/2: ${tool}.pdf`,
      metadata: { percent, message: `Converting 1/2: ${tool}.pdf`, ...metadata },
    });

    beforeEach(() => {
      useChatStore.getState().startStreaming("index my docs");
    });

    it("attaches percent phases to the running step for that tool", () => {
      useChatStore.getState().addToolStep(calling("build_index"));
      useChatStore.getState().addToolStep(calling("web_search"));

      useChatStore.getState().setToolPhase(phase("build_index", 35));

      const steps = useChatStore.getState().streamingToolSteps;
      expect(steps[0].progress).toEqual({
        percent: 35,
        message: "Converting 1/2: build_index.pdf",
        stage: "converting",
      });
      expect(steps[1].progress).toBeUndefined();
    });

    it("does not clobber two concurrent same-tool calls with distinct scopes", () => {
      useChatStore.getState().addToolStep(calling("build_index", { scope: "session" }));
      useChatStore.getState().addToolStep(calling("build_index", { scope: "project" }));

      useChatStore
        .getState()
        .setToolPhase(phase("build_index", 80, { scope: "project" }));

      const steps = useChatStore.getState().streamingToolSteps;
      expect(steps[0].progress).toBeUndefined();
      expect(steps[1].progress?.percent).toBe(80);
    });

    it("keeps phases without a percent out of the step model", () => {
      useChatStore.getState().addToolStep(calling("build_index"));
      useChatStore.getState().setToolPhase({
        type: "tool_phase",
        tool_id: "build_index",
        phase: "generating",
        message: "Thinking...",
        metadata: {},
      });

      expect(useChatStore.getState().streamingToolSteps[0].progress).toBeUndefined();
      expect(useChatStore.getState().toolPhase?.phase).toBe("generating");
    });
  });
});
