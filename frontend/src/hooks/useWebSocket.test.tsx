import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWebSocketChat } from "./useWebSocket";
import { useChatStore } from "@/stores";
import * as sessionsApi from "@/api/sessions";

// Mock the sessions API
vi.mock("@/api/sessions", () => ({
  addSessionMessage: vi.fn().mockResolvedValue({}),
}));

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  sentMessages: string[] = [];
  readyState = 1; // OPEN

  constructor(_url: string) {
    MockWebSocket.instances.push(this);
    // Simulate async connection
    setTimeout(() => this.onopen?.(), 0);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.onclose?.();
  }

  // Test helper to simulate server messages
  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

// Replace global WebSocket
const originalWebSocket = global.WebSocket;
beforeEach(() => {
  MockWebSocket.instances = [];
  global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
  vi.clearAllMocks();
});
afterEach(() => {
  global.WebSocket = originalWebSocket;
});

// Test wrapper with QueryClient
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useWebSocketChat", () => {
  it("should NOT call addSessionMessage API - backend handles message persistence", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    // Send a message
    await act(async () => {
      result.current.sendMessage("Hello");
    });

    // Wait for WebSocket to connect
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    // Simulate the full streaming response
    await act(async () => {
      ws.simulateMessage({ type: "token", content: "Hi " });
      ws.simulateMessage({ type: "token", content: "there!" });
      ws.simulateMessage({ type: "sources", data: [] });
      ws.simulateMessage({
        type: "done",
        content: "Hi there!",
        confidence_level: "normal",
      });
    });

    // The bug: addSessionMessage should NOT be called at all
    // Backend WebSocket handler already saves messages
    // If this fails, we have duplicate message saving
    expect(sessionsApi.addSessionMessage).not.toHaveBeenCalled();
  });

  it("should send prompt over WebSocket", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      result.current.sendMessage("Test prompt");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    await waitFor(() => {
      expect(ws.sentMessages).toContainEqual(
        JSON.stringify({ prompt: "Test prompt" })
      );
    });
  });

  it("should handle status messages and update store", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    // Reset store
    await act(async () => {
      useChatStore.getState().reset();
    });

    await act(async () => {
      result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    // Simulate status updates
    await act(async () => {
      ws.simulateMessage({ type: "status", status: "retrieving" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("retrieving");

    await act(async () => {
      ws.simulateMessage({ type: "status", status: "thinking" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("thinking");

    await act(async () => {
      ws.simulateMessage({ type: "status", status: "generating" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("generating");
  });

  it("should handle thinking messages and accumulate content", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    // Reset store
    await act(async () => {
      useChatStore.getState().reset();
    });

    await act(async () => {
      result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    // Simulate thinking tokens
    await act(async () => {
      ws.simulateMessage({ type: "thinking", content: "Let me " });
      ws.simulateMessage({ type: "thinking", content: "analyze " });
      ws.simulateMessage({ type: "thinking", content: "this." });
    });

    expect(useChatStore.getState().streamingThinking).toBe(
      "Let me analyze this."
    );
  });

  it("should clear status on streaming finish", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    // Reset store
    await act(async () => {
      useChatStore.getState().reset();
    });

    await act(async () => {
      result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    // Simulate full streaming flow
    await act(async () => {
      ws.simulateMessage({ type: "status", status: "retrieving" });
      ws.simulateMessage({ type: "status", status: "generating" });
      ws.simulateMessage({ type: "token", content: "Response" });
      ws.simulateMessage({
        type: "done",
        content: "Response",
        confidence_level: "normal",
      });
    });

    // Status should be cleared after done
    expect(useChatStore.getState().pipelineStatus).toBe(null);
    expect(useChatStore.getState().isStreaming).toBe(false);
  });
});
