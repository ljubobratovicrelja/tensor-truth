import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWebSocketChat } from "./useWebSocket";
import { useChatStore } from "@/stores";
import * as sessionsApi from "@/api/sessions";
import * as runsApi from "@/api/runs";

// Mock the sessions API
vi.mock("@/api/sessions", () => ({
  addSessionMessage: vi.fn().mockResolvedValue({}),
}));

// Mock the runs API (server-owned resumable runs)
vi.mock("@/api/runs", () => ({
  startRun: vi.fn(),
  getActiveRun: vi.fn(),
  getRunEvents: vi.fn(),
  cancelRun: vi.fn(),
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

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(_url: string) {
    MockWebSocket.instances.push(this);
    // Simulate async connection (microtask: resolved by any await in tests)
    Promise.resolve().then(() => this.onopen?.());
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.readyState = 3; // CLOSED
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
  useChatStore.getState().reset();
  vi.mocked(runsApi.startRun).mockResolvedValue({ run_id: "run-1" });
  vi.mocked(runsApi.getActiveRun).mockResolvedValue(null);
  vi.mocked(runsApi.getRunEvents).mockResolvedValue({
    run_id: "run-1",
    session_id: "test-session",
    prompt: "",
    status: "completed",
    events: [],
  });
  vi.mocked(runsApi.cancelRun).mockResolvedValue({
    run_id: "run-1",
    session_id: "test-session",
    prompt: "",
    status: "cancelled",
  });
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

function attachFrame(runId: string, after: number) {
  return JSON.stringify({ type: "attach", run_id: runId, after });
}

describe("useWebSocketChat", () => {
  it("should NOT call addSessionMessage API - backend handles message persistence", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Hello");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    await act(async () => {
      ws.simulateMessage({ seq: 1, type: "token", content: "Hi " });
      ws.simulateMessage({ seq: 2, type: "token", content: "there!" });
      ws.simulateMessage({ seq: 3, type: "sources", data: [] });
      ws.simulateMessage({
        seq: 4,
        type: "done",
        content: "Hi there!",
        confidence_level: "normal",
      });
    });

    expect(sessionsApi.addSessionMessage).not.toHaveBeenCalled();
  });

  it("should POST the run first, then attach with after: 0", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Test prompt");
    });

    expect(runsApi.startRun).toHaveBeenCalledWith(
      "test-session",
      "Test prompt",
      undefined
    );
    expect(runsApi.startRun).toHaveBeenCalledTimes(1);

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    await waitFor(() => {
      expect(ws.sentMessages).toContainEqual(attachFrame("run-1", 0));
    });

    // The prompt travels in the POST body, not as a bare socket message.
    expect(ws.sentMessages).not.toContainEqual(JSON.stringify({ prompt: "Test prompt" }));
  });

  it("should handle status messages and update store", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];
    let seq = 0;

    await act(async () => {
      ws.simulateMessage({ seq: ++seq, type: "status", status: "retrieving" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("retrieving");

    await act(async () => {
      ws.simulateMessage({ seq: ++seq, type: "status", status: "thinking" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("thinking");

    await act(async () => {
      ws.simulateMessage({ seq: ++seq, type: "status", status: "generating" });
    });

    expect(useChatStore.getState().pipelineStatus).toBe("generating");
  });

  it("should handle thinking messages and accumulate content", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    await act(async () => {
      ws.simulateMessage({ seq: 1, type: "thinking", content: "Let me " });
      ws.simulateMessage({ seq: 2, type: "thinking", content: "analyze " });
      ws.simulateMessage({ seq: 3, type: "thinking", content: "this." });
    });

    expect(useChatStore.getState().streamingThinking).toBe("Let me analyze this.");
  });

  it("should clear status on streaming finish", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    await act(async () => {
      ws.simulateMessage({ seq: 1, type: "status", status: "retrieving" });
      ws.simulateMessage({ seq: 2, type: "status", status: "generating" });
      ws.simulateMessage({ seq: 3, type: "token", content: "Response" });
      ws.simulateMessage({
        seq: 4,
        type: "done",
        content: "Response",
        confidence_level: "normal",
      });
    });

    expect(useChatStore.getState().pipelineStatus).toBe(null);
    expect(useChatStore.getState().isStreaming).toBe(false);
  });

  it("picks up the title event that lands after `done`", async () => {
    // The first-turn title is generated after the answer, so its event sits
    // behind `done` in the run log while the socket is already closing. The
    // sidebar used to keep the provisional title until an unrelated refetch.
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Fetch the BYOL paper");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    const ws = MockWebSocket.instances[0];

    vi.mocked(runsApi.getRunEvents).mockResolvedValue({
      run_id: "run-1",
      session_id: "test-session",
      prompt: "",
      status: "completed",
      events: [{ seq: 2, type: "title", title: "Study BYOL Paper" } as never],
    });

    await act(async () => {
      ws.simulateMessage({
        seq: 1,
        type: "done",
        content: "Answer",
        title_pending: true,
      });
    });

    await waitFor(
      () => {
        expect(runsApi.getRunEvents).toHaveBeenCalledWith("run-1", 1);
      },
      { timeout: 4000 }
    );
  });

  it("should POST the cancel endpoint when the user cancels", async () => {
    const { result } = renderHook(
      () => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }),
      { wrapper: createWrapper() }
    );

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });

    await act(async () => {
      result.current.cancelStreaming();
    });

    expect(runsApi.cancelRun).toHaveBeenCalledWith("run-1");
    expect(useChatStore.getState().isStreaming).toBe(false);
  });
});
