import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWebSocketChat } from "./useWebSocket";
import { useChatStore } from "@/stores";
import * as runsApi from "@/api/runs";

vi.mock("@/api/sessions", () => ({
  addSessionMessage: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/api/runs", () => ({
  startRun: vi.fn(),
  getActiveRun: vi.fn(),
  getRunEvents: vi.fn(),
  cancelRun: vi.fn(),
}));

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  sentMessages: string[] = [];
  readyState = 1;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(_url: string) {
    MockWebSocket.instances.push(this);
    Promise.resolve().then(() => this.onopen?.());
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

const originalWebSocket = global.WebSocket;
beforeEach(() => {
  MockWebSocket.instances = [];
  global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
  vi.clearAllMocks();
  useChatStore.getState().reset();
  vi.mocked(runsApi.getRunEvents).mockResolvedValue({
    run_id: "run-9",
    session_id: "test-session",
    prompt: "Hello",
    status: "running",
    events: [],
  });
});
afterEach(() => {
  global.WebSocket = originalWebSocket;
});

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

const activeRun = {
  run_id: "run-9",
  session_id: "test-session",
  prompt: "Hello",
  status: "running" as const,
};

const replayedEvents = [
  { seq: 1, type: "status" as const, status: "retrieving" as const },
  { seq: 2, type: "token" as const, content: "Hel" },
  { seq: 3, type: "token" as const, content: "lo" },
];

describe("useWebSocketChat resumable runs", () => {
  it("hydrates a running run on mount and attaches from the replay cursor", async () => {
    vi.mocked(runsApi.getActiveRun).mockResolvedValue({
      ...activeRun,
      events: replayedEvents,
    });

    renderHook(() => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }), {
      wrapper: createWrapper(),
    });

    // Store is hydrated from the server-owned run, not from local state.
    await waitFor(() => {
      expect(useChatStore.getState().isStreaming).toBe(true);
      expect(useChatStore.getState().streamingContent).toBe("Hello");
      expect(useChatStore.getState().pendingUserMessage).toBe("Hello");
    });
    expect(runsApi.getActiveRun).toHaveBeenCalledWith("test-session");

    // Then attach live from seq 3 so the server replays nothing already applied.
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBe(1);
    });
    const ws = MockWebSocket.instances[0];
    await waitFor(() => {
      expect(ws.sentMessages).toContainEqual(attachFrame("run-9", 3));
    });

    // No initial POST: the run already exists server-side.
    expect(runsApi.startRun).not.toHaveBeenCalled();
  });

  it("drops duplicate seq on re-attach and resumes from the cursor", async () => {
    vi.mocked(runsApi.getActiveRun).mockResolvedValue({
      ...activeRun,
      events: replayedEvents,
    });

    renderHook(() => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(useChatStore.getState().streamingContent).toBe("Hello");
    });

    const ws = MockWebSocket.instances[0];
    await waitFor(() => {
      expect(ws.sentMessages).toContainEqual(attachFrame("run-9", 3));
    });

    // A duplicate delivery of seq 3 is dropped; seq 4 is applied once.
    await act(async () => {
      ws.simulateMessage({ seq: 3, type: "token", content: "lo" });
      ws.simulateMessage({ seq: 4, type: "token", content: "!" });
    });
    expect(useChatStore.getState().streamingContent).toBe("Hello!");

    // Socket drops mid-run: no error card, and a new socket attaches at seq 4.
    await act(async () => {
      ws.close();
    });
    expect(useChatStore.getState().error).toBe(null);
    expect(useChatStore.getState().isStreaming).toBe(true);

    await waitFor(
      () => {
        expect(MockWebSocket.instances.length).toBe(2);
      },
      { timeout: 2000 }
    );
    const resumed = MockWebSocket.instances[1];
    await waitFor(() => {
      expect(resumed.sentMessages).toContainEqual(attachFrame("run-9", 4));
    });

    // Replayed duplicates after re-attach are no-ops; only seq 5 lands.
    await act(async () => {
      resumed.simulateMessage({ seq: 4, type: "token", content: "!" });
      resumed.simulateMessage({ seq: 5, type: "token", content: "?" });
    });
    expect(useChatStore.getState().streamingContent).toBe("Hello!?");

    await act(async () => {
      resumed.simulateMessage({
        seq: 6,
        type: "done",
        content: "Hello!?",
        confidence_level: "normal",
      });
    });
    expect(useChatStore.getState().isStreaming).toBe(false);
    expect(useChatStore.getState().streamingContent).toBe("Hello!?");
  });

  it("does not hydrate an elaboration run as a chat answer", async () => {
    vi.mocked(runsApi.getActiveRun).mockResolvedValue({
      ...activeRun,
      kind: "elaboration",
      events: replayedEvents,
    });

    renderHook(() => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(runsApi.getActiveRun).toHaveBeenCalledWith("test-session");
    });
    // The passage explanation belongs to the elaboration panel, never here.
    expect(MockWebSocket.instances.length).toBe(0);
    expect(useChatStore.getState().isStreaming).toBe(false);
    expect(useChatStore.getState().streamingContent).toBe("");
  });

  it("does not hydrate when the session has no active run", async () => {
    vi.mocked(runsApi.getActiveRun).mockResolvedValue(null);

    renderHook(() => useWebSocketChat({ sessionId: "test-session", onError: vi.fn() }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(runsApi.getActiveRun).toHaveBeenCalledWith("test-session");
    });
    expect(MockWebSocket.instances.length).toBe(0);
    expect(useChatStore.getState().isStreaming).toBe(false);
  });
});
