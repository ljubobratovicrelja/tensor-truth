import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useElaboration } from "../useElaboration";
import { useElaborationStore } from "@/stores/elaborationStore";
import { cancelRun, startRun } from "@/api/runs";
import { QUERY_KEYS } from "@/lib/constants";

const { FakeWebSocket } = vi.hoisted(() => {
  class FakeWebSocket {
    static instances: FakeWebSocket[] = [];
    url: string;
    sent: string[] = [];
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(url: string) {
      this.url = url;
      FakeWebSocket.instances.push(this);
    }

    send(data: string) {
      this.sent.push(data);
    }

    close() {}
  }
  return { FakeWebSocket };
});

vi.mock("@/api/client", () => ({
  createWebSocket: (sessionId: string) => new FakeWebSocket(sessionId),
}));

vi.mock("@/api/runs", () => ({
  startRun: vi.fn(),
  cancelRun: vi.fn(),
  getRunEvents: vi.fn(),
}));

const anchor = {
  messageIndex: 1,
  selectedText: "the passage",
  occurrenceIndex: 0,
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function mountHook() {
  return renderHook(() => useElaboration("session-1"), { wrapper });
}

describe("useElaboration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    FakeWebSocket.instances = [];
    useElaborationStore.getState().reset();
  });

  it("starts an elaboration run and attaches to it from seq 0", async () => {
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-1" });
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });

    expect(startRun).toHaveBeenCalledWith("session-1", "the passage", undefined, {
      kind: "elaboration",
      message_index: 1,
      selected_text: "the passage",
      occurrence_index: 0,
    });

    const ws = FakeWebSocket.instances[0];
    expect(ws.url).toBe("session-1");

    act(() => ws.onopen?.());
    expect(JSON.parse(ws.sent[0])).toEqual({
      type: "attach",
      run_id: "run-1",
      after: 0,
    });
    expect(useElaborationStore.getState().runId).toBe("run-1");
  });

  it("streams tokens and finishes with the persisted id", async () => {
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-1" });
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());

    act(() =>
      ws.onmessage?.({ data: JSON.stringify({ type: "token", content: "Expl", seq: 1 }) })
    );
    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({ type: "token", content: "ained", seq: 2 }),
      })
    );

    expect(useElaborationStore.getState().content).toBe("Explained");

    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({
          type: "done",
          content: "Explained.",
          confidence_level: "normal",
          elaboration_id: "elab-1",
          seq: 3,
        }),
      })
    );

    const state = useElaborationStore.getState();
    expect(state.isStreaming).toBe(false);
    expect(state.content).toBe("Explained.");
    expect(state.activeElaborationId).toBe("elab-1");
  });

  it("drops duplicate seqs", async () => {
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-1" });
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());

    act(() =>
      ws.onmessage?.({ data: JSON.stringify({ type: "token", content: "a", seq: 5 }) })
    );
    act(() =>
      ws.onmessage?.({ data: JSON.stringify({ type: "token", content: "b", seq: 5 }) })
    );

    expect(useElaborationStore.getState().content).toBe("a");
  });

  it("surfaces an error event without touching the chat store", async () => {
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-1" });
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({ type: "error", detail: "model unavailable", seq: 2 }),
      })
    );

    const state = useElaborationStore.getState();
    expect(state.error).toBe("model unavailable");
    expect(state.isStreaming).toBe(false);
  });

  it("shows a start failure in the panel", async () => {
    vi.mocked(startRun).mockRejectedValue(new Error("409 run already active"));
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });

    expect(useElaborationStore.getState().error).toBe("409 run already active");
    expect(useElaborationStore.getState().open).toBe(true);
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("cancels the run and stops streaming", async () => {
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-1" });
    vi.mocked(cancelRun).mockResolvedValue({
      run_id: "run-1",
      session_id: "session-1",
      prompt: "the passage",
      status: "cancelled",
    });
    const { result } = mountHook();

    await act(async () => {
      await result.current.start(anchor);
    });
    act(() => FakeWebSocket.instances[0].onopen?.());

    result.current.cancel();

    expect(cancelRun).toHaveBeenCalledWith("run-1");
    expect(useElaborationStore.getState().isStreaming).toBe(false);
  });

  it("posts a follow-up for the active elaboration and streams into reply", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-2" });
    const invalidateSpy = vi.spyOn(QueryClient.prototype, "invalidateQueries");
    const { result } = mountHook();

    await act(async () => {
      await result.current.followUp("  why?  ");
    });

    expect(startRun).toHaveBeenCalledWith("session-1", "why?", undefined, {
      kind: "elaboration_followup",
      message_index: 1,
      elaboration_id: "elab-1",
    });

    let state = useElaborationStore.getState();
    expect(state.isReplying).toBe(true);
    expect(state.thread).toEqual([{ role: "user", content: "why?" }]);

    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    expect(JSON.parse(ws.sent[0])).toEqual({
      type: "attach",
      run_id: "run-2",
      after: 0,
    });

    act(() =>
      ws.onmessage?.({ data: JSON.stringify({ type: "token", content: "Bec", seq: 1 }) })
    );
    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({ type: "token", content: "ause.", seq: 2 }),
      })
    );

    state = useElaborationStore.getState();
    expect(state.reply).toBe("Because.");
    // The original explanation is never overwritten by a follow-up.
    expect(state.content).toBe("Explained.");

    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({
          type: "done",
          content: "Because.",
          confidence_level: "normal",
          elaboration_id: "elab-1",
          seq: 3,
        }),
      })
    );

    state = useElaborationStore.getState();
    expect(state.isReplying).toBe(false);
    expect(state.reply).toBe("");
    expect(state.thread).toEqual([
      { role: "user", content: "why?" },
      { role: "assistant", content: "Because." },
    ]);
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: QUERY_KEYS.messages("session-1"),
    });
    invalidateSpy.mockRestore();
  });

  it("surfaces a follow-up error in the thread", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-2" });
    const { result } = mountHook();

    await act(async () => {
      await result.current.followUp("why?");
    });
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    act(() =>
      ws.onmessage?.({
        data: JSON.stringify({ type: "error", detail: "model unavailable", seq: 2 }),
      })
    );

    const state = useElaborationStore.getState();
    expect(state.followUpError).toBe("model unavailable");
    expect(state.isReplying).toBe(false);
    expect(state.error).toBeNull();
  });

  it("does not send a follow-up without a persisted elaboration", async () => {
    useElaborationStore.getState().openFor(anchor);
    const { result } = mountHook();

    await act(async () => {
      await result.current.followUp("why?");
    });

    expect(startRun).not.toHaveBeenCalled();
    expect(useElaborationStore.getState().thread).toEqual([]);
  });

  it("cancels the live follow-up run", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    vi.mocked(startRun).mockResolvedValue({ run_id: "run-2" });
    vi.mocked(cancelRun).mockResolvedValue({
      run_id: "run-2",
      session_id: "session-1",
      prompt: "why?",
      status: "cancelled",
    });
    const { result } = mountHook();

    await act(async () => {
      await result.current.followUp("why?");
    });
    act(() => FakeWebSocket.instances[0].onopen?.());

    result.current.cancel();

    expect(cancelRun).toHaveBeenCalledWith("run-2");
    const state = useElaborationStore.getState();
    expect(state.isReplying).toBe(false);
    expect(state.reply).toBe("");
    expect(state.thread).toEqual([{ role: "user", content: "why?" }]);
  });
});
