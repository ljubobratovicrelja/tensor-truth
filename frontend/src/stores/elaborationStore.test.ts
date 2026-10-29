import { beforeEach, describe, expect, it } from "vitest";
import { useElaborationStore } from "./elaborationStore";

const anchor = {
  messageIndex: 3,
  selectedText: "the passage",
  occurrenceIndex: 1,
};

const storedThread = [
  { role: "user" as const, content: "why?", created_at: "2026-10-28T20:00:00Z" },
  {
    role: "assistant" as const,
    content: "Because of X.",
    created_at: "2026-10-28T20:00:01Z",
  },
];

describe("elaborationStore", () => {
  beforeEach(() => {
    useElaborationStore.getState().reset();
  });

  it("opens for an anchor in a clean streaming state", () => {
    useElaborationStore.getState().openFor(anchor);

    const state = useElaborationStore.getState();
    expect(state.open).toBe(true);
    expect(state.anchor).toEqual(anchor);
    expect(state.isStreaming).toBe(true);
    expect(state.content).toBe("");
    expect(state.error).toBeNull();
    expect(state.activeElaborationId).toBeNull();
    expect(state.thread).toEqual([]);
    expect(state.isReplying).toBe(false);
  });

  it("replaces the previous anchor and content on a new elaboration", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("first answer");

    useElaborationStore.getState().openFor({
      messageIndex: 5,
      selectedText: "another",
      occurrenceIndex: 0,
    });

    const state = useElaborationStore.getState();
    expect(state.anchor?.messageIndex).toBe(5);
    expect(state.content).toBe("");
    expect(state.isStreaming).toBe(true);
  });

  it("accumulates tokens and thinking separately", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("Hello ");
    store.appendToken("world");
    store.appendThinking("step 1");
    store.appendThinking("; step 2");

    const state = useElaborationStore.getState();
    expect(state.content).toBe("Hello world");
    expect(state.thinking).toBe("step 1; step 2");
  });

  it("finishes with the persisted id and stops streaming", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("partial");
    store.setRunId("run-1");
    store.finish("Full explanation", "elab-9");

    const state = useElaborationStore.getState();
    expect(state.content).toBe("Full explanation");
    expect(state.activeElaborationId).toBe("elab-9");
    expect(state.isStreaming).toBe(false);
    expect(state.thinking).toBe("");
    expect(state.runId).toBeNull();
    expect(state.open).toBe(true);
    expect(state.anchor).toEqual(anchor);
  });

  it("records an error without closing the panel", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.setRunId("run-1");
    store.setError("model unavailable");

    const state = useElaborationStore.getState();
    expect(state.error).toBe("model unavailable");
    expect(state.isStreaming).toBe(false);
    expect(state.open).toBe(true);
  });

  it("close only hides the panel, keeping the live run and text", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("streamed");
    store.close();

    const state = useElaborationStore.getState();
    expect(state.open).toBe(false);
    // A background run must survive a click away.
    expect(state.isStreaming).toBe(true);
    expect(state.content).toBe("streamed");
    expect(state.anchor).toEqual(anchor);
  });

  it("reopen shows the same in-flight elaboration", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("partial");
    store.close();
    useElaborationStore.getState().reopen();

    const state = useElaborationStore.getState();
    expect(state.open).toBe(true);
    expect(state.content).toBe("partial");
    expect(state.isStreaming).toBe(true);
  });

  it("reset returns to the initial state", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("streamed");
    store.reset();

    const state = useElaborationStore.getState();
    expect(state.open).toBe(false);
    expect(state.anchor).toBeNull();
    expect(state.content).toBe("");
    expect(state.activeElaborationId).toBeNull();
  });

  it("opens a stored elaboration with its thread", () => {
    useElaborationStore
      .getState()
      .openStored(anchor, "elab-1", "Explained.", storedThread);

    const state = useElaborationStore.getState();
    expect(state.open).toBe(true);
    expect(state.activeElaborationId).toBe("elab-1");
    expect(state.content).toBe("Explained.");
    expect(state.thread).toEqual(storedThread);
    // A copy, so a later server payload cannot mutate panel state in place.
    expect(state.thread).not.toBe(storedThread);
    expect(state.isStreaming).toBe(false);
  });

  it("opens a stored elaboration without a thread", () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    expect(useElaborationStore.getState().thread).toEqual([]);
  });

  it("openFor clears a previous thread and reply state", () => {
    useElaborationStore
      .getState()
      .openStored(anchor, "elab-1", "Explained.", storedThread);
    useElaborationStore.getState().startFollowUp("more?");
    useElaborationStore.getState().appendReplyToken("partial");

    useElaborationStore.getState().openFor(anchor);
    const fresh = useElaborationStore.getState();
    expect(fresh.thread).toEqual([]);
    expect(fresh.reply).toBe("");
    expect(fresh.isReplying).toBe(false);
    expect(fresh.isStreaming).toBe(true);
  });

  it("startFollowUp appends the user turn optimistically", () => {
    useElaborationStore
      .getState()
      .openStored(anchor, "elab-1", "Explained.", storedThread);
    useElaborationStore.getState().startFollowUp("  and then?  ");

    const state = useElaborationStore.getState();
    expect(state.isReplying).toBe(true);
    expect(state.reply).toBe("");
    expect(state.followUpError).toBeNull();
    expect(state.thread).toHaveLength(3);
    expect(state.thread[2]).toEqual({ role: "user", content: "  and then?  " });
  });

  it("appendReplyToken grows only the reply buffer, not the explanation", () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    useElaborationStore.getState().startFollowUp("why?");

    useElaborationStore.getState().appendReplyToken("Bec");
    useElaborationStore.getState().appendReplyToken("ause.");

    expect(useElaborationStore.getState().reply).toBe("Because.");
    expect(useElaborationStore.getState().content).toBe("Explained.");
    expect(useElaborationStore.getState().thread).toHaveLength(1);
  });

  it("finishFollowUp moves the answer into the thread and stops replying", () => {
    useElaborationStore
      .getState()
      .openStored(anchor, "elab-1", "Explained.", storedThread);
    useElaborationStore.getState().startFollowUp("why?");
    useElaborationStore.getState().appendReplyToken("partial");

    useElaborationStore.getState().finishFollowUp("Because.");

    const state = useElaborationStore.getState();
    expect(state.isReplying).toBe(false);
    expect(state.reply).toBe("");
    expect(state.content).toBe("Explained.");
    expect(state.thread.at(-1)).toEqual({ role: "assistant", content: "Because." });
    expect(state.thread.map((m) => m.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
  });

  it("failFollowUp drops the partial answer and keeps the question", () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    useElaborationStore.getState().startFollowUp("why?");
    useElaborationStore.getState().appendReplyToken("partial");

    useElaborationStore.getState().failFollowUp("model unavailable");

    const state = useElaborationStore.getState();
    expect(state.isReplying).toBe(false);
    expect(state.reply).toBe("");
    expect(state.followUpError).toBe("model unavailable");
    expect(state.thread).toEqual([{ role: "user", content: "why?" }]);
  });
});
