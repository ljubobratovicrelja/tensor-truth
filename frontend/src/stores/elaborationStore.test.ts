import { beforeEach, describe, expect, it } from "vitest";
import { useElaborationStore } from "./elaborationStore";

const anchor = {
  messageIndex: 3,
  selectedText: "the passage",
  occurrenceIndex: 1,
};

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

  it("close clears transient state but keeps the anchor", () => {
    const store = useElaborationStore.getState();
    store.openFor(anchor);
    store.appendToken("streamed");
    store.close();

    const state = useElaborationStore.getState();
    expect(state.open).toBe(false);
    expect(state.isStreaming).toBe(false);
    expect(state.anchor).toEqual(anchor);
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
});
