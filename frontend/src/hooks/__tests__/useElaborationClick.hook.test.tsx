import { afterEach, describe, expect, it } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useElaborationClick } from "../useElaborationClick";
import { useElaborationStore } from "@/stores/elaborationStore";
import type { MessageResponse } from "@/api/types";

const TEXT = "alpha beta gamma";

function mountMessage(): { content: HTMLElement; node: Text } {
  const root = document.createElement("div");
  root.setAttribute("data-chat-message", "1");
  root.setAttribute("data-role", "assistant");
  const content = document.createElement("div");
  content.setAttribute("data-message-content", "");
  const p = document.createElement("p");
  p.textContent = TEXT;
  content.appendChild(p);
  root.appendChild(content);
  document.body.appendChild(root);
  const node = p.firstChild as Text;
  return { content, node };
}

function messages(): MessageResponse[] {
  return [
    { role: "user", content: "q" },
    {
      role: "assistant",
      content: TEXT,
      elaborations: [
        {
          id: "elab-1",
          selected_text: "beta",
          occurrence_index: 0,
          content: "explanation",
          created_at: "2026-10-28T22:00:00+00:00",
          messages: [],
        },
      ],
    },
  ];
}

/**
 * jsdom has no layout, so `caretRangeFromPoint` is stubbed to point at a fixed
 * offset inside the fixture text (the "beta" token).
 */
function stubCaret(node: Text, offset: number) {
  const range = document.createRange();
  range.setStart(node, offset);
  range.setEnd(node, offset);
  (
    document as Document & { caretRangeFromPoint?: (x: number, y: number) => Range }
  ).caretRangeFromPoint = () => range;
}

afterEach(() => {
  document.body.innerHTML = "";
  delete (document as unknown as { caretRangeFromPoint?: unknown }).caretRangeFromPoint;
  useElaborationStore.getState().reset();
});

describe("useElaborationClick", () => {
  it("shows a hand cursor over an elaborated passage only", async () => {
    const { content, node } = mountMessage();
    stubCaret(node, 8); // inside "beta"
    renderHook(() => useElaborationClick(messages()));

    await act(async () => {
      content.dispatchEvent(
        new MouseEvent("mousemove", { bubbles: true, clientX: 10, clientY: 10 })
      );
      await new Promise((resolve) => setTimeout(resolve, 40));
    });

    await waitFor(() => expect(content.style.cursor).toBe("pointer"));

    // Outside the passage (offset 1, inside "alpha") the default cursor returns.
    stubCaret(node, 1);
    await act(async () => {
      content.dispatchEvent(
        new MouseEvent("mousemove", { bubbles: true, clientX: 10, clientY: 10 })
      );
      await new Promise((resolve) => setTimeout(resolve, 40));
    });

    await waitFor(() => expect(content.style.cursor).toBe(""));
  });

  it("opens the stored thread on click", async () => {
    const { content, node } = mountMessage();
    stubCaret(node, 8);
    renderHook(() => useElaborationClick(messages()));

    await act(async () => {
      content.dispatchEvent(
        new MouseEvent("click", { bubbles: true, clientX: 10, clientY: 10 })
      );
    });

    const state = useElaborationStore.getState();
    expect(state.open).toBe(true);
    expect(state.activeElaborationId).toBe("elab-1");
    expect(state.content).toBe("explanation");
    expect(state.anchor).toEqual({
      messageIndex: 1,
      selectedText: "beta",
      occurrenceIndex: 0,
    });
  });

  it("does not intercept clicks on links inside a passage", () => {
    const { content, node } = mountMessage();
    const link = document.createElement("a");
    link.href = "#x";
    link.textContent = "beta";
    content.querySelector("p")!.textContent = "";
    content.querySelector("p")!.appendChild(link);
    stubCaret(node, 0);
    renderHook(() => useElaborationClick(messages()));

    link.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    expect(useElaborationStore.getState().open).toBe(false);
  });
});
