import { describe, expect, it } from "vitest";
import { collectElaborationSpecs, resolveTextRange } from "../useElaborationHighlights";
import type { MessageResponse } from "@/api/types";

function mount(html: string): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = html;
  document.body.appendChild(container);
  return container;
}

describe("resolveTextRange", () => {
  it("resolves the requested occurrence inside one text node", () => {
    const container = mount("<p>alpha beta alpha</p>");

    const first = resolveTextRange(container, "alpha", 0);
    const second = resolveTextRange(container, "alpha", 1);

    expect(first?.toString()).toBe("alpha");
    expect(second?.toString()).toBe("alpha");
    expect(first?.startOffset).toBe(0);
    expect(second?.startOffset).toBe(11);
  });

  it("resolves a passage spanning inline elements", () => {
    const container = mount("<p>foo <strong>bar baz</strong> qux</p>");

    const range = resolveTextRange(container, "bar baz", 0);

    expect(range?.toString()).toBe("bar baz");
    expect(range?.startContainer.parentElement?.tagName).toBe("STRONG");
  });

  it("resolves a passage that crosses element boundaries", () => {
    const container = mount("<p>foo <strong>bar</strong> baz</p>");

    const range = resolveTextRange(container, "foo bar", 0);

    expect(range?.toString()).toBe("foo bar");
  });

  it("returns null when the passage is no longer present", () => {
    const container = mount("<p>alpha</p>");

    expect(resolveTextRange(container, "missing", 0)).toBeNull();
    expect(resolveTextRange(container, "alpha", 3)).toBeNull();
    expect(resolveTextRange(container, "", 0)).toBeNull();
  });
});

describe("collectElaborationSpecs", () => {
  const elaboration = (selectedText: string, occurrenceIndex = 0) => ({
    id: "e1",
    selected_text: selectedText,
    occurrence_index: occurrenceIndex,
    content: "explanation",
    created_at: "2026-10-28T20:00:00+00:00",
  });

  it("flattens assistant elaborations with their message index", () => {
    const messages: MessageResponse[] = [
      { role: "user", content: "question" },
      {
        role: "assistant",
        content: "answer",
        elaborations: [elaboration("passage", 2)],
      },
      {
        role: "assistant",
        content: "second answer",
        elaborations: [elaboration("other")],
      },
    ];

    expect(collectElaborationSpecs(messages)).toEqual([
      {
        id: "e1",
        messageIndex: 1,
        selectedText: "passage",
        occurrenceIndex: 2,
        content: "explanation",
      },
      {
        id: "e1",
        messageIndex: 2,
        selectedText: "other",
        occurrenceIndex: 0,
        content: "explanation",
      },
    ]);
  });

  it("skips user messages, missing fields and empty selections", () => {
    const messages: MessageResponse[] = [
      { role: "user", content: "x", elaborations: [elaboration("nope")] },
      { role: "assistant", content: "no elaborations" },
      { role: "assistant", content: "a", elaborations: [elaboration("")] },
      { role: "assistant", content: "b", elaborations: null },
    ];

    expect(collectElaborationSpecs(messages)).toEqual([]);
    expect(collectElaborationSpecs(undefined)).toEqual([]);
  });
});
