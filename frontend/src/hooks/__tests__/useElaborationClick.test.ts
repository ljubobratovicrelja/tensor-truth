import { describe, expect, it } from "vitest";
import { elaborationAtPoint } from "../useElaborationClick";
import { collectElaborationSpecs } from "../useElaborationHighlights";
import type { MessageResponse } from "@/api/types";

function mount(html: string): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = html;
  document.body.appendChild(container);
  return container;
}

function specsFor(selectedText: string, occurrenceIndex = 0, content = "why") {
  const messages: MessageResponse[] = [
    { role: "user", content: "q" },
    {
      role: "assistant",
      content: "answer",
      elaborations: [
        {
          id: "elab-1",
          selected_text: selectedText,
          occurrence_index: occurrenceIndex,
          content,
          created_at: "2026-10-28T20:00:00+00:00",
        },
      ],
    },
  ];
  return collectElaborationSpecs(messages);
}

function textNode(container: HTMLElement): Text {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const node = walker.nextNode() as Text | null;
  if (!node) throw new Error("fixture has no text node");
  return node;
}

describe("elaborationAtPoint", () => {
  it("returns the stored elaboration when the caret is inside its range", () => {
    const container = mount("<p>alpha beta gamma</p>");
    const node = textNode(container);

    const hit = elaborationAtPoint(specsFor("beta"), container, node, 8);

    expect(hit?.id).toBe("elab-1");
    expect(hit?.content).toBe("why");
    expect(hit?.anchor).toEqual({
      messageIndex: 1,
      selectedText: "beta",
      occurrenceIndex: 0,
    });
  });

  it("resolves the requested occurrence, not just the first match", () => {
    const container = mount("<p>alpha beta alpha beta</p>");
    const node = textNode(container);

    const first = elaborationAtPoint(specsFor("beta", 0), container, node, 8);
    const second = elaborationAtPoint(specsFor("beta", 1), container, node, 19);

    expect(first?.anchor.occurrenceIndex).toBe(0);
    expect(second?.anchor.occurrenceIndex).toBe(1);
    // The first highlight's range does not cover the second occurrence.
    expect(elaborationAtPoint(specsFor("beta", 0), container, node, 19)).toBeNull();
  });

  it("returns null outside the passage and when nothing is stored", () => {
    const container = mount("<p>alpha beta</p>");
    const node = textNode(container);

    expect(elaborationAtPoint(specsFor("beta"), container, node, 1)).toBeNull();
    expect(elaborationAtPoint([], container, node, 8)).toBeNull();
  });

  it("matches passages that span block boundaries", () => {
    const container = mount("<p>first block</p><p>second block</p>");
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodes: Text[] = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n as Text);

    // The selection's own toString() inserts a newline; the matcher tolerates it.
    const specs = specsFor("block\nsecond");

    expect(elaborationAtPoint(specs, container, nodes[0], 7)?.id).toBe("elab-1");
    expect(elaborationAtPoint(specs, container, nodes[1], 3)).not.toBeNull();
  });
});
