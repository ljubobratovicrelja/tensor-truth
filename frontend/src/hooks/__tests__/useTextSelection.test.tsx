import { describe, expect, it } from "vitest";
import { act, render } from "@testing-library/react";
import {
  occurrenceIndexFor,
  resolveAnchorFromRange,
  textOffsetFromContainer,
  useTextSelection,
} from "../useTextSelection";
import { resolveTextRange } from "../useElaborationHighlights";

function selectText(node: Text, start: number, end: number): Range {
  const range = document.createRange();
  range.setStart(node, start);
  range.setEnd(node, end);
  return range;
}

function textNode(container: HTMLElement): Text {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const node = walker.nextNode() as Text | null;
  if (!node) throw new Error("fixture has no text node");
  return node;
}

describe("occurrenceIndexFor", () => {
  it("counts whole occurrences before the offset", () => {
    const text = "alpha beta alpha gamma alpha";
    expect(occurrenceIndexFor(text, 0, "alpha")).toBe(0);
    expect(occurrenceIndexFor(text, 11, "alpha")).toBe(1);
    expect(occurrenceIndexFor(text, text.length, "alpha")).toBe(3);
  });

  it("does not count an occurrence that starts at the offset", () => {
    expect(occurrenceIndexFor("alpha", 0, "alpha")).toBe(0);
  });

  it("handles repeated adjacent text without overlapping", () => {
    expect(occurrenceIndexFor("aaaa", 3, "aa")).toBe(2);
  });

  it("returns 0 for empty needles and clamps out-of-range offsets", () => {
    expect(occurrenceIndexFor("anything", 5, "")).toBe(0);
    expect(occurrenceIndexFor("abc", -10, "a")).toBe(0);
    expect(occurrenceIndexFor("abc", 99, "a")).toBe(1);
  });
});

describe("resolveAnchorFromRange", () => {
  it("computes the occurrence index of the second identical passage", () => {
    const { container } = render(
      <div data-chat-message="2" data-role="assistant">
        <div data-message-content>
          <p>alpha beta alpha</p>
        </div>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const node = textNode(content);

    const anchor = resolveAnchorFromRange(selectText(node, 11, 16));

    expect(anchor).toEqual({
      messageIndex: 2,
      selectedText: "alpha",
      occurrenceIndex: 1,
    });
  });

  it("trims surrounding whitespace from the selected text", () => {
    const { container } = render(
      <div data-chat-message="0" data-role="assistant">
        <div data-message-content>
          <p> alpha beta</p>
        </div>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const node = textNode(content);

    // Selecting the leading spaces plus the word must yield the bare word, and
    // it must still be the first occurrence.
    const anchor = resolveAnchorFromRange(selectText(node, 0, 7));

    expect(anchor).toEqual({
      messageIndex: 0,
      selectedText: "alpha",
      occurrenceIndex: 0,
    });
  });

  it("spans inline elements (markdown formatting)", () => {
    const { container } = render(
      <div data-chat-message="4" data-role="assistant">
        <div data-message-content>
          <p>
            foo <strong>bar baz</strong> qux
          </p>
        </div>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
    const nodes: Text[] = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n as Text);
    // nodes: "foo ", "bar baz", " qux"
    const range = document.createRange();
    range.setStart(nodes[0], 1);
    range.setEnd(nodes[1], 3);

    const anchor = resolveAnchorFromRange(range);

    expect(anchor?.selectedText).toBe("oo bar");
    expect(anchor?.occurrenceIndex).toBe(0);
  });

  it("anchors a selection that crosses block elements", () => {
    const { container } = render(
      <div data-chat-message="3" data-role="assistant">
        <div data-message-content>
          <p>first block</p>
          <p>second block</p>
        </div>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
    const nodes: Text[] = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n as Text);

    const range = document.createRange();
    range.setStart(nodes[0], 6);
    range.setEnd(nodes[1], 6);

    const anchor = resolveAnchorFromRange(range);

    expect(anchor?.messageIndex).toBe(3);
    expect(anchor?.selectedText).toContain("block");
    expect(anchor?.selectedText).toContain("second");
    expect(anchor?.occurrenceIndex).toBe(0);
    // The persisted anchor must resolve back to a highlight range.
    const resolved = anchor
      ? resolveTextRange(content, anchor.selectedText, anchor.occurrenceIndex)
      : null;
    expect(resolved).not.toBeNull();
  });

  it("ignores user messages", () => {
    const { container } = render(
      <div data-chat-message="1" data-role="user">
        <div data-message-content>
          <p>user text</p>
        </div>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const node = textNode(content);

    expect(resolveAnchorFromRange(selectText(node, 0, 4))).toBeNull();
  });

  it("ignores selections that cross two messages", () => {
    const { container } = render(
      <div>
        <div data-chat-message="0" data-role="assistant">
          <div data-message-content>
            <p>first answer</p>
          </div>
        </div>
        <div data-chat-message="1" data-role="assistant">
          <div data-message-content>
            <p>second answer</p>
          </div>
        </div>
      </div>
    );
    const first = container.querySelectorAll<HTMLElement>("[data-message-content]")[0];
    const second = container.querySelectorAll<HTMLElement>("[data-message-content]")[1];
    const range = document.createRange();
    range.setStart(textNode(first), 2);
    range.setEnd(textNode(second), 4);

    expect(resolveAnchorFromRange(range)).toBeNull();
  });

  it("ignores selections outside the message content wrapper", () => {
    const { container } = render(
      <div data-chat-message="0" data-role="assistant">
        <div data-message-content>
          <p>answer</p>
        </div>
        <span data-testid="footer">copy</span>
      </div>
    );
    const footer = container.querySelector<HTMLElement>("[data-testid='footer']")!;

    expect(
      resolveAnchorFromRange(selectText(footer.firstChild as Text, 0, 4))
    ).toBeNull();
  });
});

describe("textOffsetFromContainer", () => {
  it("counts rendered characters up to the range boundary", () => {
    const { container } = render(
      <div data-message-content>
        <p>abc</p>
        <p>def</p>
      </div>
    );
    const content = container.querySelector<HTMLElement>("[data-message-content]")!;
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
    const first = walker.nextNode() as Text;
    const second = walker.nextNode() as Text;

    expect(textOffsetFromContainer(content, first, 2)).toBe(2);
    expect(textOffsetFromContainer(content, second, 1)).toBe(4);
  });

  it("returns null for nodes outside the container", () => {
    const outside = document.createElement("span");
    outside.textContent = "outside";
    const container = document.createElement("div");

    expect(textOffsetFromContainer(container, outside.firstChild as Text, 0)).toBeNull();
  });
});

describe("useTextSelection", () => {
  function Harness() {
    const { selecting } = useTextSelection();
    return <div data-testid="flag">{selecting ? "selecting" : "idle"}</div>;
  }

  const flag = (container: HTMLElement) =>
    container.querySelector<HTMLElement>('[data-testid="flag"]')!.textContent;

  /** Pointer events live outside React, so flush them through `act`. */
  const fire = (target: EventTarget, event: Event) =>
    act(() => {
      target.dispatchEvent(event);
    });

  it("reports a drag only while the primary button is held", () => {
    const { container } = render(<Harness />);
    expect(flag(container)).toBe("idle");

    fire(document, new MouseEvent("pointerdown", { button: 0, bubbles: true }));
    expect(flag(container)).toBe("selecting");

    fire(document, new MouseEvent("pointerup", { button: 0, bubbles: true }));
    expect(flag(container)).toBe("idle");
  });

  it("ignores non-primary buttons", () => {
    const { container } = render(<Harness />);

    fire(document, new MouseEvent("pointerdown", { button: 2, bubbles: true }));
    expect(flag(container)).toBe("idle");
  });

  it("stays idle when the press lands on the elaboration trigger", () => {
    const { container } = render(<Harness />);
    const trigger = document.createElement("button");
    trigger.setAttribute("data-elaboration-trigger", "");
    document.body.appendChild(trigger);

    // A press on the trigger must not turn it inert before its click lands.
    fire(trigger, new MouseEvent("pointerdown", { button: 0, bubbles: true }));
    expect(flag(container)).toBe("idle");

    trigger.remove();
  });

  it("clears the drag state when the pointer is cancelled or the window blurs", () => {
    const { container } = render(<Harness />);
    fire(document, new MouseEvent("pointerdown", { button: 0, bubbles: true }));
    fire(document, new MouseEvent("pointercancel", { bubbles: true }));
    expect(flag(container)).toBe("idle");

    fire(document, new MouseEvent("pointerdown", { button: 0, bubbles: true }));
    fire(window, new Event("blur"));
    expect(flag(container)).toBe("idle");
  });
});
