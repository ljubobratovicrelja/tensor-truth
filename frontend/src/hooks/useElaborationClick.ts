import { useEffect } from "react";
import type { MessageResponse } from "@/api/types";
import { useElaborationStore } from "@/stores/elaborationStore";
import type { ElaborationAnchor } from "@/stores/elaborationStore";
import {
  collectElaborationSpecs,
  type ElaborationHighlightSpec,
} from "@/hooks/useElaborationHighlights";
import { rangeForOccurrence } from "@/lib/elaborationText";

/**
 * Which persisted elaboration covers `(node, offset)` inside `container`, if
 * any. Exported for tests: caret resolution itself needs a browser API, but the
 * hit test is pure DOM.
 */
export function elaborationAtPoint(
  specs: ElaborationHighlightSpec[],
  container: Node,
  node: Node,
  offset: number
): { anchor: ElaborationAnchor; id: string; content: string } | null {
  for (const spec of specs) {
    const range = rangeForOccurrence(container, spec.selectedText, spec.occurrenceIndex);
    if (!range) continue;
    if (!range.isPointInRange(node, offset)) continue;
    return {
      anchor: {
        messageIndex: spec.messageIndex,
        selectedText: spec.selectedText,
        occurrenceIndex: spec.occurrenceIndex,
      },
      id: spec.id,
      content: spec.content,
    };
  }
  return null;
}

interface CaretPoint {
  node: Node;
  offset: number;
}

function caretFromPoint(x: number, y: number): CaretPoint | null {
  const doc = document as Document & {
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
    caretPositionFromPoint?: (
      x: number,
      y: number
    ) => { offsetNode: Node; offset: number } | null;
  };

  if (typeof doc.caretRangeFromPoint === "function") {
    const range = doc.caretRangeFromPoint(x, y);
    return range ? { node: range.startContainer, offset: range.startOffset } : null;
  }
  if (typeof doc.caretPositionFromPoint === "function") {
    const position = doc.caretPositionFromPoint(x, y);
    return position ? { node: position.offsetNode, offset: position.offset } : null;
  }
  return null;
}

/**
 * Clicking a passage that was already explained re-opens its panel with the
 * stored answer instead of starting a new run. Distinct from selection: a
 * click produces a collapsed selection, so the trigger logic ignores it.
 */
export function useElaborationClick(messages: MessageResponse[] | undefined | null) {
  useEffect(() => {
    if (!messages || messages.length === 0) return;

    const handleClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest("[data-elaboration-panel]")) return;
      // Links and controls inside a passage keep their own behaviour.
      if (target.closest("a, button, summary, [role='button']")) return;

      // A real text selection is the trigger's job, not a stored elaboration's.
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;

      const content = target.closest<HTMLElement>("[data-message-content]");
      const bubble = content?.closest<HTMLElement>(
        `[data-chat-message][data-role="assistant"]`
      );
      if (!content || !bubble) return;

      const messageIndex = Number.parseInt(
        bubble.getAttribute("data-chat-message") ?? "",
        10
      );
      if (!Number.isInteger(messageIndex) || messageIndex < 0) return;

      const caret = caretFromPoint(event.clientX, event.clientY);
      if (!caret || !content.contains(caret.node)) return;

      const specs = collectElaborationSpecs(messages).filter(
        (spec) => spec.messageIndex === messageIndex
      );
      const hit = elaborationAtPoint(specs, content, caret.node, caret.offset);
      if (!hit) return;

      event.preventDefault();
      event.stopPropagation();
      useElaborationStore.getState().openStored(hit.anchor, hit.id, hit.content);
    };

    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [messages]);
}
