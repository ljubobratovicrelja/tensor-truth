import { useEffect, useRef } from "react";
import type { MessageResponse, ThreadMessage } from "@/api/types";
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
): {
  anchor: ElaborationAnchor;
  id: string;
  content: string;
  messages: ThreadMessage[];
} | null {
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
      messages: spec.messages,
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

interface HitTestResult {
  content: HTMLElement;
  hit: NonNullable<ReturnType<typeof elaborationAtPoint>>;
}

interface LiveElaboration {
  anchor: ElaborationAnchor;
  content: string;
  messages: ThreadMessage[];
}

function sameAnchor(a: ElaborationAnchor, b: ElaborationAnchor): boolean {
  return (
    a.messageIndex === b.messageIndex &&
    a.selectedText === b.selectedText &&
    a.occurrenceIndex === b.occurrenceIndex
  );
}

/** Resolve a pointer position to the stored elaboration under it, if any. */
function hitTestEvent(
  target: EventTarget | null,
  x: number,
  y: number,
  messages: MessageResponse[],
  live: LiveElaboration | null
): HitTestResult | null {
  if (!(target instanceof Element)) return null;
  if (target.closest("[data-elaboration-panel]")) return null;
  // Links and controls inside a passage keep their own behaviour.
  if (target.closest("a, button, summary, [role='button']")) return null;

  const content = target.closest<HTMLElement>("[data-message-content]");
  const bubble = content?.closest<HTMLElement>(
    `[data-chat-message][data-role="assistant"]`
  );
  if (!content || !bubble) return null;

  const messageIndex = Number.parseInt(
    bubble.getAttribute("data-chat-message") ?? "",
    10
  );
  if (!Number.isInteger(messageIndex) || messageIndex < 0) return null;

  const caret = caretFromPoint(x, y);
  if (!caret || !content.contains(caret.node)) return null;

  const specs = collectElaborationSpecs(messages).filter(
    (spec) => spec.messageIndex === messageIndex
  );
  // A run that has not been persisted yet still needs hover/click support.
  if (
    live &&
    live.anchor.messageIndex === messageIndex &&
    !specs.some((spec) => sameAnchor(spec, live.anchor))
  ) {
    specs.push({
      id: "live",
      messageIndex,
      selectedText: live.anchor.selectedText,
      occurrenceIndex: live.anchor.occurrenceIndex,
      content: live.content,
      messages: live.messages,
    });
  }

  const hit = elaborationAtPoint(specs, content, caret.node, caret.offset);
  return hit ? { content, hit } : null;
}

/**
 * Clicking a passage that was already explained re-opens its panel with the
 * stored answer instead of starting a new run. Also gives those passages a
 * hand cursor: `::highlight()` cannot set `cursor`, so the pointer style is
 * applied to the content container while it hovers an elaboration.
 */
export function useElaborationClick(
  messages: MessageResponse[] | undefined | null,
  live?: LiveElaboration | null
) {
  // The live elaboration changes on every streamed token; keep it in a ref so
  // the DOM listeners are not torn down and re-added per token.
  const liveRef = useRef<LiveElaboration | null>(live ?? null);
  useEffect(() => {
    liveRef.current = live ?? null;
  }, [live]);

  useEffect(() => {
    if (!messages || messages.length === 0) return;

    let hovered: HTMLElement | null = null;
    const setCursor = (element: HTMLElement | null, pointer: boolean) => {
      if (hovered && hovered !== element) hovered.style.cursor = "";
      hovered = element;
      if (element) element.style.cursor = pointer ? "pointer" : "";
    };

    const handleClick = (event: MouseEvent) => {
      // A real text selection is the trigger's job, not a stored elaboration's.
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;

      const result = hitTestEvent(
        event.target,
        event.clientX,
        event.clientY,
        messages,
        liveRef.current
      );
      if (!result) return;

      event.preventDefault();
      event.stopPropagation();
      const store = useElaborationStore.getState();
      const currentLive = liveRef.current;
      const isLiveHit =
        currentLive !== null && sameAnchor(currentLive.anchor, result.hit.anchor);
      if (isLiveHit) {
        // In-flight or fetched elaboration: reopen what the store holds so the
        // panel resumes exactly where the run is.
        store.reopen();
      } else {
        store.openStored(
          result.hit.anchor,
          result.hit.id,
          result.hit.content,
          result.hit.messages
        );
      }
    };

    let frame = 0;
    let latest: MouseEvent | null = null;
    const applyCursor = () => {
      frame = 0;
      const event = latest;
      if (!event) return;
      const result = hitTestEvent(
        event.target,
        event.clientX,
        event.clientY,
        messages,
        liveRef.current
      );
      if (result) {
        setCursor(result.content, true);
      } else {
        setCursor(null, false);
      }
    };

    const handleMouseMove = (event: MouseEvent) => {
      latest = event;
      if (frame) return; // one caret lookup per animation frame
      frame = requestAnimationFrame(applyCursor);
    };

    const clearCursor = () => {
      latest = null;
      setCursor(null, false);
    };

    document.addEventListener("click", handleClick);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", clearCursor);
    return () => {
      document.removeEventListener("click", handleClick);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", clearCursor);
      if (frame) cancelAnimationFrame(frame);
      if (hovered) hovered.style.cursor = "";
    };
  }, [messages]);
}
