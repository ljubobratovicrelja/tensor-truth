import { useCallback, useEffect, useRef, useState } from "react";
import type { ElaborationAnchor } from "@/stores/elaborationStore";
import { findOccurrences, textContentOffset } from "@/lib/elaborationText";

/** Backend rejects elaborations longer than this, so never offer the trigger. */
export const MAX_SELECTED_TEXT_LENGTH = 4000;

export interface SelectionRect {
  top: number;
  left: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

export interface TextSelection {
  anchor: ElaborationAnchor;
  /** Last client rect of the selected range (where the trigger is placed). */
  rect: SelectionRect;
}

/**
 * Number of whole, non-overlapping occurrences of `needle` in `haystack` that
 * start before `offset`. This is the `occurrence_index` persisted with an
 * elaboration.
 */
export function occurrenceIndexFor(
  haystack: string,
  offset: number,
  needle: string
): number {
  if (!needle) return 0;
  const limit = Math.max(0, Math.min(offset, haystack.length));
  return findOccurrences(haystack, needle).filter((o) => o.start < limit).length;
}

/**
 * Character offset of (node, offset) from the start of `container`, counting
 * only rendered text (same basis as `textContent` and `Range.toString()`).
 */
export function textOffsetFromContainer(
  container: HTMLElement,
  node: Node,
  offset: number
): number | null {
  if (!container.contains(node)) return null;
  try {
    const measure = document.createRange();
    measure.selectNodeContents(container);
    measure.setEnd(node, offset);
    return measure.toString().length;
  } catch {
    return null;
  }
}

function rectFor(range: Range): SelectionRect | null {
  const rects = range.getClientRects();
  const last = rects.length > 0 ? rects[rects.length - 1] : range.getBoundingClientRect();
  if (!last) return null;
  if (
    last.width === 0 &&
    last.height === 0 &&
    last.top === 0 &&
    last.left === 0 &&
    last.right === 0 &&
    last.bottom === 0
  ) {
    // jsdom and other layout-less environments report a zero rect.
    return null;
  }
  return {
    top: last.top,
    left: last.left,
    right: last.right,
    bottom: last.bottom,
    width: last.width,
    height: last.height,
  };
}

const BUBBLE_SELECTOR = `[data-chat-message][data-role="assistant"]`;

/**
 * Turn a DOM range into an elaboration anchor, or null when it is not a
 * single non-empty passage of exactly one assistant answer.
 */
export function resolveAnchorFromRange(range: Range): ElaborationAnchor | null {
  const startEl =
    range.startContainer.nodeType === Node.ELEMENT_NODE
      ? (range.startContainer as Element)
      : range.startContainer.parentElement;
  const endEl =
    range.endContainer.nodeType === Node.ELEMENT_NODE
      ? (range.endContainer as Element)
      : range.endContainer.parentElement;
  if (!startEl || !endEl) return null;

  const bubble = startEl.closest(BUBBLE_SELECTOR);
  if (!bubble) return null;
  // Crossing messages (or escaping the bubble) does not describe one passage.
  if (!bubble.contains(endEl)) return null;

  const content = bubble.querySelector<HTMLElement>("[data-message-content]");
  if (!content) return null;
  if (!content.contains(range.startContainer) || !content.contains(range.endContainer)) {
    return null;
  }

  const raw = range.toString();
  const selectedText = raw.trim();
  if (!selectedText) return null;
  if (selectedText.length > MAX_SELECTED_TEXT_LENGTH) return null;

  const rawStart = textContentOffset(content, range.startContainer, range.startOffset);
  if (rawStart === null) return null;
  const leadingWhitespace = raw.length - raw.trimStart().length;
  const selectedStart = rawStart + leadingWhitespace;
  const containerText = content.textContent ?? "";
  if (selectedStart > containerText.length) return null;

  const messageIndex = Number.parseInt(
    bubble.getAttribute("data-chat-message") ?? "",
    10
  );
  if (!Number.isInteger(messageIndex) || messageIndex < 0) return null;

  return {
    messageIndex,
    selectedText,
    occurrenceIndex: occurrenceIndexFor(containerText, selectedStart, selectedText),
  };
}

/**
 * Turn a DOM selection into an elaboration anchor plus the rect of its last
 * line, or null when it is not a single non-empty passage of one answer.
 */
export function resolveSelectionAnchor(
  selection: Selection | null
): TextSelection | null {
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;

  const range = selection.getRangeAt(0);
  const anchor = resolveAnchorFromRange(range);
  if (!anchor) return null;

  const rect = rectFor(range);
  if (!rect) return null;

  return { anchor, rect };
}

interface UseTextSelectionResult {
  selection: TextSelection | null;
  clear: () => void;
}

/**
 * Tracks the current passage selected inside an assistant bubble. The floating
 * trigger is only shown while a valid selection exists; scrolling or Escape
 * dismisses it.
 */
export function useTextSelection(): UseTextSelectionResult {
  const [selection, setSelection] = useState<TextSelection | null>(null);
  const rafRef = useRef(0);

  const clear = useCallback(() => {
    setSelection(null);
    window.getSelection()?.removeAllRanges();
  }, []);

  const update = useCallback(() => {
    setSelection(resolveSelectionAnchor(window.getSelection()));
  }, []);

  useEffect(() => {
    const updateSoon = () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(update);
    };

    // A collapsed selection is what a plain click produces: dismiss at once.
    const handleSelectionChange = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) {
        cancelAnimationFrame(rafRef.current);
        setSelection(null);
        return;
      }
      updateSoon();
    };

    const handlePointerUp = () => {
      cancelAnimationFrame(rafRef.current);
      update();
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") clear();
    };

    const handleScroll = () => setSelection(null);

    document.addEventListener("selectionchange", handleSelectionChange);
    document.addEventListener("mouseup", handlePointerUp);
    document.addEventListener("keyup", handlePointerUp);
    document.addEventListener("keydown", handleKeyDown);
    // Capture phase: the message list scrolls, not the window.
    window.addEventListener("scroll", handleScroll, true);

    return () => {
      cancelAnimationFrame(rafRef.current);
      document.removeEventListener("selectionchange", handleSelectionChange);
      document.removeEventListener("mouseup", handlePointerUp);
      document.removeEventListener("keyup", handlePointerUp);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("scroll", handleScroll, true);
    };
  }, [clear, update]);

  return { selection, clear };
}
