import { useCallback, useEffect, useRef, useState } from "react";

interface UseAutoScrollOptions {
  /** Distance from bottom (px) within which the user is considered "back at bottom" */
  nearBottomThreshold?: number;
  /** Content-change deps that trigger auto-scroll */
  deps: unknown[];
  /** Gate auto-scroll (e.g., only during streaming). Default true */
  enabled?: boolean;
}

interface UseAutoScrollResult {
  /** Ref callback — attach to the scrollable container */
  scrollRef: React.RefCallback<HTMLElement>;
  /** True when the user has intentionally scrolled away from the bottom */
  isScrolledAway: boolean;
  /** Smooth-scroll to bottom and re-engage auto-scroll */
  scrollToBottom: (behavior?: ScrollBehavior) => void;
  /** Force re-engage (e.g., when a new streaming session starts) */
  reEngage: () => void;
}

/** A scroll landing this close to the position we assigned is ours, not the user's. */
const PROGRAMMATIC_TOLERANCE_PX = 2;
/** Finger travel (px) before a touch is treated as a deliberate upward drag. */
const TOUCH_INTENT_PX = 2;
/** Pointer travel (px) before a drag is treated as a deliberate upward drag. */
const POINTER_INTENT_PX = 4;

function distanceFromBottom(el: HTMLElement): number {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}

/**
 * Reusable auto-scroll hook. Follows new content while the user is at the
 * bottom; once the user takes the scroll away with a gesture it stays put.
 *
 * Intent comes from **input events** (`wheel`, touch, pointer, keydown), never
 * from scroll-position deltas: wheel/touch/key are unambiguous, whereas deltas
 * are corrupted by coalesced wheel events, scrollbar drags, momentum scrolling
 * and content growth.
 *
 * Our own scrolls are recognised by remembering the exact `scrollTop` we
 * assigned (`programmaticTopRef`) and comparing the arriving scroll event
 * against it. That is a value comparison, not a one-shot boolean, so a
 * programmatic scroll that is a no-op (already at the bottom → no scroll event)
 * cannot leave a flag armed to swallow the user's next scroll.
 *
 * Dual tracking: `isScrolledAwayRef` for synchronous checks inside effects and
 * event handlers, `isScrolledAway` state for UI (scroll-to-bottom button).
 */
export function useAutoScroll({
  nearBottomThreshold = 24,
  deps,
  enabled = true,
}: UseAutoScrollOptions): UseAutoScrollResult {
  const containerRef = useRef<HTMLElement | null>(null);

  // Dual tracking: ref for synchronous checks, state for UI rendering
  const isScrolledAwayRef = useRef(false);
  const [isScrolledAway, setIsScrolledAway] = useState(false);

  /** scrollTop we last assigned programmatically (null when unknown) */
  const programmaticTopRef = useRef<number | null>(null);
  /** Last scrollTop observed — scrollbar-drag fallback only */
  const lastTopRef = useRef(0);
  /** Gesture bookkeeping for touch / pointer direction inference */
  const touchStartYRef = useRef<number | null>(null);
  const pointerStartYRef = useRef<number | null>(null);
  const pointerActiveRef = useRef(false);
  const gestureStartTopRef = useRef(0);

  /** Removes every listener currently attached to the container */
  const detachRef = useRef<(() => void) | null>(null);

  // Helper: update ref first (synchronous), then state only when it changes
  const setScrolledAway = useCallback((value: boolean) => {
    if (isScrolledAwayRef.current === value) return;
    isScrolledAwayRef.current = value;
    setIsScrolledAway(value);
  }, []);

  const disengage = useCallback(() => setScrolledAway(true), [setScrolledAway]);
  const engage = useCallback(() => setScrolledAway(false), [setScrolledAway]);

  const engageIfNearBottom = useCallback(
    (el: HTMLElement) => {
      if (distanceFromBottom(el) <= nearBottomThreshold) engage();
    },
    [engage, nearBottomThreshold]
  );

  // ------------------------------------------------------------------
  // scroll — only re-engages; upward intent comes from gestures
  // ------------------------------------------------------------------
  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;

    const top = el.scrollTop;
    const target = programmaticTopRef.current;

    // Our own scroll: never touches intent state.
    if (target !== null && Math.abs(top - target) <= PROGRAMMATIC_TOLERANCE_PX) {
      lastTopRef.current = top;
      return;
    }
    programmaticTopRef.current = null;

    const previousTop = lastTopRef.current;
    lastTopRef.current = top;

    // Fallback for scrollbar drags / momentum, which can reach us as scroll
    // events only (native scrollbars do not dispatch pointer events).
    if (top < previousTop - PROGRAMMATIC_TOLERANCE_PX) {
      disengage();
      return;
    }

    engageIfNearBottom(el);
  }, [disengage, engageIfNearBottom]);

  // ------------------------------------------------------------------
  // wheel — synchronous disengage on upward intent
  // ------------------------------------------------------------------
  const handleWheel = useCallback(
    (event: WheelEvent) => {
      const el = containerRef.current;
      programmaticTopRef.current = null;
      if (event.deltaY < 0) {
        disengage();
        return;
      }
      if (el && event.deltaY > 0) engageIfNearBottom(el);
    },
    [disengage, engageIfNearBottom]
  );

  // ------------------------------------------------------------------
  // touch — finger moving down drags the content up
  // ------------------------------------------------------------------
  const handleTouchStart = useCallback((event: TouchEvent) => {
    programmaticTopRef.current = null;
    const touch = event.touches[0];
    touchStartYRef.current = touch ? touch.clientY : null;
    const el = containerRef.current;
    gestureStartTopRef.current = el ? el.scrollTop : 0;
  }, []);

  const handleTouchMove = useCallback(
    (event: TouchEvent) => {
      const startY = touchStartYRef.current;
      const touch = event.touches[0];
      if (startY === null || !touch) return;
      if (touch.clientY - startY > TOUCH_INTENT_PX) {
        touchStartYRef.current = touch.clientY;
        programmaticTopRef.current = null;
        disengage();
      }
    },
    [disengage]
  );

  const handleTouchEnd = useCallback(() => {
    touchStartYRef.current = null;
  }, []);

  // ------------------------------------------------------------------
  // pointer — scrollbar drag or drag-anywhere; direction from Y / scrollTop
  // ------------------------------------------------------------------
  const handlePointerDown = useCallback((event: PointerEvent) => {
    const el = containerRef.current;
    programmaticTopRef.current = null;
    pointerActiveRef.current = true;
    pointerStartYRef.current = event.clientY;
    gestureStartTopRef.current = el ? el.scrollTop : 0;
  }, []);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      if (!pointerActiveRef.current) return;
      const startY = pointerStartYRef.current;
      if (startY === null) return;

      const el = containerRef.current;
      const draggedDown = event.clientY - startY > POINTER_INTENT_PX;
      const scrolledUp = !!el && el.scrollTop < gestureStartTopRef.current - 1;

      if (draggedDown || scrolledUp) {
        pointerStartYRef.current = event.clientY;
        programmaticTopRef.current = null;
        disengage();
      }
    },
    [disengage]
  );

  const handlePointerEnd = useCallback(() => {
    pointerActiveRef.current = false;
    pointerStartYRef.current = null;
  }, []);

  // ------------------------------------------------------------------
  // keydown — keyboard scrolling is unambiguous intent
  // ------------------------------------------------------------------
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
      ) {
        return;
      }

      if (event.key === "ArrowUp" || event.key === "PageUp" || event.key === "Home") {
        programmaticTopRef.current = null;
        disengage();
      } else if (
        event.key === "ArrowDown" ||
        event.key === "PageDown" ||
        event.key === "End"
      ) {
        // The resulting scroll event re-engages if it lands near the bottom.
        programmaticTopRef.current = null;
      }
    },
    [disengage]
  );

  // ------------------------------------------------------------------
  // Ref callback — symmetric attach/detach of every listener
  // ------------------------------------------------------------------
  const scrollRefCb = useCallback(
    (node: HTMLElement | null) => {
      detachRef.current?.();
      detachRef.current = null;
      containerRef.current = node;
      if (!node) return;

      lastTopRef.current = node.scrollTop;
      programmaticTopRef.current = null;
      touchStartYRef.current = null;
      pointerStartYRef.current = null;
      pointerActiveRef.current = false;

      const listeners: [string, EventListener, AddEventListenerOptions][] = [
        ["scroll", handleScroll as EventListener, { passive: true }],
        ["wheel", handleWheel as EventListener, { passive: true }],
        ["touchstart", handleTouchStart as EventListener, { passive: true }],
        ["touchmove", handleTouchMove as EventListener, { passive: true }],
        ["touchend", handleTouchEnd as EventListener, { passive: true }],
        ["touchcancel", handleTouchEnd as EventListener, { passive: true }],
        ["pointerdown", handlePointerDown as EventListener, {}],
        ["pointermove", handlePointerMove as EventListener, {}],
        ["pointerup", handlePointerEnd as EventListener, {}],
        ["pointercancel", handlePointerEnd as EventListener, {}],
        ["keydown", handleKeyDown as EventListener, {}],
      ];

      for (const [type, listener, options] of listeners) {
        node.addEventListener(type, listener, options);
      }

      detachRef.current = () => {
        for (const [type, listener, options] of listeners) {
          node.removeEventListener(type, listener, options);
        }
      };
    },
    [
      handleScroll,
      handleWheel,
      handleTouchStart,
      handleTouchMove,
      handleTouchEnd,
      handlePointerDown,
      handlePointerMove,
      handlePointerEnd,
      handleKeyDown,
    ]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      detachRef.current?.();
      detachRef.current = null;
    };
  }, []);

  // ------------------------------------------------------------------
  // Programmatic scroll helpers
  // ------------------------------------------------------------------
  const doScrollToBottom = useCallback((behavior: ScrollBehavior = "instant") => {
    const el = containerRef.current;
    if (!el) return;

    const maxScroll = el.scrollHeight - el.clientHeight;
    programmaticTopRef.current = maxScroll;

    if (behavior !== "instant" && typeof el.scrollTo === "function") {
      el.scrollTo({ top: maxScroll, behavior });
    } else {
      el.scrollTop = maxScroll;
    }
    lastTopRef.current = el.scrollTop;
  }, []);

  // ------------------------------------------------------------------
  // Public scrollToBottom — also re-engages
  // ------------------------------------------------------------------
  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = "smooth") => {
      setScrolledAway(false);
      doScrollToBottom(behavior);
    },
    [doScrollToBottom, setScrolledAway]
  );

  // ------------------------------------------------------------------
  // Public reEngage — used when a new streaming session starts
  // ------------------------------------------------------------------
  const reEngage = useCallback(() => {
    setScrolledAway(false);
  }, [setScrolledAway]);

  // ------------------------------------------------------------------
  // Auto-scroll on content changes — REF check, never re-engages
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!enabled || isScrolledAwayRef.current) return;
    const el = containerRef.current;
    if (!el) return;
    if (el.scrollHeight - el.clientHeight <= 0) return;
    doScrollToBottom("instant");
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps are caller-supplied content deps; isScrolledAway state is deliberately excluded so that running this effect can never re-engage
  }, [enabled, doScrollToBottom, ...deps]);

  return { scrollRef: scrollRefCb, isScrolledAway, scrollToBottom, reEngage };
}
