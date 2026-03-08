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

/**
 * Reusable auto-scroll hook. Auto-scrolls to the bottom on content changes
 * unless the user has scrolled away.
 *
 * Uses a ref + state dual-tracking pattern:
 * - Ref (`isScrolledAwayRef`) for synchronous checks inside effects (immediate,
 *   survives the React render cycle)
 * - State (`isScrolledAway`) for UI (scroll-to-bottom button visibility)
 *
 * Detects scroll **direction**: any upward scroll disengages immediately,
 * even if still within the near-bottom threshold. Programmatic scrolls always
 * go downward (to max position), so they never trigger disengagement.
 * Re-engage only happens when scrolling back down to within the threshold.
 */
export function useAutoScroll({
  nearBottomThreshold = 50,
  deps,
  enabled = true,
}: UseAutoScrollOptions): UseAutoScrollResult {
  const containerRef = useRef<HTMLElement | null>(null);

  // Dual tracking: ref for synchronous effect checks, state for UI rendering
  const isScrolledAwayRef = useRef(false);
  const [isScrolledAway, setIsScrolledAway] = useState(false);

  // Track last scroll position for direction detection
  const lastScrollTopRef = useRef(0);

  // Flag to suppress scroll-handler during programmatic scrolls
  const isProgrammaticRef = useRef(false);

  // Helper: update both ref and state in sync
  const setScrolledAway = useCallback((value: boolean) => {
    isScrolledAwayRef.current = value;
    setIsScrolledAway(value);
  }, []);

  // ------------------------------------------------------------------
  // Passive scroll listener — direction-based user intent detection
  // ------------------------------------------------------------------
  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;

    const { scrollTop, scrollHeight, clientHeight } = el;

    // Skip handling for programmatic scrolls — clear flag synchronously
    if (isProgrammaticRef.current) {
      isProgrammaticRef.current = false;
      lastScrollTopRef.current = scrollTop;
      return;
    }

    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    const scrollingUp = scrollTop < lastScrollTopRef.current;
    lastScrollTopRef.current = scrollTop;

    if (scrollingUp) {
      // Any upward user scroll → disengage immediately
      setScrolledAway(true);
    } else if (distanceFromBottom <= nearBottomThreshold) {
      // User scrolled back down to bottom → re-engage
      setScrolledAway(false);
    }
  }, [nearBottomThreshold, setScrolledAway]);

  // ------------------------------------------------------------------
  // Ref callback — attach/detach scroll listener
  // ------------------------------------------------------------------
  const scrollRefCb = useCallback(
    (node: HTMLElement | null) => {
      if (containerRef.current) {
        containerRef.current.removeEventListener("scroll", handleScroll);
      }
      containerRef.current = node;
      if (node) {
        node.addEventListener("scroll", handleScroll, { passive: true });
        // Initialize last scroll position
        lastScrollTopRef.current = node.scrollTop;
      }
    },
    [handleScroll]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (containerRef.current) {
        containerRef.current.removeEventListener("scroll", handleScroll);
      }
    };
  }, [handleScroll]);

  // ------------------------------------------------------------------
  // Programmatic scroll helper
  // ------------------------------------------------------------------
  const doScrollToBottom = useCallback((behavior: ScrollBehavior = "instant") => {
    const el = containerRef.current;
    if (!el) return;

    isProgrammaticRef.current = true;
    if (behavior === "instant") {
      el.scrollTop = el.scrollHeight;
    } else {
      el.scrollTo({ top: el.scrollHeight, behavior });
    }
    // Update baseline so next user scroll detects direction correctly
    lastScrollTopRef.current = el.scrollTop;
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
  // Auto-scroll on content changes — check REF (not state) for immediate value
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!enabled || isScrolledAwayRef.current) return;
    doScrollToBottom("instant");
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps spread from caller; isScrolledAway state included to re-run when re-engaged
  }, [enabled, isScrolledAway, doScrollToBottom, ...deps]);

  return { scrollRef: scrollRefCb, isScrolledAway, scrollToBottom, reEngage };
}
