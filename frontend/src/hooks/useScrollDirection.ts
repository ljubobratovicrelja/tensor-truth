import { useState, useEffect, useCallback, useRef } from "react";

interface ScrollDirectionOptions {
  /** Threshold in pixels before "down" direction is registered */
  threshold?: number;
  /** Threshold in pixels before "up" direction is registered (higher to avoid bounce effects) */
  upThreshold?: number;
  /** Percentage of scroll height considered "near top" (0-1) */
  topThreshold?: number;
}

interface ScrollDirectionResult {
  /** Current scroll direction */
  direction: "up" | "down" | null;
  /** Whether we're at the absolute top (scrollTop === 0) */
  isAtTop: boolean;
  /** Whether we're near the top of the scrollable area */
  isNearTop: boolean;
  /** Whether the content is scrollable (overflows the container) */
  isScrollable: boolean;
  /** Ref to attach to the scrollable element */
  scrollRef: React.RefCallback<HTMLElement>;
}

/**
 * Hook to track scroll direction and position within a scrollable element.
 */
export function useScrollDirection(
  options: ScrollDirectionOptions = {}
): ScrollDirectionResult {
  const { threshold = 10, upThreshold = 150, topThreshold = 0.1 } = options;

  // Initialize to values that will likely change on first scroll check
  // This ensures the effect in consumers triggers after mount
  const [direction, setDirection] = useState<"up" | "down" | null>(null);
  const [isAtTop, setIsAtTop] = useState(false); // Will become true if at top
  const [isNearTop, setIsNearTop] = useState(false); // Will become true if near top
  const [isScrollable, setIsScrollable] = useState(true); // Will become false if not scrollable

  const lastScrollTop = useRef(0);
  const elementRef = useRef<HTMLElement | null>(null);

  const handleScroll = useCallback(() => {
    const element = elementRef.current;
    if (!element) return;

    const scrollTop = element.scrollTop;
    const scrollHeight = element.scrollHeight;
    const clientHeight = element.clientHeight;

    // Check if at absolute top
    setIsAtTop(scrollTop === 0);

    // Check if content is scrollable
    const maxScroll = scrollHeight - clientHeight;
    setIsScrollable(maxScroll > 0);

    // Calculate if near top (within topThreshold of total scrollable area)
    const scrollPercentage = maxScroll > 0 ? scrollTop / maxScroll : 0;
    setIsNearTop(scrollPercentage < topThreshold || maxScroll <= 0);

    // Determine direction with threshold
    // Use larger threshold for "up" to avoid iOS bounce triggering header show
    const diff = scrollTop - lastScrollTop.current;
    const requiredThreshold = diff < 0 ? upThreshold : threshold;

    if (Math.abs(diff) > requiredThreshold) {
      setDirection(diff > 0 ? "down" : "up");
      lastScrollTop.current = scrollTop;
    }
  }, [threshold, upThreshold, topThreshold]);

  // Ref callback to attach scroll listener
  const scrollRef = useCallback(
    (node: HTMLElement | null) => {
      // Cleanup previous listener
      if (elementRef.current) {
        elementRef.current.removeEventListener("scroll", handleScroll);
      }

      elementRef.current = node;

      // Attach new listener
      if (node) {
        node.addEventListener("scroll", handleScroll, { passive: true });
        // Initialize state
        handleScroll();
      }
    },
    [handleScroll]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (elementRef.current) {
        elementRef.current.removeEventListener("scroll", handleScroll);
      }
    };
  }, [handleScroll]);

  return { direction, isAtTop, isNearTop, isScrollable, scrollRef };
}
