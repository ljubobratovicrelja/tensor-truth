import { useSyncExternalStore, useCallback } from "react";

/**
 * Hook that returns true if the given media query matches.
 * Updates when the viewport changes.
 * Uses useSyncExternalStore for proper synchronization.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (callback: () => void) => {
      if (typeof window === "undefined") return () => {};

      const mediaQuery = window.matchMedia(query);
      mediaQuery.addEventListener("change", callback);
      return () => mediaQuery.removeEventListener("change", callback);
    },
    [query]
  );

  const getSnapshot = useCallback(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  }, [query]);

  const getServerSnapshot = useCallback(() => false, []);

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

// Tailwind breakpoints:
// sm: 640px, md: 768px, lg: 1024px, xl: 1280px, 2xl: 1536px

/**
 * Returns true when viewport is below the md breakpoint (768px).
 * Use this for mobile-specific behavior.
 */
export function useIsMobile(): boolean {
  return useMediaQuery("(max-width: 767px)");
}

/**
 * Returns true when viewport is below the lg breakpoint (1024px).
 * Use this for tablet and mobile behavior.
 */
export function useIsTabletOrSmaller(): boolean {
  return useMediaQuery("(max-width: 1023px)");
}

/**
 * Returns true when viewport is at or above the lg breakpoint (1024px).
 * Use this for desktop-specific behavior.
 */
export function useIsDesktop(): boolean {
  return useMediaQuery("(min-width: 1024px)");
}
