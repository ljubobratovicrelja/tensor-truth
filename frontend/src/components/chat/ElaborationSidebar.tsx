import { useCallback, useEffect, useRef, useState } from "react";
import { useElaborationStore } from "@/stores/elaborationStore";
import { useIsTabletOrSmaller } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";
import { ElaborationPanel } from "./ElaborationPanel";

const DEFAULT_WIDTH = 380;
const MIN_WIDTH = 300;
const MAX_WIDTH = 720;
const WIDTH_STORAGE_KEY = "tensortruth-elaboration-width";

function readStoredWidth(): number {
  if (typeof window === "undefined") return DEFAULT_WIDTH;
  const stored = Number.parseFloat(window.localStorage.getItem(WIDTH_STORAGE_KEY) ?? "");
  if (!Number.isFinite(stored)) return DEFAULT_WIDTH;
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, stored));
}

interface ElaborationSidebarProps {
  /** Cancels the in-flight explanation run. */
  onCancel?: () => void;
  /** Sends a follow-up turn in the open elaboration's thread. */
  onFollowUp?: (text: string) => void;
}

/**
 * Right-hand panel for the current elaboration. Its open state lives in
 * `useElaborationStore`, deliberately separate from `uiStore.rightSidebarOpen`
 * (which belongs to the project sidebar).
 *
 * On large screens it is an inline panel with a drag handle (width persisted);
 * below the lg breakpoint it becomes a full-page overlay so a narrow/portrait
 * window never squeezes the chat into an unusable column.
 */
export function ElaborationSidebar({ onCancel, onFollowUp }: ElaborationSidebarProps) {
  const open = useElaborationStore((state) => state.open);
  const close = useElaborationStore((state) => state.close);
  const fullPage = useIsTabletOrSmaller();
  const [width, setWidth] = useState(readStoredWidth);
  const widthRef = useRef(width);
  const isResizing = useRef(false);

  useEffect(() => {
    widthRef.current = width;
  }, [width]);

  // Click outside the panel (and outside the trigger) dismisses it.
  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (
        target.closest("[data-elaboration-panel]") ||
        target.closest("[data-elaboration-trigger]")
      ) {
        return;
      }
      close();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, close]);

  // Full-page mode: lock page scroll behind the overlay.
  useEffect(() => {
    if (!fullPage || !open) return;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [fullPage, open]);

  // Drag the left edge to resize; the width survives reloads.
  const handleMouseDown = useCallback(
    (event: React.MouseEvent) => {
      if (fullPage) return;
      event.preventDefault();
      isResizing.current = true;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [fullPage]
  );

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!isResizing.current) return;
      const next = Math.min(
        MAX_WIDTH,
        Math.max(MIN_WIDTH, window.innerWidth - event.clientX)
      );
      setWidth(next);
    };
    const handleMouseUp = () => {
      if (!isResizing.current) return;
      isResizing.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.localStorage.setItem(WIDTH_STORAGE_KEY, String(widthRef.current));
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // Narrow or portrait viewports: the panel takes the whole page.
  if (fullPage) {
    return (
      <>
        <div
          data-elaboration-backdrop=""
          aria-hidden="true"
          onClick={close}
          className={cn(
            "fixed inset-0 z-30 bg-black/50 transition-opacity duration-300",
            open ? "opacity-100" : "pointer-events-none opacity-0"
          )}
        />
        <aside
          data-elaboration-panel=""
          className={cn(
            "bg-background fixed inset-0 z-40 h-dvh w-full shadow-xl transition-transform duration-300 ease-in-out",
            open ? "translate-x-0" : "translate-x-full"
          )}
        >
          <div className="flex h-full flex-col overflow-hidden">
            <ElaborationPanel onCancel={onCancel} onFollowUp={onFollowUp} />
          </div>
        </aside>
      </>
    );
  }

  // Large screens: inline panel that shrinks the chat column.
  return (
    <aside
      data-elaboration-panel=""
      className={cn(
        "relative h-full shrink-0 overflow-hidden transition-[width] duration-200",
        open ? "border-l" : "w-0"
      )}
      style={{ width: open ? width : 0 }}
    >
      {open && (
        <div
          onMouseDown={handleMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize explanation panel"
          className="hover:bg-primary/20 active:bg-primary/30 absolute top-0 left-0 z-10 h-full w-1 cursor-col-resize"
        />
      )}
      <div className="flex h-full flex-col overflow-hidden" style={{ width }}>
        <ElaborationPanel onCancel={onCancel} onFollowUp={onFollowUp} />
      </div>
    </aside>
  );
}
