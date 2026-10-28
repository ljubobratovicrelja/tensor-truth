import { useEffect } from "react";
import { useElaborationStore } from "@/stores/elaborationStore";
import { useIsMobile } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";
import { ElaborationPanel } from "./ElaborationPanel";

const PANEL_WIDTH = 380;

interface ElaborationSidebarProps {
  /** Cancels the in-flight explanation run. */
  onCancel?: () => void;
}

/**
 * Right-hand panel for the current elaboration. Its open state lives in
 * `useElaborationStore`, deliberately separate from `uiStore.rightSidebarOpen`
 * (which belongs to the project sidebar).
 */
export function ElaborationSidebar({ onCancel }: ElaborationSidebarProps) {
  const open = useElaborationStore((state) => state.open);
  const close = useElaborationStore((state) => state.close);
  const isMobile = useIsMobile();

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

  // Mobile: lock page scroll behind the full-screen overlay.
  useEffect(() => {
    if (!isMobile || !open) return;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [isMobile, open]);

  // Mobile: full-screen overlay with backdrop.
  if (isMobile) {
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
            "bg-background fixed inset-y-0 right-0 z-40 h-full w-full shadow-xl transition-transform duration-300 ease-in-out",
            open ? "translate-x-0" : "translate-x-full"
          )}
        >
          <div className="flex h-full flex-col overflow-hidden">
            <ElaborationPanel onCancel={onCancel} />
          </div>
        </aside>
      </>
    );
  }

  // Desktop: inline panel that shrinks the chat column.
  return (
    <aside
      data-elaboration-panel=""
      className={cn(
        "relative h-full shrink-0 overflow-hidden transition-[width] duration-200",
        open ? "border-l" : "w-0"
      )}
      style={{ width: open ? PANEL_WIDTH : 0 }}
    >
      <div
        className="flex h-full flex-col overflow-hidden"
        style={{ width: PANEL_WIDTH }}
      >
        <ElaborationPanel onCancel={onCancel} />
      </div>
    </aside>
  );
}
