import { type ReactNode, useCallback, useRef, useEffect } from "react";
import { useUIStore } from "@/stores";
import { useIsMobile } from "@/hooks";
import { cn } from "@/lib/utils";

const MIN_WIDTH = 260;
const MAX_WIDTH = 520;

interface RightSidebarProps {
  children: ReactNode;
}

export function RightSidebar({ children }: RightSidebarProps) {
  const rightSidebarOpen = useUIStore((state) => state.rightSidebarOpen);
  const setRightSidebarOpen = useUIStore((state) => state.setRightSidebarOpen);
  const rightSidebarWidth = useUIStore((state) => state.rightSidebarWidth);
  const setRightSidebarWidth = useUIStore((state) => state.setRightSidebarWidth);
  const isMobile = useIsMobile();
  const isResizing = useRef(false);
  const sidebarRef = useRef<HTMLElement>(null);

  // Lock body scroll when open on mobile
  useEffect(() => {
    if (isMobile && rightSidebarOpen) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [isMobile, rightSidebarOpen]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (isMobile) return;
      e.preventDefault();
      isResizing.current = true;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [isMobile]
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      // Width = distance from right edge of viewport to cursor
      const newWidth = Math.min(
        MAX_WIDTH,
        Math.max(MIN_WIDTH, window.innerWidth - e.clientX)
      );
      setRightSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      if (isResizing.current) {
        isResizing.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [setRightSidebarWidth]);

  // Mobile: full-screen overlay
  if (isMobile) {
    return (
      <>
        {/* Backdrop */}
        <div
          className={cn(
            "fixed inset-0 z-30 bg-black/50 transition-opacity duration-300",
            rightSidebarOpen ? "opacity-100" : "pointer-events-none opacity-0"
          )}
          onClick={() => setRightSidebarOpen(false)}
          aria-hidden="true"
        />
        <aside
          ref={sidebarRef}
          className={cn(
            "bg-background fixed inset-y-0 right-0 z-40 h-full w-full shadow-xl transition-transform duration-300 ease-in-out",
            rightSidebarOpen ? "translate-x-0" : "translate-x-full"
          )}
        >
          <div className="flex h-full flex-col overflow-hidden">{children}</div>
        </aside>
      </>
    );
  }

  // Desktop: inline resizable panel
  return (
    <aside
      ref={sidebarRef}
      className={cn(
        "relative h-full overflow-hidden transition-[width] duration-200",
        !rightSidebarOpen && "w-0"
      )}
      style={{ width: rightSidebarOpen ? rightSidebarWidth : 0 }}
    >
      {/* Resize handle — left edge */}
      {rightSidebarOpen && (
        <div
          onMouseDown={handleMouseDown}
          className="hover:bg-primary/20 active:bg-primary/30 absolute top-0 left-0 z-10 h-full w-1 cursor-col-resize"
        />
      )}
      <div
        className="border-border flex h-full flex-col overflow-y-auto border-l"
        style={{ width: rightSidebarWidth }}
      >
        {children}
      </div>
    </aside>
  );
}
