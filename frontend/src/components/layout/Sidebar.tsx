import { type ReactNode, useCallback, useRef, useEffect } from "react";
import { useUIStore } from "@/stores";
import { useIsMobile } from "@/hooks";
import { cn } from "@/lib/utils";

const MIN_WIDTH = 200;
const MAX_WIDTH = 480;
const MOBILE_WIDTH = 300;

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const sidebarOpen = useUIStore((state) => state.sidebarOpen);
  const sidebarWidth = useUIStore((state) => state.sidebarWidth);
  const setSidebarWidth = useUIStore((state) => state.setSidebarWidth);
  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);

  const isMobile = useIsMobile();
  const isResizing = useRef(false);
  const prevIsMobile = useRef(isMobile);

  // Auto-close sidebar when transitioning to mobile
  useEffect(() => {
    if (isMobile && !prevIsMobile.current && sidebarOpen) {
      setSidebarOpen(false);
    }
    prevIsMobile.current = isMobile;
  }, [isMobile, sidebarOpen, setSidebarOpen]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Disable resize on mobile
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
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX));
      setSidebarWidth(newWidth);
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
  }, [setSidebarWidth]);

  // Mobile: overlay/drawer mode
  if (isMobile) {
    return (
      <aside
        className={cn(
          "bg-sidebar fixed inset-y-0 left-0 z-40 h-full shadow-xl transition-transform duration-300 ease-in-out",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
        style={{ width: MOBILE_WIDTH }}
      >
        <div className="flex h-full flex-col border-r border-border">{children}</div>
      </aside>
    );
  }

  // Desktop: inline resizable mode
  return (
    <aside
      className={cn(
        "bg-sidebar relative h-full overflow-hidden transition-[width] duration-200",
        !sidebarOpen && "w-0"
      )}
      style={{ width: sidebarOpen ? sidebarWidth : 0 }}
    >
      <div
        className="flex h-full flex-col border-r border-border"
        style={{ width: sidebarWidth }}
      >
        {children}
      </div>
      {/* Resize handle - desktop only */}
      {sidebarOpen && (
        <div
          onMouseDown={handleMouseDown}
          className="absolute top-0 right-0 h-full w-1 cursor-col-resize hover:bg-primary/20 active:bg-primary/30"
        />
      )}
    </aside>
  );
}
