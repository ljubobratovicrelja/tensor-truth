import { type ReactNode } from "react";
import { useUIStore } from "@/stores";
import { useIsMobile } from "@/hooks";
import { cn } from "@/lib/utils";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";

const RIGHT_SIDEBAR_WIDTH = 280;

interface RightSidebarProps {
  children: ReactNode;
}

export function RightSidebar({ children }: RightSidebarProps) {
  const rightSidebarOpen = useUIStore((state) => state.rightSidebarOpen);
  const setRightSidebarOpen = useUIStore((state) => state.setRightSidebarOpen);
  const isMobile = useIsMobile();

  // Mobile: Sheet/drawer from the right side
  if (isMobile) {
    return (
      <Sheet open={rightSidebarOpen} onOpenChange={setRightSidebarOpen}>
        <SheetContent side="right" className="w-[300px] p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Project Settings</SheetTitle>
          </SheetHeader>
          <div className="flex h-full flex-col overflow-y-auto">{children}</div>
        </SheetContent>
      </Sheet>
    );
  }

  // Desktop: inline fixed-width panel
  return (
    <aside
      className={cn(
        "relative h-full overflow-hidden transition-[width] duration-200",
        !rightSidebarOpen && "w-0"
      )}
      style={{ width: rightSidebarOpen ? RIGHT_SIDEBAR_WIDTH : 0 }}
    >
      <div
        className="border-border flex h-full flex-col overflow-y-auto border-l"
        style={{ width: RIGHT_SIDEBAR_WIDTH }}
      >
        {children}
      </div>
    </aside>
  );
}
