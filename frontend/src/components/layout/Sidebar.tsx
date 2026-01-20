import type { ReactNode } from "react";
import { useUIStore } from "@/stores";
import { cn } from "@/lib/utils";

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const sidebarOpen = useUIStore((state) => state.sidebarOpen);

  return (
    <aside
      className={cn(
        "border-border bg-sidebar h-full overflow-hidden border-r transition-all duration-200",
        sidebarOpen ? "w-64" : "w-0"
      )}
    >
      <div className="flex h-full w-64 flex-col">{children}</div>
    </aside>
  );
}
