import type { ReactNode } from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { useUIStore } from "@/stores";
import { useIsMobile } from "@/hooks";
import { cn } from "@/lib/utils";

interface AppLayoutProps {
  sidebar: ReactNode;
  children: ReactNode;
}

export function AppLayout({ sidebar, children }: AppLayoutProps) {
  const sidebarOpen = useUIStore((state) => state.sidebarOpen);
  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);
  const headerHidden = useUIStore((state) => state.headerHidden);
  const isMobile = useIsMobile();

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header />
      <div
        className={cn(
          "relative flex flex-1 overflow-hidden transition-all duration-300 ease-in-out",
          // On mobile, expand content area when header is hidden
          isMobile && headerHidden && "-mt-14"
        )}
      >
        <Sidebar>{sidebar}</Sidebar>

        {/* Backdrop for mobile sidebar */}
        {isMobile && (
          <div
            className={cn(
              "fixed inset-0 z-30 bg-black/50 transition-opacity duration-300",
              sidebarOpen ? "opacity-100" : "pointer-events-none opacity-0"
            )}
            onClick={() => setSidebarOpen(false)}
            aria-hidden="true"
          />
        )}

        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
