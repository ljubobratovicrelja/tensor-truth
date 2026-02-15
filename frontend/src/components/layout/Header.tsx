import { useLocation, Link } from "react-router-dom";
import {
  PanelLeftClose,
  PanelLeft,
  PanelRight,
  PanelRightClose,
  Menu,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUIStore } from "@/stores";
import { useIsMobile } from "@/hooks";
import { ConfigPanel } from "@/components/config";
import { SystemStatusPanel } from "@/components/status";
import { ThemeToggle } from "./ThemeToggle";
import { cn } from "@/lib/utils";

export function Header() {
  const location = useLocation();
  const {
    sidebarOpen,
    toggleSidebar,
    rightSidebarOpen,
    toggleRightSidebar,
    headerHidden,
  } = useUIStore();
  const isMobile = useIsMobile();
  const isInChat = location.pathname.includes("/chat/");
  const isInProject = location.pathname.startsWith("/projects/");

  // Choose icon based on mobile state and sidebar state
  const getIcon = () => {
    if (isMobile) {
      return sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />;
    }
    return sidebarOpen ? (
      <PanelLeftClose className="h-5 w-5" />
    ) : (
      <PanelLeft className="h-5 w-5" />
    );
  };

  return (
    <header
      className={cn(
        "border-border bg-background flex h-14 items-center gap-2 border-b px-3 md:gap-4 md:px-4",
        "transition-transform duration-300 ease-in-out",
        // On mobile, slide up when hidden
        isMobile && headerHidden && "-translate-y-full"
      )}
    >
      <Button variant="ghost" size="icon" onClick={toggleSidebar} className="shrink-0">
        {getIcon()}
      </Button>
      <div className="flex flex-1 items-center gap-2">
        <Link
          to="/"
          className="flex items-center gap-2 transition-opacity hover:opacity-80"
        >
          <img src="/logo.png" alt="TensorTruth" className="h-7 w-7" />
          <h1 className="hidden text-lg font-semibold sm:block">TensorTruth</h1>
        </Link>
      </div>
      <SystemStatusPanel />
      <ThemeToggle />
      {!isInChat && <ConfigPanel />}
      {isInProject && (
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleRightSidebar}
          className="shrink-0"
          title={rightSidebarOpen ? "Close project panel" : "Open project panel"}
        >
          {rightSidebarOpen ? (
            <PanelRightClose className="h-5 w-5" />
          ) : (
            <PanelRight className="h-5 w-5" />
          )}
        </Button>
      )}
    </header>
  );
}
