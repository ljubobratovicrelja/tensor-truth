import { PanelLeftClose, PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUIStore } from "@/stores";
import { ConfigPanel } from "@/components/config";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  const { sidebarOpen, toggleSidebar } = useUIStore();

  return (
    <header className="border-border flex h-14 items-center gap-4 border-b px-4">
      <Button variant="ghost" size="icon" onClick={toggleSidebar} className="shrink-0">
        {sidebarOpen ? (
          <PanelLeftClose className="h-5 w-5" />
        ) : (
          <PanelLeft className="h-5 w-5" />
        )}
      </Button>
      <div className="flex flex-1 items-center gap-2">
        <img src="/logo.png" alt="TensorTruth" className="h-7 w-7" />
        <h1 className="text-lg font-semibold">TensorTruth</h1>
      </div>
      <ThemeToggle />
      <ConfigPanel />
    </header>
  );
}
