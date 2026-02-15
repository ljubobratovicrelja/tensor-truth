import { useLocation, useNavigate } from "react-router-dom";
import { MessageSquare, FolderKanban } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function SidebarNav() {
  const location = useLocation();
  const navigate = useNavigate();

  const isChatsActive =
    location.pathname === "/" || location.pathname.startsWith("/chat");
  const isProjectsActive = location.pathname.startsWith("/projects");

  return (
    <div className="border-border flex shrink-0 gap-1 border-b p-2">
      <Button
        variant="ghost"
        size="sm"
        className={cn(
          "flex-1 gap-2",
          isChatsActive && "bg-sidebar-accent text-sidebar-accent-foreground"
        )}
        onClick={() => navigate("/")}
      >
        <MessageSquare className="h-4 w-4" />
        Chats
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className={cn(
          "flex-1 gap-2",
          isProjectsActive && "bg-sidebar-accent text-sidebar-accent-foreground"
        )}
        onClick={() => navigate("/projects")}
      >
        <FolderKanban className="h-4 w-4" />
        Projects
      </Button>
    </div>
  );
}
