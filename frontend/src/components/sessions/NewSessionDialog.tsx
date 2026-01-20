import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks";
import { useUIStore } from "@/stores";

export function NewSessionDialog() {
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);

  const handleNewChat = () => {
    navigate("/");
    // Close sidebar on mobile when starting a new chat
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  return (
    <Button className="w-full" variant="outline" onClick={handleNewChat}>
      <Plus className="mr-2 h-4 w-4" />
      New Chat
    </Button>
  );
}
