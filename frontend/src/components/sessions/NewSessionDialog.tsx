import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function NewSessionDialog() {
  const navigate = useNavigate();

  const handleNewChat = () => {
    navigate("/");
  };

  return (
    <Button className="w-full" variant="outline" onClick={handleNewChat}>
      <Plus className="mr-2 h-4 w-4" />
      New Chat
    </Button>
  );
}
