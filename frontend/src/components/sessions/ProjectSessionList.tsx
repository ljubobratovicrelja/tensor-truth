import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { ArrowLeft, Plus } from "lucide-react";
import {
  useProjectSessions,
  useCreateProjectSession,
  useDeleteSession,
  useUpdateSession,
  useIsMobile,
} from "@/hooks";
import { useUIStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { SessionItem } from "./SessionItem";

interface ProjectSessionListProps {
  projectId: string;
}

export function ProjectSessionList({ projectId }: ProjectSessionListProps) {
  const { data, isLoading, error } = useProjectSessions(projectId);
  const createSession = useCreateProjectSession();
  const deleteSession = useDeleteSession();
  const updateSession = useUpdateSession();
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams<{ sessionId: string }>();
  const isMobile = useIsMobile();
  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);

  const handleNewChat = async () => {
    try {
      const result = await createSession.mutateAsync({ projectId });
      navigate(`/projects/${projectId}/chat/${result.session_id}`);
      if (isMobile) {
        setSidebarOpen(false);
      }
    } catch (err) {
      console.error("Failed to create session:", err);
      toast.error("Failed to create chat session");
    }
  };

  const handleSessionClick = (sessionId: string) => {
    navigate(`/projects/${projectId}/chat/${sessionId}`);
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    const isCurrentSession = activeSessionId === sessionId;
    try {
      await deleteSession.mutateAsync(sessionId);
      toast.success("Session deleted");
      if (isCurrentSession) {
        navigate(`/projects/${projectId}`, { replace: true });
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
      toast.error("Failed to delete session");
    }
  };

  const handleRename = async (sessionId: string, newTitle: string) => {
    try {
      await updateSession.mutateAsync({
        sessionId,
        data: { title: newTitle },
      });
      toast.success("Session renamed");
    } catch (err) {
      console.error("Failed to rename session:", err);
      toast.error("Failed to rename session");
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="shrink-0 space-y-2 p-3">
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground w-full justify-start gap-2"
          onClick={() => navigate("/projects")}
        >
          <ArrowLeft className="h-4 w-4" />
          All projects
        </Button>
        <Button
          className="w-full"
          variant="outline"
          onClick={handleNewChat}
          disabled={createSession.isPending}
        >
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>

      <ScrollArea className="flex-1 overflow-auto">
        <div className="space-y-1 px-3 pb-3">
          {isLoading ? (
            <>
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
            </>
          ) : error ? (
            <p className="text-muted-foreground px-3 py-2 text-sm">
              Failed to load sessions
            </p>
          ) : data?.sessions.length === 0 ? (
            <p className="text-muted-foreground px-3 py-2 text-sm">
              No chat sessions yet
            </p>
          ) : (
            data?.sessions.map((session) => (
              <SessionItem
                key={session.session_id}
                session={session}
                isActive={activeSessionId === session.session_id}
                onClick={() => handleSessionClick(session.session_id)}
                onDelete={() => handleDelete(session.session_id)}
                onRename={(newTitle) => handleRename(session.session_id, newTitle)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
