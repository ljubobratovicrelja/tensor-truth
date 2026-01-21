import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { useSessions, useDeleteSession, useUpdateSession, useIsMobile } from "@/hooks";
import { useUIStore } from "@/stores";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { NewSessionDialog } from "./NewSessionDialog";
import { SessionItem } from "./SessionItem";

export function SessionList() {
  const { data, isLoading, error } = useSessions();
  const deleteSession = useDeleteSession();
  const updateSession = useUpdateSession();
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams<{ sessionId: string }>();
  const isMobile = useIsMobile();
  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);

  const handleSessionClick = (sessionId: string) => {
    navigate(`/chat/${sessionId}`);
    // Close sidebar on mobile after selecting a session
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
        navigate("/", { replace: true });
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      toast.error("Failed to delete session");
    }
  };

  const handleRename = async (sessionId: string, newTitle: string) => {
    try {
      await updateSession.mutateAsync({ sessionId, data: { title: newTitle } });
      toast.success("Session renamed");
    } catch (error) {
      console.error("Failed to rename session:", error);
      toast.error("Failed to rename session");
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="shrink-0 p-3">
        <NewSessionDialog />
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
