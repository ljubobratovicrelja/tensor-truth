import { useSessions, useDeleteSession } from "@/hooks";
import { useSessionStore } from "@/stores";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { NewSessionDialog } from "./NewSessionDialog";
import { SessionItem } from "./SessionItem";

export function SessionList() {
  const { data, isLoading, error } = useSessions();
  const deleteSession = useDeleteSession();
  const { activeSessionId, setActiveSessionId } = useSessionStore();

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSession.mutateAsync(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="p-3">
        <NewSessionDialog />
      </div>

      <ScrollArea className="flex-1">
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
                onClick={() => setActiveSessionId(session.session_id)}
                onDelete={() => handleDelete(session.session_id)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
