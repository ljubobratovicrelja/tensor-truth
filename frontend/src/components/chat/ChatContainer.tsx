import { useEffect } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { useSessionStore, useChatStore } from "@/stores";
import { useSessionMessages, useSession, useWebSocketChat } from "@/hooks";
import { PdfDialog } from "@/components/pdfs";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

export function ChatContainer() {
  const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { activeSessionId, setActiveSessionId } = useSessionStore();
  const { streamingContent, streamingSources, isStreaming, error, pendingUserMessage } =
    useChatStore();

  // Sync URL param with store
  useEffect(() => {
    if (urlSessionId && urlSessionId !== activeSessionId) {
      setActiveSessionId(urlSessionId);
    } else if (!urlSessionId && activeSessionId) {
      setActiveSessionId(null);
    }
  }, [urlSessionId, activeSessionId, setActiveSessionId]);

  const { data: sessionData, error: sessionError } = useSession(activeSessionId);
  const { data: messagesData, isLoading: messagesLoading } =
    useSessionMessages(activeSessionId);

  // Redirect to home if session doesn't exist
  useEffect(() => {
    if (sessionError && urlSessionId) {
      toast.error("Session not found");
      navigate("/", { replace: true });
    }
  }, [sessionError, urlSessionId, navigate]);

  // Scroll to hash anchor after messages load
  useEffect(() => {
    if (!messagesLoading && location.hash) {
      const id = location.hash.slice(1);
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [messagesLoading, location.hash]);

  const { sendMessage, cancelStreaming } = useWebSocketChat({
    sessionId: activeSessionId,
    onError: (err) => toast.error(err),
  });

  if (!activeSessionId) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center">
        <p>Select or create a chat session to get started</p>
      </div>
    );
  }

  const handleSend = (message: string) => {
    sendMessage(message);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <h2 className="text-muted-foreground text-sm font-medium">
          {sessionData?.title ?? "Chat"}
        </h2>
        <PdfDialog sessionId={activeSessionId} />
      </div>
      <MessageList
        messages={messagesData?.messages ?? []}
        isLoading={messagesLoading}
        pendingUserMessage={pendingUserMessage}
        streamingContent={streamingContent || undefined}
        streamingSources={streamingSources.length > 0 ? streamingSources : undefined}
        isStreaming={isStreaming}
      />
      {error && (
        <div className="border-destructive bg-destructive/10 text-destructive border-t py-2 text-sm">
          <div className="chat-content-width">{error}</div>
        </div>
      )}
      <div className="border-t py-4">
        <div className="chat-content-width">
          <ChatInput
            onSend={handleSend}
            onStop={cancelStreaming}
            isStreaming={isStreaming}
            placeholder={isStreaming ? "Generating response..." : undefined}
          />
        </div>
      </div>
    </div>
  );
}
