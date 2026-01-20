import { toast } from "sonner";
import { useSessionStore, useChatStore } from "@/stores";
import { useSessionMessages, useSession, useWebSocketChat } from "@/hooks";
import { PdfDialog } from "@/components/pdfs";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

export function ChatContainer() {
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const { streamingContent, streamingSources, isStreaming, error } = useChatStore();

  const { data: sessionData } = useSession(activeSessionId);
  const { data: messagesData, isLoading: messagesLoading } =
    useSessionMessages(activeSessionId);

  const { sendMessage } = useWebSocketChat({
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
            disabled={isStreaming}
            placeholder={isStreaming ? "Generating response..." : undefined}
          />
        </div>
      </div>
    </div>
  );
}
