import { useEffect, useRef } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { MessageItem } from "./MessageItem";
import { StreamingIndicator } from "./StreamingIndicator";
import type { MessageResponse, SourceNode } from "@/api/types";

interface MessageListProps {
  messages: MessageResponse[];
  isLoading?: boolean;
  streamingContent?: string;
  streamingSources?: SourceNode[];
  isStreaming?: boolean;
}

export function MessageList({
  messages,
  isLoading,
  streamingContent,
  streamingSources,
  isStreaming,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingContent]);

  if (isLoading) {
    return (
      <div className="flex-1 py-4">
        <div className="chat-content-width space-y-4">
          <Skeleton className="h-16 w-3/4" />
          <Skeleton className="ml-auto h-12 w-2/3" />
          <Skeleton className="h-20 w-3/4" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto" ref={scrollRef}>
      <div className="chat-content-width py-4">
        {messages.length === 0 && !isStreaming ? (
          <div className="text-muted-foreground flex h-full min-h-[200px] items-center justify-center">
            <p>Send a message to start the conversation</p>
          </div>
        ) : (
          <div className="space-y-2">
            {messages.map((message, index) => (
              <MessageItem key={index} message={message} />
            ))}
            {isStreaming && !streamingContent && <StreamingIndicator />}
            {isStreaming && streamingContent && (
              <MessageItem
                message={{ role: "assistant", content: streamingContent }}
                sources={streamingSources}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
