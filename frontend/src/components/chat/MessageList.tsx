import { useEffect, useRef } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { MessageItem } from "./MessageItem";
import { StreamingIndicator } from "./StreamingIndicator";
import { ThinkingBox } from "./ThinkingBox";
import { useScrollDirection, useIsMobile } from "@/hooks";
import { useUIStore } from "@/stores";
import type { MessageResponse, SourceNode } from "@/api/types";
import type { PipelineStatus } from "@/stores/chatStore";

interface MessageListProps {
  messages: MessageResponse[];
  isLoading?: boolean;
  pendingUserMessage?: string | null;
  streamingContent?: string;
  streamingThinking?: string;
  streamingSources?: SourceNode[];
  isStreaming?: boolean;
  pipelineStatus?: PipelineStatus;
}

export function MessageList({
  messages,
  isLoading,
  pendingUserMessage,
  streamingContent,
  streamingThinking,
  streamingSources,
  isStreaming,
  pipelineStatus,
}: MessageListProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isMobile = useIsMobile();
  const setHeaderHidden = useUIStore((state) => state.setHeaderHidden);

  const { direction, isAtTop, isNearTop, isScrollable, scrollRef } = useScrollDirection({
    threshold: 10,
    topThreshold: 0.1,
  });

  // Combine refs - we need both for scroll tracking and auto-scroll
  const combinedRef = (node: HTMLDivElement | null) => {
    scrollContainerRef.current = node;
    scrollRef(node);
  };

  // Update header visibility based on scroll (mobile only)
  useEffect(() => {
    if (!isMobile) {
      setHeaderHidden(false);
      return;
    }

    // Show header unconditionally if at the very top or content not scrollable
    if (isAtTop || !isScrollable) {
      setHeaderHidden(false);
      return;
    }

    // Otherwise: show on scroll up, hide on scroll down
    const shouldHide = direction === "down" && !isNearTop;
    setHeaderHidden(shouldHide);
  }, [direction, isAtTop, isNearTop, isScrollable, isMobile, setHeaderHidden]);

  // Reset header when unmounting or switching to desktop
  useEffect(() => {
    return () => setHeaderHidden(false);
  }, [setHeaderHidden]);

  // Auto-scroll to bottom on new content
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, pendingUserMessage, streamingContent, streamingThinking]);

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
    <div className="flex-1 overflow-y-auto" ref={combinedRef}>
      <div className="chat-content-width py-4">
        {messages.length === 0 && !isStreaming && !pendingUserMessage ? (
          <div className="text-muted-foreground flex h-full min-h-[200px] items-center justify-center">
            <p>Send a message to start the conversation</p>
          </div>
        ) : (
          <div className="space-y-2">
            {messages.map((message, index) => (
              <MessageItem key={index} message={message} />
            ))}
            {/* Show pending message only if not already in fetched messages (dedup) */}
            {pendingUserMessage &&
              !messages.some(
                (m) => m.role === "user" && m.content === pendingUserMessage
              ) && (
                <MessageItem message={{ role: "user", content: pendingUserMessage }} />
              )}
            {/* Streaming: show status indicator and thinking before content appears */}
            {isStreaming && !streamingContent && (
              <>
                {streamingThinking && <ThinkingBox content={streamingThinking} />}
                <StreamingIndicator status={pipelineStatus} />
              </>
            )}
            {/* Streaming: show streaming response with thinking */}
            {isStreaming && streamingContent && (
              <MessageItem
                message={{ role: "assistant", content: streamingContent }}
                sources={streamingSources}
                thinking={streamingThinking}
                isStreaming
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
