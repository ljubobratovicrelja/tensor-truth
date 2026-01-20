import { useEffect, useState, useCallback } from "react";
import { ArrowDown } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { MessageItem } from "./MessageItem";
import { StreamingIndicator } from "./StreamingIndicator";
import { ThinkingBox } from "./ThinkingBox";
import { useScrollDirection, useIsMobile } from "@/hooks";
import { useUIStore } from "@/stores";
import { cn } from "@/lib/utils";
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
  // Use state for container so effects re-run when it's set
  const [scrollContainer, setScrollContainer] = useState<HTMLDivElement | null>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const isMobile = useIsMobile();
  const setHeaderHidden = useUIStore((state) => state.setHeaderHidden);
  const setInputHidden = useUIStore((state) => state.setInputHidden);

  const { direction, isAtTop, isNearTop, isScrollable, scrollRef } = useScrollDirection({
    threshold: 10,
    topThreshold: 0.1,
  });

  // Combine refs - we need both for scroll tracking and auto-scroll
  const combinedRef = useCallback((node: HTMLDivElement | null) => {
    setScrollContainer(node);
    scrollRef(node);
  }, [scrollRef]);

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

  // Reset header and input when unmounting or switching to desktop
  useEffect(() => {
    return () => {
      setHeaderHidden(false);
      setInputHidden(false);
    };
  }, [setHeaderHidden, setInputHidden]);

  // Update input visibility (mobile) and scroll button visibility (all)
  useEffect(() => {
    if (!scrollContainer) {
      setInputHidden(false);
      setShowScrollButton(false);
      return;
    }

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      const scrollableHeight = scrollHeight - clientHeight;

      // Show input if near bottom (within 10% of scrollable area or 150px)
      const threshold = Math.max(scrollableHeight * 0.1, 150);
      const isNearBottom = distanceFromBottom <= threshold;

      if (isMobile) {
        setInputHidden(!isNearBottom);
      }

      // Show scroll button only when input is hidden (same threshold)
      setShowScrollButton(!isNearBottom);
    };

    // Initial check
    handleScroll();

    scrollContainer.addEventListener("scroll", handleScroll);
    return () => scrollContainer.removeEventListener("scroll", handleScroll);
  }, [isMobile, scrollContainer, setInputHidden]);

  // Auto-scroll to bottom on new content, but only if already near bottom
  useEffect(() => {
    if (!scrollContainer) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // Only auto-scroll if already within 150px of bottom
    // This respects user's position if they've scrolled up
    if (distanceFromBottom < 150) {
      scrollContainer.scrollTop = scrollHeight;
    }
  }, [scrollContainer, messages, pendingUserMessage, streamingContent, streamingThinking]);

  const scrollToBottom = useCallback(() => {
    scrollContainer?.scrollTo({
      top: scrollContainer.scrollHeight,
      behavior: "smooth",
    });
  }, [scrollContainer]);

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
    <div className="relative flex-1 overflow-y-auto" ref={combinedRef}>
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

      {/* Scroll to bottom button */}
      <Button
        onClick={scrollToBottom}
        size="icon"
        variant="secondary"
        className={cn(
          "fixed bottom-6 right-4 z-10 h-8 w-8 rounded-full opacity-60 shadow-md transition-all hover:opacity-100",
          "md:bottom-44 md:right-8",
          showScrollButton
            ? "translate-y-0 scale-100"
            : "pointer-events-none translate-y-4 scale-75 opacity-0"
        )}
        aria-label="Scroll to bottom"
      >
        <ArrowDown className="h-4 w-4" />
      </Button>
    </div>
  );
}
