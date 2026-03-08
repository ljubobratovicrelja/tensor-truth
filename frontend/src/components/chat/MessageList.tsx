import { useEffect, useState, useCallback, useRef } from "react";
import { ArrowDown } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { MessageItem } from "./MessageItem";
import { StreamingIndicator } from "./StreamingIndicator";
import { AgentProgress } from "./AgentProgress";
import { useScrollDirection, useIsMobile, useAutoScroll } from "@/hooks";
import { useUIStore } from "@/stores";
import { cn } from "@/lib/utils";
import type {
  ImageRef,
  MessageResponse,
  RetrievalMetrics,
  SourceNode,
  StreamAgentProgress,
} from "@/api/types";
import { ToolSteps } from "./ToolSteps";
import type { ToolStepWithStatus } from "./ToolSteps";
import type { PipelineStatus } from "@/stores/chatStore";

interface MessageListProps {
  sessionId: string;
  messages: MessageResponse[];
  isLoading?: boolean;
  pendingUserMessage?: string | null;
  streamingContent?: string;
  streamingThinking?: string;
  streamingSources?: SourceNode[];
  streamingMetrics?: RetrievalMetrics | null;
  streamingToolSteps?: ToolStepWithStatus[];
  isStreaming?: boolean;
  pipelineStatus?: PipelineStatus;
  agentProgress?: StreamAgentProgress | null;
  confidenceLevel?: string | null;
  streamingReasoning?: string;
  pendingUserImages?: (ImageRef & { previewUrl?: string })[] | null;
}

export function MessageList({
  sessionId,
  messages,
  isLoading,
  pendingUserMessage,
  streamingContent,
  streamingThinking,
  streamingSources,
  streamingMetrics,
  streamingToolSteps,
  isStreaming,
  pipelineStatus,
  agentProgress,
  confidenceLevel,
  streamingReasoning,
  pendingUserImages,
}: MessageListProps) {
  const [scrollContainer, setScrollContainer] = useState<HTMLDivElement | null>(null);
  const isMobile = useIsMobile();
  const setHeaderHidden = useUIStore((state) => state.setHeaderHidden);
  const setInputHidden = useUIStore((state) => state.setInputHidden);

  const { direction, isAtTop, isNearTop, isScrollable, scrollRef } = useScrollDirection({
    threshold: 10,
    topThreshold: 0.1,
  });

  // --- Auto-scroll (replaces old threshold-based useEffect) ---
  const {
    scrollRef: autoScrollRef,
    isScrolledAway,
    scrollToBottom,
    reEngage,
  } = useAutoScroll({
    nearBottomThreshold: 50,
    deps: [
      messages,
      pendingUserMessage,
      streamingContent,
      streamingThinking,
      streamingReasoning,
    ],
    enabled: true,
  });

  // Re-engage auto-scroll when a new streaming session starts
  const prevStreamingRef = useRef(false);
  useEffect(() => {
    if (isStreaming && !prevStreamingRef.current) {
      reEngage();
    }
    prevStreamingRef.current = !!isStreaming;
  }, [isStreaming, reEngage]);

  // --- Streaming→complete transition: preserve thinking, dedup ---
  // Keep the last non-empty thinking content so it survives finishStreaming()
  // which clears streamingThinking. The conditional updates are safe (no loops).
  const [savedThinking, setSavedThinking] = useState("");
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: preserve previous value across store clears
    if (streamingThinking) setSavedThinking(streamingThinking);
    if (isStreaming && !streamingContent) setSavedThinking("");
  }, [streamingThinking, isStreaming, streamingContent]);

  // Check if the messages array already contains the streaming message (dedup)
  const streamingDeduplicated = !!(
    streamingContent &&
    messages.some((m) => m.role === "assistant" && m.content === streamingContent)
  );

  // Combine refs - we need scroll direction, auto-scroll, and container state
  const combinedRef = useCallback(
    (node: HTMLDivElement | null) => {
      setScrollContainer(node);
      scrollRef(node);
      autoScrollRef(node);
    },
    [scrollRef, autoScrollRef]
  );

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

  // Update input visibility on mobile (based on scroll position)
  useEffect(() => {
    if (!scrollContainer) {
      setInputHidden(false);
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
    };

    handleScroll();
    scrollContainer.addEventListener("scroll", handleScroll);
    return () => scrollContainer.removeEventListener("scroll", handleScroll);
  }, [isMobile, scrollContainer, setInputHidden]);

  // Only show loading skeleton if we have nothing to display
  // If there's a pending message or streaming content, show that instead
  const hasContentToShow = pendingUserMessage || isStreaming || messages.length > 0;
  if (isLoading && !hasContentToShow) {
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
              <MessageItem
                key={`${sessionId}-${index}`}
                message={message}
                sessionId={sessionId}
              />
            ))}
            {/* Show pending message only if not already in fetched messages (dedup) */}
            {pendingUserMessage &&
              !messages.some(
                (m) => m.role === "user" && m.content === pendingUserMessage
              ) && (
                <MessageItem
                  message={{ role: "user", content: pendingUserMessage }}
                  sessionId={sessionId}
                  pendingImages={pendingUserImages ?? undefined}
                />
              )}
            {/* Streaming: show unified activity box before content appears.
                Thinking is displayed inside ToolPhaseIndicator (via StreamingIndicator)
                alongside reasoning — no separate ThinkingBox needed here. */}
            {isStreaming && !streamingContent && !streamingThinking && (
              <>
                {agentProgress ? (
                  <AgentProgress progress={agentProgress} />
                ) : (
                  <StreamingIndicator status={pipelineStatus} />
                )}
                {streamingToolSteps && streamingToolSteps.length > 0 && (
                  <div className="mb-2 px-4">
                    <ToolSteps steps={streamingToolSteps} defaultOpen />
                  </div>
                )}
              </>
            )}
            {/* Streaming / transition: show streaming response with thinking and status.
                After finishStreaming, streamingContent persists (only reset() clears it)
                so the MessageItem stays mounted until the refetch deduplicates it. */}
            {(streamingContent || streamingThinking) && !streamingDeduplicated && (
              <>
                {agentProgress ? (
                  <AgentProgress progress={agentProgress} />
                ) : (
                  pipelineStatus && <StreamingIndicator status={pipelineStatus} />
                )}
                <MessageItem
                  message={{ role: "assistant", content: streamingContent ?? "" }}
                  sources={streamingSources}
                  metrics={streamingMetrics}
                  thinking={streamingThinking || savedThinking}
                  toolSteps={streamingToolSteps}
                  confidenceLevel={confidenceLevel ?? undefined}
                  isStreaming={!!isStreaming}
                />
              </>
            )}
          </div>
        )}
      </div>

      {/* Scroll to bottom button */}
      <Button
        onClick={() => scrollToBottom()}
        size="icon"
        variant="secondary"
        className={cn(
          "fixed right-4 bottom-6 z-10 h-8 w-8 rounded-full opacity-60 shadow-md transition-all hover:opacity-100",
          "md:right-8 md:bottom-44",
          isScrolledAway
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
