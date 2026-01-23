import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { MemoizedMarkdown } from "./MemoizedMarkdown";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * Component that renders streaming text with optimized performance.
 *
 * During streaming: renders plain text only (fast DOM updates, no markdown parsing)
 * On stream complete: cross-fades to rendered markdown (single parse)
 */
export function StreamingText({ content, isStreaming, className }: StreamingTextProps) {
  // Track whether we're in the brief transition period after streaming ends
  const [isTransitioning, setIsTransitioning] = useState(false);
  const wasStreamingRef = useRef(isStreaming);

  useEffect(() => {
    // Only trigger transition when streaming stops (true -> false)
    if (wasStreamingRef.current && !isStreaming) {
      // Use queueMicrotask to avoid synchronous setState in effect
      queueMicrotask(() => setIsTransitioning(true));
      const timer = setTimeout(() => setIsTransitioning(false), 200);
      return () => clearTimeout(timer);
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Streaming: plain text only (fast)
  if (isStreaming) {
    return (
      <div className={cn("chat-markdown max-w-none", className)}>
        <div className="whitespace-pre-wrap">{content}</div>
        <span className="streaming-cursor" />
      </div>
    );
  }

  // Transitioning: cross-fade animation
  if (isTransitioning) {
    return (
      <div className={cn("chat-markdown max-w-none relative", className)}>
        <div className="whitespace-pre-wrap animate-fade-out" aria-hidden="true">
          {content}
        </div>
        <div className="absolute inset-0 animate-fade-in">
          <MemoizedMarkdown content={content} />
        </div>
      </div>
    );
  }

  // Rendered: memoized markdown only
  return <MemoizedMarkdown content={content} className={className} />;
}
