import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { StreamingBlockRenderer } from "./StreamingBlockRenderer";
import {
  parseMarkdownBlocks,
  finalizeState,
  initialParserState,
  type ParserState,
} from "@/lib/markdownBlockParser";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * Component that renders text with progressive markdown blocks.
 *
 * - Parses content into logical markdown blocks
 * - Renders completed blocks with markdown formatting and animation
 * - Up to 2 blocks can animate concurrently for smoother appearance
 * - Keeps block-by-block rendering permanently (isolates parsing errors)
 * - Works for both streaming and non-streaming (historical) content
 */
export function StreamingText({
  content,
  isStreaming,
  className,
}: StreamingTextProps) {
  // Track if this message was ever streamed (vs loaded as history)
  // If it started as non-streaming with content, it's historical - no animations
  const [shouldAnimate] = useState(() => isStreaming === true || !content);

  // Lazy initial state: for non-streaming content, parse everything upfront
  const [parserState, setParserState] = useState<ParserState>(() => {
    if (!isStreaming && content) {
      const parsed = parseMarkdownBlocks(initialParserState, content);
      return finalizeState(parsed);
    }
    return initialParserState;
  });
  const prevContentRef = useRef(!isStreaming && content ? content : "");
  const wasStreamingRef = useRef(isStreaming);

  // Process new tokens as they arrive during streaming
  useEffect(() => {
    if (isStreaming && content !== prevContentRef.current) {
      // Calculate what's new since last update
      const newToken = content.slice(prevContentRef.current.length);
      prevContentRef.current = content;

      if (newToken) {
        setParserState((prev) => parseMarkdownBlocks(prev, newToken));
      }
    }
  }, [content, isStreaming]);

  // Reset state when starting new stream
  useEffect(() => {
    if (isStreaming && content === "" && prevContentRef.current !== "") {
      queueMicrotask(() => {
        setParserState(initialParserState);
        prevContentRef.current = "";
      });
    }
  }, [isStreaming, content]);

  // Handle transition when streaming ends - finalize pending content
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming) {
      // Finalize any pending content
      queueMicrotask(() => setParserState((prev) => finalizeState(prev)));
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Keep block-by-block rendering permanently - no final re-render needed
  // This isolates any math/parsing errors to individual blocks
  // Animate if this was ever a streaming message (not historical)
  return (
    <div className={cn("chat-markdown max-w-none", className)}>
      <StreamingBlockRenderer parserState={parserState} animate={shouldAnimate} />
    </div>
  );
}
