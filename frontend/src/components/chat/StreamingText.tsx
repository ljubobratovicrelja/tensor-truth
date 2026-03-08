import { useState, useEffect, useRef, useMemo } from "react";
import { cn } from "@/lib/utils";
import { MemoizedMarkdown } from "./MemoizedMarkdown";
import { splitMarkdown, EMPTY_RESULT, type SplitResult } from "@/lib/markdownSplitter";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * Hook that coalesces streaming content updates and splits markdown
 * into stable/unstable blocks at most once per animation frame.
 */
function useStreamingSplit(content: string, isStreaming: boolean): SplitResult {
  // For non-streaming (historical) messages, compute directly via useMemo
  const staticResult = useMemo(
    () => (content && !isStreaming ? splitMarkdown(content) : null),
    [content, isStreaming]
  );

  // Streaming state: updated via rAF coalescing
  const [streamResult, setStreamResult] = useState<SplitResult>(EMPTY_RESULT);
  const pendingRef = useRef(content);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      // Cancel any pending rAF when streaming stops
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
      }
      return;
    }

    pendingRef.current = content;
    if (rafRef.current) return; // coalesce — just update ref, skip scheduling

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      setStreamResult(splitMarkdown(pendingRef.current));
    });
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
      }
    };
  }, [content, isStreaming]);

  return staticResult ?? streamResult;
}

/**
 * Component that renders streaming text with progressive markdown blocks.
 *
 * Uses remark AST splitting: all blocks except the last are stable (cached,
 * never re-rendered). The last block is unstable and re-renders each frame
 * to show partial content immediately — no hidden pending buffer.
 */
export function StreamingText({ content, isStreaming, className }: StreamingTextProps) {
  const { stableBlocks, unstableBlock } = useStreamingSplit(
    content,
    isStreaming ?? false
  );

  return (
    <div className={cn("streaming-blocks", isStreaming && "is-streaming", className)}>
      {stableBlocks.map((block, i) => (
        <MemoizedMarkdown key={i} content={block} />
      ))}
      {unstableBlock && <MemoizedMarkdown content={unstableBlock} />}
    </div>
  );
}
