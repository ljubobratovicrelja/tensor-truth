import { memo, useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { MemoizedMarkdown } from "./MemoizedMarkdown";

interface ThinkingBoxProps {
  content: string;
  /** Start in collapsed state (for saved messages) */
  isCollapsed?: boolean;
  /** Thinking is complete (render markdown even if still in streaming session) */
  thinkingComplete?: boolean;
  className?: string;
}

function ThinkingBoxComponent({
  content,
  isCollapsed = false,
  thinkingComplete = false,
  className,
}: ThinkingBoxProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(!isCollapsed);

  // Thinking content is still streaming (not complete, not collapsed)
  const isThinkingStreaming = !isCollapsed && !thinkingComplete;

  // Auto-scroll to bottom as content streams in (only when expanded and streaming)
  useEffect(() => {
    if (scrollRef.current && expanded && isThinkingStreaming) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, expanded, isThinkingStreaming]);

  if (!content) return null;

  const toggleExpanded = () => setExpanded((prev) => !prev);

  return (
    <div className={cn("border-muted bg-muted/30 mb-3 rounded-lg border", className)}>
      <button
        type="button"
        onClick={toggleExpanded}
        className="border-muted hover:bg-muted/50 flex w-full items-center justify-between gap-2 border-b px-3 py-2 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Brain className="text-muted-foreground h-4 w-4" />
          <span className="text-muted-foreground text-sm font-medium">Reasoning</span>
        </div>
        {expanded ? (
          <ChevronUp className="text-muted-foreground h-4 w-4" />
        ) : (
          <ChevronDown className="text-muted-foreground h-4 w-4" />
        )}
      </button>
      <div
        ref={scrollRef}
        className={cn(
          "overflow-y-auto px-3 py-2 text-sm transition-all",
          expanded ? "max-h-48" : "max-h-0 overflow-hidden py-0"
        )}
      >
        {isThinkingStreaming ? (
          <pre className="text-muted-foreground font-mono whitespace-pre-wrap">
            {content}
          </pre>
        ) : (
          <MemoizedMarkdown content={content} className="text-muted-foreground" />
        )}
      </div>
    </div>
  );
}

// Memoize to prevent re-renders of historical thinking boxes during streaming
export const ThinkingBox = memo(ThinkingBoxComponent, (prev, next) => {
  // If thinking is still streaming, always re-render
  const prevStreaming = !prev.isCollapsed && !prev.thinkingComplete;
  const nextStreaming = !next.isCollapsed && !next.thinkingComplete;
  if (prevStreaming || nextStreaming) return false;

  // Once thinking is complete (or saved), only re-render if content changed
  return prev.content === next.content && prev.className === next.className;
});
