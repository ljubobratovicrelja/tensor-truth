import { useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThinkingBoxProps {
  content: string;
  /** Start in collapsed state (for saved messages) */
  isCollapsed?: boolean;
  className?: string;
}

export function ThinkingBox({ content, isCollapsed = false, className }: ThinkingBoxProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(!isCollapsed);

  // Auto-scroll to bottom as content streams in (only when expanded)
  useEffect(() => {
    if (scrollRef.current && expanded && !isCollapsed) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, expanded, isCollapsed]);

  if (!content) return null;

  const toggleExpanded = () => setExpanded((prev) => !prev);

  return (
    <div
      className={cn(
        "border-muted bg-muted/30 mb-3 rounded-lg border",
        className
      )}
    >
      <button
        type="button"
        onClick={toggleExpanded}
        className="flex w-full items-center justify-between gap-2 border-b border-muted px-3 py-2 hover:bg-muted/50 transition-colors"
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
          "overflow-y-auto px-3 py-2 font-mono text-sm transition-all",
          expanded ? "max-h-48" : "max-h-0 py-0 overflow-hidden"
        )}
      >
        <pre className="text-muted-foreground whitespace-pre-wrap">{content}</pre>
      </div>
    </div>
  );
}
