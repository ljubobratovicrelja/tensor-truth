import { CornerDownRight } from "lucide-react";
import { MemoizedMarkdown } from "./MemoizedMarkdown";

interface NarrationBlockProps {
  content: string;
}

/**
 * One intermediate step the orchestrator narrated while deciding to use tools.
 *
 * These blocks deliberately sit outside the assistant answer bubble: narrative
 * text is not part of the final answer and must not merge with the next step or
 * with the answer's paragraphs.
 */
export function NarrationBlock({ content }: NarrationBlockProps) {
  return (
    <div
      data-testid="narration-block"
      className="border-border/60 bg-muted/20 text-muted-foreground rounded-r-xl border-y border-r border-l-2 border-l-amber-500/50 px-3 py-2"
    >
      <div className="text-muted-foreground/70 mb-1 flex items-center gap-1.5 text-[11px] font-medium tracking-wide uppercase">
        <CornerDownRight className="h-3 w-3" />
        <span>Narration</span>
      </div>
      <MemoizedMarkdown content={content} className="chat-markdown-narration" />
    </div>
  );
}
