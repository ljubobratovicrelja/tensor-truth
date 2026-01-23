import { memo, useState, useRef } from "react";
import { User, Bot, Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { SourcesList } from "./SourceCard";
import { ThinkingBox } from "./ThinkingBox";
import { StreamingText } from "./StreamingText";
import { MemoizedMarkdown } from "./MemoizedMarkdown";
import type { MessageResponse, RetrievalMetrics, SourceNode } from "@/api/types";

interface MessageItemProps {
  message: MessageResponse;
  sources?: SourceNode[];
  metrics?: RetrievalMetrics | null;
  /** Override thinking content (used during streaming) */
  thinking?: string;
  /** Whether this message is currently being streamed */
  isStreaming?: boolean;
}

function MessageItemComponent({
  message,
  sources,
  metrics,
  thinking,
  isStreaming,
}: MessageItemProps) {
  const isUser = message.role === "user";
  const messageSources = sources ?? (message.sources as SourceNode[] | undefined);
  // Use prop metrics (streaming) or message.metrics (saved)
  const messageMetrics = metrics ?? message.metrics;
  // Use prop thinking (streaming) or message.thinking (saved)
  const thinkingContent = thinking ?? message.thinking;
  const [copied, setCopied] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  const handleCopy = async (e: React.MouseEvent) => {
    const content = message.content;

    try {
      if (e.shiftKey && contentRef.current) {
        // Copy as rich text
        if (navigator.clipboard?.write) {
          const html = contentRef.current.innerHTML;
          await navigator.clipboard.write([
            new ClipboardItem({
              "text/html": new Blob([html], { type: "text/html" }),
              "text/plain": new Blob([content], { type: "text/plain" }),
            }),
          ]);
        } else {
          // Fallback: select and copy rendered content
          const range = document.createRange();
          range.selectNodeContents(contentRef.current);
          const selection = window.getSelection();
          selection?.removeAllRanges();
          selection?.addRange(range);
          document.execCommand("copy");
          selection?.removeAllRanges();
        }
      } else {
        // Copy as raw markdown
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(content);
        } else {
          // Fallback: copy via temporary textarea
          const textArea = document.createElement("textarea");
          textArea.value = content;
          textArea.style.position = "fixed";
          textArea.style.left = "-9999px";
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand("copy");
          document.body.removeChild(textArea);
        }
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className={cn("flex gap-3 py-4", isUser ? "flex-row-reverse" : "flex-row")}>
      {isUser && <div className="hidden md:block md:w-[5%] md:shrink-0" />}
      {/* Side icon - hidden on mobile, visible on md+ */}
      <div
        className={cn(
          "hidden h-8 w-8 shrink-0 items-center justify-center rounded-full md:flex",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div
        className={cn(
          "group min-w-0 space-y-2",
          isUser ? "md:max-w-[50%]" : "w-full md:max-w-[80%]"
        )}
      >
        {/* Show thinking box for assistant messages */}
        {!isUser && thinkingContent && (
          <ThinkingBox
            content={thinkingContent}
            isCollapsed={!isStreaming}
            thinkingComplete={isStreaming && message.content.length > 0}
          />
        )}
        <div
          className={cn(
            "relative rounded-2xl px-4 py-3",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
          {/* Copy button for assistant messages */}
          {!isUser && !isStreaming && (
            <button
              onClick={handleCopy}
              className="text-muted-foreground hover:bg-background/50 hover:text-foreground absolute top-2 right-2 rounded p-1 opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
              title="Copy (Shift+click for rich text)"
            >
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            </button>
          )}

          {/* Inline header with icon - visible on mobile only */}
          <div
            className={cn(
              "mb-2 flex items-center gap-1.5 text-xs font-medium md:hidden",
              isUser ? "text-primary-foreground/70" : "text-muted-foreground"
            )}
          >
            <div
              className={cn(
                "flex h-5 w-5 items-center justify-center rounded-full",
                isUser ? "bg-primary-foreground/20" : "bg-background/50"
              )}
            >
              {isUser ? <User className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
            </div>
            <span>{isUser ? "You" : "Assistant"}</span>
          </div>

          {isStreaming ? (
            <StreamingText content={message.content} isStreaming />
          ) : (
            <div ref={contentRef}>
              <MemoizedMarkdown
                content={message.content}
                className={cn(isUser && "chat-markdown-user")}
              />
            </div>
          )}
          {!isUser && (messageSources?.length || messageMetrics) && (
            <SourcesList sources={messageSources ?? []} metrics={messageMetrics} />
          )}
        </div>
      </div>
    </div>
  );
}

// Memoize to prevent re-renders of historical messages during streaming
export const MessageItem = memo(MessageItemComponent, (prev, next) => {
  // For streaming messages, always re-render (content is changing)
  if (prev.isStreaming || next.isStreaming) return false;

  // For historical messages, only re-render if props actually changed
  return (
    prev.message.content === next.message.content &&
    prev.message.role === next.message.role &&
    prev.thinking === next.thinking &&
    prev.sources === next.sources &&
    prev.metrics === next.metrics
  );
});
