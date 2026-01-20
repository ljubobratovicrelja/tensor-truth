import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { User, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { SourcesList } from "./SourceCard";
import { ThinkingBox } from "./ThinkingBox";
import { StreamingText } from "./StreamingText";
import type { MessageResponse, SourceNode } from "@/api/types";

interface MessageItemProps {
  message: MessageResponse;
  sources?: SourceNode[];
  /** Override thinking content (used during streaming) */
  thinking?: string;
  /** Whether this message is currently being streamed */
  isStreaming?: boolean;
}

export function MessageItem({ message, sources, thinking, isStreaming }: MessageItemProps) {
  const isUser = message.role === "user";
  const messageSources = sources ?? (message.sources as SourceNode[] | undefined);
  // Use prop thinking (streaming) or message.thinking (saved)
  const thinkingContent = thinking ?? message.thinking;

  return (
    <div className={cn("flex gap-3 py-4", isUser ? "md:flex-row-reverse" : "md:flex-row")}>
      {/* Side icon - hidden on mobile, visible on md+ */}
      <div
        className={cn(
          "hidden h-8 w-8 shrink-0 items-center justify-center rounded-full md:flex",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className="min-w-0 flex-1 space-y-2 md:max-w-[80%] md:flex-initial">
        {/* Show thinking box for assistant messages */}
        {!isUser && thinkingContent && (
          <ThinkingBox content={thinkingContent} isCollapsed={!isStreaming} />
        )}
        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
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
            <div className="chat-markdown max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[
                  rehypeHighlight,
                  rehypeKatex,
                  rehypeSlug,
                  [
                    rehypeAutolinkHeadings,
                    {
                      behavior: "wrap",
                      properties: {
                        className: ["header-anchor"],
                      },
                    },
                  ],
                ]}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
          {!isUser && messageSources && messageSources.length > 0 && (
            <SourcesList sources={messageSources} />
          )}
        </div>
      </div>
    </div>
  );
}
