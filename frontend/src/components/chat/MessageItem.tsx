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
    <div className={cn("flex gap-3 py-4", isUser ? "flex-row-reverse" : "flex-row")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="max-w-[80%] space-y-2">
        {/* Show thinking box for assistant messages */}
        {!isUser && thinkingContent && (
          <ThinkingBox content={thinkingContent} isCollapsed={!isStreaming} />
        )}
        <div
          className={cn(
            "rounded-lg px-4 py-2",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
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
