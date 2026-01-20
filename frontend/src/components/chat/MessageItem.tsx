import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { User, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { SourcesList } from "./SourceCard";
import type { MessageResponse, SourceNode } from "@/api/types";

interface MessageItemProps {
  message: MessageResponse;
  sources?: SourceNode[];
}

export function MessageItem({ message, sources }: MessageItemProps) {
  const isUser = message.role === "user";
  const messageSources = sources ?? (message.sources as SourceNode[] | undefined);

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
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-2",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
            {message.content}
          </ReactMarkdown>
        </div>
        {!isUser && messageSources && messageSources.length > 0 && (
          <SourcesList sources={messageSources} />
        )}
      </div>
    </div>
  );
}
