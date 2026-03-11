import { memo, useState, useRef, useEffect } from "react";
import { User, Bot, Copy, Check, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { SourcesList } from "./SourceCard";
import { ToolSteps } from "./ToolSteps";
import type { ToolStepWithStatus } from "./ToolSteps";
import { ThinkingBox } from "./ThinkingBox";
import { StreamingText } from "./StreamingText";
import { MemoizedMarkdown } from "./MemoizedMarkdown";
import type {
  ImageRef,
  MessageResponse,
  RetrievalMetrics,
  SourceNode,
  StreamConfirmationRequest,
} from "@/api/types";
import type { ResponseStats } from "@/stores/chatStore";

interface MessageItemProps {
  message: MessageResponse;
  sources?: SourceNode[];
  metrics?: RetrievalMetrics | null;
  /** Override thinking content (used during streaming) */
  thinking?: string;
  /** Tool steps to display (used during streaming or from saved message) */
  toolSteps?: ToolStepWithStatus[];
  /** Confidence level for RAG sources */
  confidenceLevel?: string;
  /** Whether this message is currently being streamed */
  isStreaming?: boolean;
  /** Session ID for constructing image URLs */
  sessionId?: string;
  /** Pending image preview URLs (optimistic UI before save) */
  pendingImages?: (ImageRef & { previewUrl?: string })[];
  /** Generation stats (shown after streaming completes on the last message) */
  responseStats?: ResponseStats | null;
  /** Active confirmation requests (streaming only) */
  confirmationRequests?: StreamConfirmationRequest[];
  /** Callback to delete this message */
  onDelete?: (messageIndex: number) => void;
  /** Index of this message in the messages array */
  messageIndex?: number;
}

function MessageItemComponent({
  message,
  sources,
  metrics,
  thinking,
  toolSteps,
  confidenceLevel,
  isStreaming,
  sessionId,
  pendingImages,
  responseStats,
  confirmationRequests,
  onDelete,
  messageIndex,
}: MessageItemProps) {
  const isUser = message.role === "user";
  const messageSources = sources ?? (message.sources as SourceNode[] | undefined);
  // Use prop metrics (streaming) or message.metrics (saved)
  const messageMetrics = metrics ?? message.metrics;
  // Use prop thinking (streaming) or message.thinking (saved)
  const thinkingContent = thinking ?? message.thinking;
  // Use prop toolSteps (streaming) or derive from saved message
  const messageToolSteps: ToolStepWithStatus[] =
    toolSteps ??
    message.tool_steps?.map((s) => ({
      ...s,
      status: (s.is_error ? "failed" : "completed") as "failed" | "completed",
    })) ??
    [];
  // Confirmation requests: prefer streaming prop, otherwise empty
  // (saved messages no longer need reconstruction since the tool now blocks
  // and applies changes directly — there's no pending state to reconstruct)
  const activeConfirmations = confirmationRequests ?? [];

  const [copied, setCopied] = useState(false);
  const [userMsgExpanded, setUserMsgExpanded] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const userContentRef = useRef<HTMLDivElement>(null);
  const [userMsgOverflows, setUserMsgOverflows] = useState(false);
  const USER_MSG_COLLAPSED_PX = 200;

  // Detect whether the user message content overflows the collapsed height
  useEffect(() => {
    if (!isUser || !userContentRef.current) return;
    setUserMsgOverflows(userContentRef.current.scrollHeight > USER_MSG_COLLAPSED_PX);
  }, [isUser, message.content]);

  const handleDelete = () => {
    if (onDelete == null || messageIndex == null) return;
    if (!confirm("Delete this message?")) return;
    onDelete(messageIndex);
  };

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
          isUser ? "md:max-w-[80%]" : "w-full md:max-w-[80%]"
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

          {/* Render images for user messages */}
          {isUser && (message.images || pendingImages) && (
            <div className="mb-2 flex flex-wrap gap-2">
              {(pendingImages || message.images)?.map((img) => {
                // Use previewUrl (blob URL) for pending images, server URL for saved
                const src =
                  "previewUrl" in img && (img as { previewUrl?: string }).previewUrl
                    ? (img as { previewUrl: string }).previewUrl
                    : sessionId
                      ? `/api/sessions/${sessionId}/images/${img.id}`
                      : undefined;
                return (
                  <img
                    key={img.id}
                    src={src}
                    alt={img.filename}
                    className="max-h-48 max-w-[200px] rounded-lg object-contain"
                  />
                );
              })}
            </div>
          )}

          {/* Always use StreamingText for assistant messages to isolate parsing errors per-block */}
          {!isUser ? (
            <div ref={contentRef}>
              <StreamingText content={message.content} isStreaming={isStreaming} />
            </div>
          ) : (
            <div ref={contentRef}>
              <div
                ref={userContentRef}
                className="overflow-hidden transition-[max-height] duration-300"
                style={{
                  maxHeight:
                    userMsgOverflows && !userMsgExpanded
                      ? `${USER_MSG_COLLAPSED_PX}px`
                      : undefined,
                }}
              >
                <MemoizedMarkdown
                  content={message.content}
                  className="chat-markdown-user"
                />
              </div>
              {userMsgOverflows && (
                <button
                  onClick={() => setUserMsgExpanded((v) => !v)}
                  className="text-primary-foreground/70 hover:text-primary-foreground mt-1 text-xs font-medium"
                >
                  {userMsgExpanded ? "Show less" : "Show more"}
                </button>
              )}
            </div>
          )}
          {!isUser && (messageSources?.length || messageMetrics) && (
            <SourcesList
              sources={messageSources ?? []}
              metrics={messageMetrics}
              confidenceLevel={confidenceLevel ?? message.confidence_level ?? "normal"}
            />
          )}
          {!isUser && messageToolSteps.length > 0 && (
            <ToolSteps
              steps={messageToolSteps}
              defaultOpen={isStreaming && !message.content && !thinkingContent}
              confirmationRequests={activeConfirmations}
            />
          )}
        </div>
        {/* Action buttons below the bubble */}
        {!isStreaming && (
          <div
            className={cn(
              "flex gap-2 px-1 pt-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100",
              isUser ? "justify-end" : "justify-start"
            )}
          >
            {!isUser && (
              <button
                onClick={handleCopy}
                className="text-muted-foreground hover:text-foreground rounded p-1"
                title="Copy (Shift+click for rich text)"
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            )}
            {onDelete != null && messageIndex != null && (
              <button
                onClick={handleDelete}
                className="text-muted-foreground hover:text-destructive rounded p-1"
                title="Delete message pair"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}
        {!isUser &&
          !isStreaming &&
          responseStats != null &&
          (responseStats.inputTokens > 0 || responseStats.outputTokens > 0) && (
            <div className="text-muted-foreground mt-1 px-1 text-xs">
              {responseStats.inputTokens} input tokens, {responseStats.outputTokens}{" "}
              output, took {responseStats.totalTimeSec}s
              {responseStats.tps != null && <> ({responseStats.tps} tok/s)</>}
            </div>
          )}
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
    prev.metrics === next.metrics &&
    prev.toolSteps === next.toolSteps &&
    prev.confidenceLevel === next.confidenceLevel &&
    prev.sessionId === next.sessionId &&
    prev.pendingImages === next.pendingImages &&
    prev.responseStats === next.responseStats &&
    prev.confirmationRequests === next.confirmationRequests &&
    prev.onDelete === next.onDelete &&
    prev.messageIndex === next.messageIndex
  );
});
