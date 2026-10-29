import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { HelpCircle, Loader2, Send, Square, X } from "lucide-react";
import { useElaborationStore } from "@/stores/elaborationStore";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { StreamingText } from "./StreamingText";

interface ElaborationPanelProps {
  /** Cancels the in-flight run (used while streaming). */
  onCancel?: () => void;
  /** Sends a follow-up turn in the open elaboration's thread. */
  onFollowUp?: (text: string) => void;
}

/**
 * Body of the elaboration sidebar: the quoted passage, the original
 * explanation, the follow-up thread, and a compact composer. All state comes
 * from `useElaborationStore`, never the chat store.
 */
export function ElaborationPanel({ onCancel, onFollowUp }: ElaborationPanelProps) {
  const anchor = useElaborationStore((state) => state.anchor);
  const content = useElaborationStore((state) => state.content);
  const thinking = useElaborationStore((state) => state.thinking);
  const isStreaming = useElaborationStore((state) => state.isStreaming);
  const error = useElaborationStore((state) => state.error);
  const activeElaborationId = useElaborationStore((state) => state.activeElaborationId);
  const thread = useElaborationStore((state) => state.thread);
  const reply = useElaborationStore((state) => state.reply);
  const isReplying = useElaborationStore((state) => state.isReplying);
  const followUpError = useElaborationStore((state) => state.followUpError);
  const close = useElaborationStore((state) => state.close);

  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest turn (or the streaming tail) in view.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [content, thread.length, reply, isReplying, isStreaming, followUpError]);

  if (!anchor) return null;

  const busy = isStreaming || isReplying;
  // A follow-up needs a persisted elaboration to attach to.
  const canSend = Boolean(activeElaborationId) && !busy && draft.trim().length > 0;

  const handleClose = () => {
    // Closing is not cancelling: the run keeps going and the underline stays.
    // Stop is the explicit way to end it.
    close();
  };

  const send = () => {
    const text = draft.trim();
    if (!text || !activeElaborationId || busy) return;
    setDraft("");
    onFollowUp?.(text);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    send();
  };

  return (
    <div className="flex h-full flex-col" data-testid="elaboration-panel">
      <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <HelpCircle className="text-muted-foreground h-4 w-4" />
          <h2 className="text-sm font-medium">Passage explained</h2>
        </div>
        <button
          type="button"
          onClick={handleClose}
          aria-label="Close explanation"
          title="Close"
          className="text-muted-foreground hover:text-foreground rounded p-1"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div
        ref={scrollRef}
        data-testid="elaboration-thread"
        className="flex-1 space-y-3 overflow-y-auto px-4 py-4"
      >
        <blockquote className="border-primary/40 text-muted-foreground line-clamp-3 border-l-2 pl-3 text-sm italic">
          {anchor.selectedText}
        </blockquote>

        {error ? (
          <p className="text-destructive text-sm" role="alert">
            {error}
          </p>
        ) : content ? (
          <div className="text-sm leading-relaxed" data-testid="elaboration-explanation">
            <StreamingText content={content} isStreaming={isStreaming} />
          </div>
        ) : isStreaming ? (
          <div className="text-muted-foreground flex items-center gap-2 text-sm">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>
              {thinking ? "Thinking about this passage..." : "Reading the passage..."}
            </span>
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">No explanation was returned.</p>
        )}

        {thread.map((message, index) => (
          <div
            key={index}
            data-testid={`elaboration-turn-${message.role}`}
            className={cn(
              "max-w-[92%] rounded-lg px-3 py-2 text-sm leading-relaxed",
              message.role === "user"
                ? "bg-muted ml-auto whitespace-pre-wrap"
                : "bg-muted/40 mr-auto"
            )}
          >
            {message.role === "assistant" ? (
              <StreamingText content={message.content} />
            ) : (
              message.content
            )}
          </div>
        ))}

        {isReplying && (
          <div
            data-testid="elaboration-reply"
            className="bg-muted/40 mr-auto max-w-[92%] rounded-lg px-3 py-2 text-sm leading-relaxed"
          >
            {reply ? (
              <StreamingText content={reply} isStreaming />
            ) : (
              <span className="text-muted-foreground flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Thinking...
              </span>
            )}
          </div>
        )}

        {followUpError && (
          <p className="text-destructive text-sm" role="alert">
            {followUpError}
          </p>
        )}
      </div>

      <div className="border-t px-3 py-3" data-testid="elaboration-composer">
        <div className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            disabled={!activeElaborationId || busy}
            placeholder={
              activeElaborationId ? "Ask a follow-up…" : "Preparing the explanation…"
            }
            aria-label="Ask a follow-up"
            className="border-input bg-background placeholder:text-muted-foreground focus-visible:ring-ring/50 min-h-[60px] flex-1 resize-none rounded-md border px-2.5 py-2 text-sm outline-none focus-visible:ring-[3px] disabled:opacity-60"
          />
          {busy ? (
            <Button
              type="button"
              size="icon"
              variant="outline"
              onClick={onCancel}
              aria-label="Stop"
              title="Stop"
            >
              <Square className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              type="button"
              size="icon"
              onClick={send}
              disabled={!canSend}
              aria-label="Send follow-up"
              title="Send"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
