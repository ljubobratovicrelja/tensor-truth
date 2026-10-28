import { HelpCircle, Loader2, Square, X } from "lucide-react";
import { useElaborationStore } from "@/stores/elaborationStore";
import { StreamingText } from "./StreamingText";

interface ElaborationPanelProps {
  /** Cancels the in-flight run (used while streaming). */
  onCancel?: () => void;
}

/**
 * Body of the elaboration sidebar: the quoted passage and the streamed
 * explanation. All state comes from `useElaborationStore`, never the chat store.
 */
export function ElaborationPanel({ onCancel }: ElaborationPanelProps) {
  const anchor = useElaborationStore((state) => state.anchor);
  const content = useElaborationStore((state) => state.content);
  const thinking = useElaborationStore((state) => state.thinking);
  const isStreaming = useElaborationStore((state) => state.isStreaming);
  const error = useElaborationStore((state) => state.error);
  const close = useElaborationStore((state) => state.close);

  if (!anchor) return null;

  const handleClose = () => {
    if (isStreaming) onCancel?.();
    close();
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

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <blockquote className="border-primary/40 text-muted-foreground line-clamp-3 border-l-2 pl-3 text-sm italic">
          {anchor.selectedText}
        </blockquote>

        {error ? (
          <p className="text-destructive text-sm" role="alert">
            {error}
          </p>
        ) : content ? (
          <div className="text-sm leading-relaxed">
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

        {isStreaming && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs"
          >
            <Square className="h-3 w-3" />
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
