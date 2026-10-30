import { HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SelectionRect } from "@/hooks/useTextSelection";

const BUTTON_SIZE = 28;
const GAP = 8;
const MARGIN = 6;

interface ElaborationTriggerProps {
  rect: SelectionRect | null;
  onOpen: () => void;
  /**
   * False while the pointer button is held. The button sits a few pixels past
   * the selection's last line, i.e. exactly where a fast drag is heading, and
   * the browser hit-tests the drag there: landing on a non-text element makes
   * Chromium clamp the selection focus to the enclosing block and select
   * everything below the anchor. Inert while dragging, clickable after.
   */
  interactive?: boolean;
}

/**
 * Floating "?" that follows the end of a selection inside an assistant bubble.
 * `onMouseDown` is prevented so clicking it does not collapse the selection
 * before the click handler reads it.
 */
export function ElaborationTrigger({
  rect,
  onOpen,
  interactive = true,
}: ElaborationTriggerProps) {
  if (!rect) return null;

  const viewportWidth = typeof window === "undefined" ? 1024 : window.innerWidth;
  const viewportHeight = typeof window === "undefined" ? 768 : window.innerHeight;

  let left = rect.right + GAP;
  if (left + BUTTON_SIZE > viewportWidth - MARGIN) {
    left = Math.max(MARGIN, viewportWidth - BUTTON_SIZE - MARGIN);
  }
  // Centre on the last selected line, then keep it inside the viewport.
  let top = rect.top + Math.max(0, (rect.height - BUTTON_SIZE) / 2);
  top = Math.min(Math.max(MARGIN, top), viewportHeight - BUTTON_SIZE - MARGIN);

  return (
    <button
      type="button"
      data-elaboration-trigger=""
      data-testid="elaboration-trigger"
      aria-label="Explain this passage"
      title="Explain this passage"
      onMouseDown={(event) => event.preventDefault()}
      onClick={onOpen}
      style={{ top, left, width: BUTTON_SIZE, height: BUTTON_SIZE }}
      className={cn(
        "border-border bg-background text-muted-foreground hover:bg-primary hover:text-primary-foreground fixed z-50 flex items-center justify-center rounded-full border shadow-md transition-colors select-none",
        !interactive && "pointer-events-none"
      )}
    >
      <HelpCircle className="h-4 w-4" />
    </button>
  );
}
