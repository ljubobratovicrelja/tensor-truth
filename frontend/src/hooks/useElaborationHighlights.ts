import { useEffect } from "react";
import type { MessageResponse } from "@/api/types";
import type { ElaborationAnchor } from "@/stores/elaborationStore";
import { rangeForOccurrence } from "@/lib/elaborationText";

export const ELABORATION_HIGHLIGHT = "tt-elaboration";
export const ACTIVE_ELABORATION_HIGHLIGHT = "tt-elaboration-active";

export interface ElaborationHighlightSpec {
  messageIndex: number;
  selectedText: string;
  occurrenceIndex: number;
}

/**
 * Every stored elaboration, flattened to what the highlighter needs. Anchors
 * are derived from the server payload so a refetch stays the single source of
 * truth.
 */
export function collectElaborationSpecs(
  messages: MessageResponse[] | undefined | null
): ElaborationHighlightSpec[] {
  if (!messages) return [];
  const specs: ElaborationHighlightSpec[] = [];
  messages.forEach((message, messageIndex) => {
    if (message.role !== "assistant") return;
    for (const elaboration of message.elaborations ?? []) {
      if (!elaboration?.selected_text) continue;
      specs.push({
        messageIndex,
        selectedText: elaboration.selected_text,
        occurrenceIndex: elaboration.occurrence_index ?? 0,
      });
    }
  });
  return specs;
}

/**
 * Resolve the `occurrenceIndex`-th occurrence of `text` inside the rendered
 * text of `container` to a DOM Range. Returns null when the passage no longer
 * exists (e.g. the message was edited or re-rendered without it).
 */
export function resolveTextRange(
  container: Node,
  text: string,
  occurrenceIndex = 0
): Range | null {
  return rangeForOccurrence(container, text, occurrenceIndex);
}

interface HighlightRegistryLike {
  set(name: string, highlight: unknown): void;
  delete(name: string): boolean;
}

type HighlightCtor = new (...ranges: Range[]) => unknown;

function getHighlightApi(): {
  registry: HighlightRegistryLike;
  Highlight: HighlightCtor;
} | null {
  if (typeof CSS === "undefined" || typeof document === "undefined") return null;
  const registry = (CSS as unknown as { highlights?: HighlightRegistryLike }).highlights;
  const Highlight = (globalThis as unknown as { Highlight?: HighlightCtor }).Highlight;
  if (!registry || typeof Highlight !== "function") return null;
  return { registry, Highlight };
}

function contentFor(messageIndex: number): HTMLElement | null {
  const bubble = document.querySelector<HTMLElement>(
    `[data-chat-message="${messageIndex}"][data-role="assistant"]`
  );
  return bubble?.querySelector<HTMLElement>("[data-message-content]") ?? null;
}

/**
 * Mark every stored elaboration with the default underline, and the passage
 * currently open in the panel with the active tint. Uses the CSS Custom
 * Highlight API; where it is unsupported the trigger still works, only the
 * markers are missing.
 */
export function useElaborationHighlights(
  messages: MessageResponse[] | undefined | null,
  activeAnchor: ElaborationAnchor | null
): void {
  useEffect(() => {
    const api = getHighlightApi();
    if (!api) return;
    const { registry, Highlight } = api;

    const ranges: Range[] = [];
    for (const spec of collectElaborationSpecs(messages)) {
      const container = contentFor(spec.messageIndex);
      if (!container) continue;
      const range = resolveTextRange(container, spec.selectedText, spec.occurrenceIndex);
      if (range) ranges.push(range);
    }

    if (ranges.length > 0) {
      registry.set(ELABORATION_HIGHLIGHT, new Highlight(...ranges));
    } else {
      registry.delete(ELABORATION_HIGHLIGHT);
    }

    if (activeAnchor) {
      const container = contentFor(activeAnchor.messageIndex);
      const range = container
        ? resolveTextRange(
            container,
            activeAnchor.selectedText,
            activeAnchor.occurrenceIndex
          )
        : null;
      if (range) {
        registry.set(ACTIVE_ELABORATION_HIGHLIGHT, new Highlight(range));
      } else {
        registry.delete(ACTIVE_ELABORATION_HIGHLIGHT);
      }
    } else {
      registry.delete(ACTIVE_ELABORATION_HIGHLIGHT);
    }

    return () => {
      registry.delete(ELABORATION_HIGHLIGHT);
      registry.delete(ACTIVE_ELABORATION_HIGHLIGHT);
    };
  }, [messages, activeAnchor]);
}
