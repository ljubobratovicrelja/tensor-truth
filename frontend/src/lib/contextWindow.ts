/**
 * Selectable LLM context-window sizes, in tokens.
 *
 * Single source of truth for every context-window dropdown in the UI
 * (global config, session settings, project settings). Keep entries sorted
 * ascending; the backend stores whatever integer the client sends.
 */
export const CONTEXT_WINDOW_OPTIONS = [
  2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
] as const;

export type ContextWindowSize = (typeof CONTEXT_WINDOW_OPTIONS)[number];

/** Format a token count for display, e.g. 1048576 -> "1,048,576". */
export function formatContextWindow(size: number): string {
  return size.toLocaleString("en-US");
}
