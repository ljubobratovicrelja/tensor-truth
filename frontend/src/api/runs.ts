import { apiGet, apiPost } from "./client";
import type {
  ActiveRunResponse,
  RunImagePayload,
  RunEventsResponse,
  RunKind,
  RunSummary,
  StartRunResponse,
} from "./types";

/**
 * Server-owned resumable chat runs (docs/RUNS.md).
 *
 * A turn is a server-side run with an append-only, seq-numbered event log. The
 * client POSTs the run, then attaches over the WebSocket from a cursor; a
 * reload or a dropped socket replays from `seq` instead of losing the turn.
 */

export interface StartRunOptions {
  /** Defaults to `chat` server-side. */
  kind?: RunKind;
  /** Assistant message the selection came from (elaboration runs only). */
  message_index?: number | null;
  /** Passage the reader selected (elaboration runs only). */
  selected_text?: string | null;
  occurrence_index?: number;
}

export interface StartRunRequest extends StartRunOptions {
  prompt: string;
  images?: RunImagePayload[];
}

export async function startRun(
  sessionId: string,
  prompt: string,
  images?: RunImagePayload[],
  opts?: StartRunOptions
): Promise<StartRunResponse> {
  const body: StartRunRequest = { prompt, ...opts };
  if (images && images.length > 0) {
    body.images = images;
  }
  return apiPost<StartRunResponse, StartRunRequest>(
    `/sessions/${encodeURIComponent(sessionId)}/runs`,
    body
  );
}

/** Latest non-terminal run for a session, with its full event log. */
export async function getActiveRun(sessionId: string): Promise<ActiveRunResponse> {
  return apiGet<ActiveRunResponse>(
    `/sessions/${encodeURIComponent(sessionId)}/runs/active`
  );
}

/** Run status plus the events with `seq > after` (REST catch-up / polling). */
export async function getRunEvents(runId: string, after = 0): Promise<RunEventsResponse> {
  return apiGet<RunEventsResponse>(
    `/runs/${encodeURIComponent(runId)}?after=${encodeURIComponent(String(after))}`
  );
}

/** Cooperative cancel; the runner emits a terminal event and stops. */
export async function cancelRun(runId: string): Promise<RunSummary> {
  return apiPost<RunSummary>(`/runs/${encodeURIComponent(runId)}/cancel`);
}
