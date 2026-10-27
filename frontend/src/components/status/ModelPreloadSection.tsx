/**
 * RAG model preload rows for the System Status panel.
 *
 * Shows one row per model (embedding encoder, reranker) with the background
 * load state: spinner + current step while loading, model name/device/time when
 * ready, and the error message when the load failed.
 */

import { AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ModelLoadState, ModelPreloadStatusResponse } from "@/api/system";

interface ModelPreloadSectionProps {
  status?: ModelPreloadStatusResponse;
  isLoading?: boolean;
}

function formatModelName(name: string | null) {
  if (!name) return "Unknown";
  return name.split("/").pop() || name;
}

function formatSeconds(seconds: number) {
  return `${seconds.toFixed(1)}s`;
}

function StateIcon({ state }: { state: ModelLoadState["state"] }) {
  if (state === "loading") {
    return (
      <Loader2
        data-testid="model-state-spinner"
        className="text-primary h-3.5 w-3.5 animate-spin"
        aria-hidden="true"
      />
    );
  }
  if (state === "ready") {
    return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" aria-hidden="true" />;
  }
  if (state === "error") {
    return <AlertCircle className="text-destructive h-3.5 w-3.5" aria-hidden="true" />;
  }
  return <Clock className="text-muted-foreground h-3.5 w-3.5" aria-hidden="true" />;
}

function ModelRow({ label, model }: { label: string; model: ModelLoadState }) {
  return (
    <div className="rounded border p-2" data-testid={`model-row-${model.kind}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-muted-foreground flex items-center gap-1.5 text-xs font-medium">
          <StateIcon state={model.state} />
          {label}
        </span>
        {model.device && model.state !== "idle" && (
          <Badge variant="outline" className="text-xs">
            {model.device.toUpperCase()}
          </Badge>
        )}
      </div>

      {model.state === "idle" && (
        <p className="text-muted-foreground mt-1 text-xs">Not loaded yet</p>
      )}

      {model.state === "loading" && (
        <div className="mt-1 space-y-0.5">
          <p className="font-mono text-sm">{formatModelName(model.model_name)}</p>
          <p className="text-muted-foreground text-xs">
            {model.step || "loading"} · {formatSeconds(model.elapsed_seconds)}
          </p>
        </div>
      )}

      {model.state === "ready" && (
        <div className="mt-1 space-y-0.5">
          <p className="font-mono text-sm">{formatModelName(model.model_name)}</p>
          <p className="text-muted-foreground text-xs">
            loaded in {formatSeconds(model.elapsed_seconds)}
          </p>
        </div>
      )}

      {model.state === "error" && (
        <p className="text-destructive mt-1 text-xs" role="alert">
          {model.error || "Failed to load model"}
        </p>
      )}
    </div>
  );
}

export function ModelPreloadSection({ status, isLoading }: ModelPreloadSectionProps) {
  if (isLoading && !status) {
    return (
      <div className="space-y-2">
        <div className="bg-muted h-14 w-full animate-pulse rounded" />
        <div className="bg-muted h-14 w-full animate-pulse rounded" />
      </div>
    );
  }

  if (!status) return null;

  const badge = status.all_ready
    ? { variant: "default" as const, label: "Ready" }
    : status.any_loading
      ? { variant: "secondary" as const, label: "Warming up" }
      : status.embedder.state === "error" || status.reranker.state === "error"
        ? { variant: "destructive" as const, label: "Load failed" }
        : { variant: "secondary" as const, label: "Not loaded" };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Badge variant={badge.variant}>{badge.label}</Badge>
        {status.any_loading && (
          <span className="text-muted-foreground text-xs">
            models load in the background
          </span>
        )}
      </div>
      <ModelRow label="Embedding model" model={status.embedder} />
      <ModelRow label="Reranker" model={status.reranker} />
    </div>
  );
}
