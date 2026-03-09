import { useState, useEffect } from "react";
import { Check, X, Server, Plus, Pencil, Trash2, Loader2 } from "lucide-react";
import {
  approveMcpProposal,
  rejectMcpProposal,
  getMcpProposalStatus,
} from "@/api/client";
import { useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import type { StreamApprovalRequest } from "@/api/types";
import { cn } from "@/lib/utils";

interface McpApprovalCardProps {
  request: StreamApprovalRequest;
  /** True when this card comes from a live streaming session (not saved message) */
  isLive?: boolean;
}

const ACTION_ICONS = {
  add: Plus,
  update: Pencil,
  remove: Trash2,
} as const;

const ACTION_LABELS = {
  add: "Add",
  update: "Update",
  remove: "Remove",
} as const;

const ACTION_COLORS = {
  add: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  update: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  remove: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
} as const;

export function McpApprovalCard({ request, isLive }: McpApprovalCardProps) {
  const [status, setStatus] = useState<"loading" | "pending" | "approved" | "rejected">(
    isLive ? "pending" : "loading"
  );
  const [resolvedConfig, setResolvedConfig] = useState<Record<string, unknown> | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // For saved messages, fetch the real proposal status from backend
  useEffect(() => {
    if (isLive || !request.proposal_id) {
      // Live streaming cards are always pending; no proposal_id means we can't check
      if (!isLive && !request.proposal_id) setStatus("approved");
      return;
    }

    let cancelled = false;
    getMcpProposalStatus(request.proposal_id).then((result) => {
      if (cancelled) return;
      if (!result) {
        // Proposal expired or not found — it was likely already acted on
        setStatus("approved");
        return;
      }
      setStatus(result.status === "pending" ? "pending" : result.status);
      // Use the backend config (has auto-filled values) instead of raw LLM params
      if (result.config) setResolvedConfig(result.config);
    });

    return () => {
      cancelled = true;
    };
  }, [request.proposal_id, isLive]);

  const ActionIcon = ACTION_ICONS[request.action] || Server;
  const actionLabel = ACTION_LABELS[request.action] || request.action;
  const actionColor = ACTION_COLORS[request.action] || "";

  // Prefer backend-resolved config (auto-filled), fall back to request config
  const config = (resolvedConfig || request.config) as Record<string, unknown>;

  const handleApprove = async () => {
    setLoading(true);
    setError(null);
    try {
      await approveMcpProposal(request.proposal_id);
      setStatus("approved");
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.mcpServers });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensions });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to approve");
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async () => {
    setLoading(true);
    setError(null);
    try {
      await rejectMcpProposal(request.proposal_id);
      setStatus("rejected");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reject");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-border bg-card my-2 rounded-lg border p-3 shadow-sm">
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <Server className="text-muted-foreground h-4 w-4" />
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
            actionColor
          )}
        >
          <ActionIcon className="h-3 w-3" />
          {actionLabel}
        </span>
        <span className="text-sm font-medium">{request.target_name}</span>
        {status === "loading" && (
          <Loader2 className="text-muted-foreground ml-auto h-3.5 w-3.5 animate-spin" />
        )}
        {(status === "approved" || status === "rejected") && (
          <span
            className={cn(
              "ml-auto rounded-md px-2 py-0.5 text-xs font-medium",
              status === "approved"
                ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
            )}
          >
            {status === "approved" ? "Approved" : "Rejected"}
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="text-muted-foreground mb-2 text-sm">{request.summary}</p>

      {/* Config details (for add/update) */}
      {request.action !== "remove" && (
        <div className="text-muted-foreground bg-muted/50 mb-2 space-y-0.5 rounded p-2 font-mono text-xs">
          {config.type && <div>Type: {String(config.type)}</div>}
          {config.command && (
            <div>
              Command: {String(config.command)}{" "}
              {Array.isArray(config.args) && config.args.join(" ")}
            </div>
          )}
          {config.url && <div>URL: {String(config.url)}</div>}
          {config.description && <div>Description: {String(config.description)}</div>}
          {config.env && typeof config.env === "object" && (
            <div>Env: {Object.keys(config.env as Record<string, string>).join(", ")}</div>
          )}
        </div>
      )}

      {/* Error message */}
      {error && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{error}</p>}

      {/* Buttons */}
      {status === "pending" && (
        <div className="flex gap-2">
          <button
            onClick={handleApprove}
            disabled={loading}
            className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
          >
            <Check className="h-3 w-3" />
            Approve
          </button>
          <button
            onClick={handleReject}
            disabled={loading}
            className="border-border text-muted-foreground hover:bg-muted inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium disabled:opacity-50"
          >
            <X className="h-3 w-3" />
            Reject
          </button>
        </div>
      )}
    </div>
  );
}
