import { useState, useEffect } from "react";
import {
  Check,
  X,
  Server,
  Plus,
  Pencil,
  Trash2,
  Loader2,
  ShieldQuestion,
} from "lucide-react";
import {
  approveToolConfirmation,
  rejectToolConfirmation,
  getToolConfirmationStatus,
} from "@/api/client";
import { useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { useChatStore } from "@/stores";
import type { StreamConfirmationRequest } from "@/api/types";
import { cn } from "@/lib/utils";

interface ConfirmationCardProps {
  request: StreamConfirmationRequest;
  /** True when this card comes from a live streaming session (not saved message) */
  isLive?: boolean;
}

const MCP_ACTION_ICONS = {
  mcp_add: Plus,
  mcp_update: Pencil,
  mcp_remove: Trash2,
} as const;

const MCP_ACTION_LABELS = {
  mcp_add: "Add",
  mcp_update: "Update",
  mcp_remove: "Remove",
} as const;

const MCP_ACTION_COLORS = {
  mcp_add: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  mcp_update: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  mcp_remove: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
} as const;

export function ConfirmationCard({ request, isLive }: ConfirmationCardProps) {
  const [status, setStatus] = useState<
    "loading" | "pending" | "approved" | "rejected" | "expired"
  >(isLive ? "pending" : "loading");
  const [resolvedDetails, setResolvedDetails] = useState<Record<string, unknown> | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const resolveConfirmationRequest = useChatStore((s) => s.resolveConfirmationRequest);

  // For saved messages, fetch the real status from backend
  useEffect(() => {
    if (isLive || !request.confirmation_id) {
      if (!isLive && !request.confirmation_id) setStatus("approved");
      return;
    }

    let cancelled = false;
    getToolConfirmationStatus(request.confirmation_id).then((result) => {
      if (cancelled) return;
      if (!result) {
        // Confirmation expired or not found — it was likely already acted on
        setStatus("approved");
        return;
      }
      setStatus(result.status === "pending" ? "pending" : result.status);
      if (result.details) setResolvedDetails(result.details);
    });

    return () => {
      cancelled = true;
    };
  }, [request.confirmation_id, isLive]);

  const isMcp = request.action_type.startsWith("mcp_");
  const ActionIcon =
    (isMcp
      ? MCP_ACTION_ICONS[request.action_type as keyof typeof MCP_ACTION_ICONS]
      : null) || ShieldQuestion;
  const actionLabel = isMcp
    ? MCP_ACTION_LABELS[request.action_type as keyof typeof MCP_ACTION_LABELS] ||
      request.action_type
    : request.action_type;
  const actionColor = isMcp
    ? MCP_ACTION_COLORS[request.action_type as keyof typeof MCP_ACTION_COLORS] || ""
    : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400";

  // Merge details from backend response or use request details
  const details = (resolvedDetails || request.details) as Record<string, unknown>;
  const config = (details.config || {}) as Record<string, unknown>;

  const handleApprove = async () => {
    setLoading(true);
    setError(null);
    try {
      await approveToolConfirmation(request.confirmation_id);
      setStatus("approved");
      resolveConfirmationRequest(request.confirmation_id, "approved");
      // Invalidate caches that may be affected
      if (isMcp) {
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.mcpServers });
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensions });
      }
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
      await rejectToolConfirmation(request.confirmation_id);
      setStatus("rejected");
      resolveConfirmationRequest(request.confirmation_id, "rejected");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reject");
    } finally {
      setLoading(false);
    }
  };

  const headerIcon = isMcp ? Server : ShieldQuestion;
  const HeaderIcon = headerIcon;
  const targetName = isMcp
    ? (details.target_name as string) || request.title
    : request.title;

  return (
    <div className="border-border bg-card my-2 rounded-lg border p-3 shadow-sm">
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <HeaderIcon className="text-muted-foreground h-4 w-4" />
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
            actionColor
          )}
        >
          <ActionIcon className="h-3 w-3" />
          {actionLabel}
        </span>
        <span className="text-sm font-medium">{targetName}</span>
        {status === "loading" && (
          <Loader2 className="text-muted-foreground ml-auto h-3.5 w-3.5 animate-spin" />
        )}
        {(status === "approved" || status === "rejected" || status === "expired") && (
          <span
            className={cn(
              "ml-auto rounded-md px-2 py-0.5 text-xs font-medium",
              status === "approved"
                ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
            )}
          >
            {status === "approved"
              ? "Approved"
              : status === "rejected"
                ? "Rejected"
                : "Expired"}
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="text-muted-foreground mb-2 text-sm">{request.summary}</p>

      {/* MCP-specific config details */}
      {isMcp && request.action_type !== "mcp_remove" && (
        <div className="text-muted-foreground bg-muted/50 mb-2 space-y-0.5 rounded p-2 font-mono text-xs">
          {config.type != null && <div>Type: {String(config.type)}</div>}
          {config.command != null && (
            <div>
              Command: {String(config.command)}{" "}
              {Array.isArray(config.args) && config.args.join(" ")}
            </div>
          )}
          {config.url != null && <div>URL: {String(config.url)}</div>}
          {config.description != null && (
            <div>Description: {String(config.description)}</div>
          )}
          {config.env != null && typeof config.env === "object" && (
            <div>Env: {Object.keys(config.env as Record<string, string>).join(", ")}</div>
          )}
        </div>
      )}

      {/* Generic details for non-MCP confirmations */}
      {!isMcp && Object.keys(details).length > 0 && (
        <div className="text-muted-foreground bg-muted/50 mb-2 space-y-0.5 rounded p-2 font-mono text-xs">
          {Object.entries(details).map(([key, value]) => (
            <div key={key}>
              {key}: {typeof value === "object" ? JSON.stringify(value) : String(value)}
            </div>
          ))}
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
