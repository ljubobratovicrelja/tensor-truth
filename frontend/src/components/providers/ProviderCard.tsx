import { Pencil, Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PROVIDER_TYPE_LABELS, PROVIDER_TYPE_COLORS } from "@/lib/constants";
import type { ProviderResponse } from "@/api/types";

interface ProviderCardProps {
  provider: ProviderResponse;
  onEdit: () => void;
  onDelete: () => void;
  isDeleting?: boolean;
}

export function ProviderCard({
  provider,
  onEdit,
  onDelete,
  isDeleting,
}: ProviderCardProps) {
  const connected = provider.status === "connected";

  return (
    <div className="bg-card flex items-center justify-between rounded-lg border p-4">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="font-medium">{provider.id}</span>
          <Badge
            variant="secondary"
            className={PROVIDER_TYPE_COLORS[provider.type] || ""}
          >
            {PROVIDER_TYPE_LABELS[provider.type] || provider.type}
          </Badge>
        </div>
        <div className="text-muted-foreground truncate text-sm">{provider.base_url}</div>
        <div className="flex items-center gap-1.5 text-sm">
          <span
            className={`inline-block h-2 w-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
          />
          <span
            className={
              connected
                ? "text-green-600 dark:text-green-400"
                : "text-red-600 dark:text-red-400"
            }
          >
            {connected
              ? `Connected (${provider.model_count} model${provider.model_count !== 1 ? "s" : ""})`
              : "Unreachable"}
          </span>
        </div>
      </div>
      <div className="ml-3 flex items-center gap-1">
        <Button variant="ghost" size="icon" onClick={onEdit} title="Edit provider">
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          disabled={isDeleting}
          title="Remove provider"
        >
          {isDeleting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Trash2 className="text-destructive h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
