import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { PROVIDER_TYPE_LABELS } from "@/lib/constants";
import type { DiscoveredServer } from "@/api/types";

interface DiscoveryBannerProps {
  servers: DiscoveredServer[];
  onAdd: (server: DiscoveredServer) => void;
}

export function DiscoveryBanner({ servers, onAdd }: DiscoveryBannerProps) {
  if (servers.length === 0) return null;

  return (
    <div className="space-y-2">
      {servers.map((server) => (
        <Alert
          key={server.base_url}
          className="border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/20"
        >
          <AlertDescription className="flex items-center justify-between">
            <span className="text-sm">
              Found <strong>{PROVIDER_TYPE_LABELS[server.type] || server.type}</strong> at{" "}
              <code className="rounded bg-blue-100 px-1 text-xs dark:bg-blue-900/50">
                {server.base_url}
              </code>{" "}
              with {server.model_count} model{server.model_count !== 1 ? "s" : ""}
            </span>
            <Button size="sm" variant="outline" onClick={() => onAdd(server)}>
              <Plus className="mr-1 h-3 w-3" />
              Add
            </Button>
          </AlertDescription>
        </Alert>
      ))}
    </div>
  );
}
