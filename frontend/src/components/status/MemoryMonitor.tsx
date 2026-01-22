/**
 * Memory Monitor Component
 *
 * Displays real-time memory usage for various system components
 * (CUDA VRAM, MPS, System RAM, Ollama VRAM)
 */

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useMemoryInfo } from "@/hooks";
import type { MemoryInfo } from "@/api/system";

interface MemoryBarProps {
  info: MemoryInfo;
}

function MemoryBar({ info }: MemoryBarProps) {
  const percentage = info.total_gb
    ? Math.min((info.allocated_gb / info.total_gb) * 100, 100)
    : 0;

  // Color based on usage percentage
  const getColor = (pct: number) => {
    if (pct >= 90) return "bg-red-500";
    if (pct >= 75) return "bg-yellow-500";
    return "bg-green-500";
  };

  // Format memory: use MB for values < 1 GB, otherwise GB
  const formatMemory = (gb: number) => {
    if (gb < 1 && gb > 0) {
      return `${(gb * 1024).toFixed(1)} MB`;
    }
    return `${gb.toFixed(2)} GB`;
  };

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-baseline justify-between">
        <span className="text-foreground text-sm font-medium">{info.name}</span>
        <span className="text-muted-foreground text-xs">
          {formatMemory(info.allocated_gb)}
          {info.total_gb && info.total_gb >= 1 && ` / ${info.total_gb.toFixed(2)} GB`}
          {info.total_gb && percentage > 0 && ` (${percentage.toFixed(0)}%)`}
        </span>
      </div>

      {/* Progress bar */}
      {info.total_gb && info.total_gb > 0 ? (
        <div className="bg-secondary h-2 w-full overflow-hidden rounded-full">
          <div
            className={cn("h-full transition-all duration-300", getColor(percentage))}
            style={{ width: `${percentage}%` }}
          />
        </div>
      ) : (
        <div className="bg-secondary h-2 w-full rounded-full" />
      )}

      {/* Details */}
      {info.details && (
        <p className="text-muted-foreground text-xs italic">{info.details}</p>
      )}
    </div>
  );
}

interface MemoryMonitorProps {
  /** Optional class name for container */
  className?: string;

  /** Whether to show the refresh button */
  showRefresh?: boolean;

  /** Whether to enable polling (only poll when visible) */
  enabled?: boolean;
}

export function MemoryMonitor({
  className,
  showRefresh = true,
  enabled = true,
}: MemoryMonitorProps) {
  const { data, isLoading, isError, refetch, isFetching } = useMemoryInfo(enabled);

  if (isError) {
    return (
      <div className={cn("rounded-lg border p-4", className)}>
        <p className="text-destructive text-sm">Failed to load memory information</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={cn("rounded-lg border p-4", className)}>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2">
              <div className="bg-muted h-4 w-32 animate-pulse rounded" />
              <div className="bg-muted h-2 w-full animate-pulse rounded-full" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const memoryData = data?.memory || [];

  return (
    <div className={cn("rounded-lg border p-4", className)}>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-foreground text-sm font-semibold">Memory Usage</h3>
        {showRefresh && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => refetch()}
            disabled={isFetching}
            title="Refresh memory stats"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
          </Button>
        )}
      </div>

      {/* Memory bars */}
      {memoryData.length === 0 ? (
        <p className="text-muted-foreground text-sm">No memory information available</p>
      ) : (
        <div className="space-y-4">
          {memoryData.map((info, idx) => (
            <MemoryBar key={idx} info={info} />
          ))}
        </div>
      )}

      {/* Auto-refresh indicator */}
      <p className="text-muted-foreground mt-4 text-xs">Updates every 5 seconds</p>
    </div>
  );
}
