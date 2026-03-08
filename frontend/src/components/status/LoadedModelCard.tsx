import { Badge } from "@/components/ui/badge";

interface LoadedModelCardProps {
  name: string;
  /** VRAM usage in GB */
  vramGb?: number;
  /** Total model size in GB */
  sizeGb?: number;
  /** Parameter count label (e.g. "8B") */
  parameters?: string | null;
  /** Context window size */
  contextLength?: number | null;
}

function formatNumber(n: number) {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return n.toString();
}

export function LoadedModelCard({
  name,
  vramGb,
  sizeGb,
  parameters,
  contextLength,
}: LoadedModelCardProps) {
  const hasStats =
    (vramGb != null && vramGb > 0) || (sizeGb != null && sizeGb > 0) || !!parameters;

  return (
    <div className="rounded border p-2">
      <div className="flex items-center justify-between">
        <p className="font-mono text-sm">{name}</p>
        {contextLength != null && contextLength > 0 && (
          <Badge variant="outline" className="text-xs">
            {formatNumber(contextLength)} ctx
          </Badge>
        )}
      </div>
      {hasStats && (
        <div className="text-muted-foreground mt-1 flex gap-3 text-xs">
          {vramGb != null && vramGb > 0 && <span>VRAM: {vramGb.toFixed(2)} GB</span>}
          {sizeGb != null && sizeGb > 0 && <span>Size: {sizeGb.toFixed(2)} GB</span>}
          {parameters && <span>{parameters}</span>}
        </div>
      )}
    </div>
  );
}
