import { useState } from "react";
import { ChevronDown, ChevronUp, Globe, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { WebSearchSource } from "@/api/types";

interface WebSearchSourcesProps {
  sources: WebSearchSource[];
}

function getStatusBadge(status: WebSearchSource["status"]) {
  switch (status) {
    case "success":
      return (
        <Badge variant="default" className="bg-green-600 text-xs">
          Fetched
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="destructive" className="text-xs">
          Failed
        </Badge>
      );
    case "skipped":
      return (
        <Badge variant="secondary" className="text-xs">
          Skipped
        </Badge>
      );
  }
}

function SourceItem({ source }: { source: WebSearchSource }) {
  const [expanded, setExpanded] = useState(false);
  const hasDetails = source.error || source.snippet;

  return (
    <div className="border-border mb-1.5 rounded-md border">
      <button
        onClick={() => hasDetails && setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left transition-colors",
          hasDetails && "hover:bg-muted/50 cursor-pointer"
        )}
        disabled={!hasDetails}
      >
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <ExternalLink className="text-muted-foreground h-3.5 w-3.5 shrink-0" />
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary block min-w-0 flex-1 truncate text-sm hover:underline"
            onClick={(e) => e.stopPropagation()}
          >
            {source.title}
          </a>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {getStatusBadge(source.status)}
          {hasDetails &&
            (expanded ? (
              <ChevronUp className="text-muted-foreground h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="text-muted-foreground h-3.5 w-3.5" />
            ))}
        </div>
      </button>
      {hasDetails && (
        <div
          className={cn(
            "overflow-hidden transition-all duration-200",
            expanded ? "max-h-48" : "max-h-0"
          )}
        >
          <div className="border-border border-t px-2.5 py-2">
            {source.error && <p className="text-destructive text-xs">{source.error}</p>}
            {source.snippet && !source.error && (
              <p className="text-muted-foreground text-xs leading-relaxed">
                {source.snippet}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function WebSearchSources({ sources }: WebSearchSourcesProps) {
  const [collapsed, setCollapsed] = useState(true);

  if (sources.length === 0) return null;

  // Calculate counts
  const fetchedCount = sources.filter((s) => s.status === "success").length;
  const failedCount = sources.filter((s) => s.status === "failed").length;
  const skippedCount = sources.filter((s) => s.status === "skipped").length;

  // Build summary text
  const summaryParts: string[] = [];
  if (fetchedCount > 0) summaryParts.push(`${fetchedCount} fetched`);
  if (failedCount > 0) summaryParts.push(`${failedCount} failed`);
  if (skippedCount > 0) summaryParts.push(`${skippedCount} skipped`);

  return (
    <div className="mt-2 border-t pt-2">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="text-muted-foreground hover:text-foreground mb-1.5 flex w-full items-center justify-between gap-2 text-xs transition-colors"
      >
        <div className="flex items-center gap-2">
          <Globe className="h-3.5 w-3.5" />
          <span className="font-medium tracking-wide uppercase">
            Web Sources ({sources.length})
          </span>
          {summaryParts.length > 0 && (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | {summaryParts.join(", ")}
            </span>
          )}
        </div>
        {collapsed ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <ChevronUp className="h-3.5 w-3.5 shrink-0" />
        )}
      </button>
      <div
        className={cn(
          "overflow-hidden transition-all duration-200",
          collapsed ? "max-h-0" : "max-h-[1000px]"
        )}
      >
        {sources.map((source, index) => (
          <SourceItem key={`${source.url}-${index}`} source={source} />
        ))}
      </div>
    </div>
  );
}
