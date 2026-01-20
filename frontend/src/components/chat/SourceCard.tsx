import { useState } from "react";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { SourceNode } from "@/api/types";

interface SourceCardProps {
  source: SourceNode;
  index: number;
}

function getConfidenceBadgeVariant(score: number | null | undefined): {
  variant: "default" | "secondary" | "destructive" | "outline";
  label: string;
} {
  if (score === null || score === undefined) {
    return { variant: "outline", label: "Unknown" };
  }
  if (score >= 0.7) {
    return { variant: "default", label: "High" };
  }
  if (score >= 0.4) {
    return { variant: "secondary", label: "Medium" };
  }
  return { variant: "destructive", label: "Low" };
}

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { variant, label } = getConfidenceBadgeVariant(source.score);

  const filename =
    (source.metadata?.filename as string) ||
    (source.metadata?.file_name as string) ||
    `Source ${index + 1}`;

  const pageNumber = source.metadata?.page as number | undefined;

  return (
    <Card className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between p-3 text-left"
      >
        <div className="flex items-center gap-2">
          <FileText className="text-muted-foreground h-4 w-4" />
          <span className="text-sm font-medium">{filename}</span>
          {pageNumber !== undefined && (
            <span className="text-muted-foreground text-xs">p. {pageNumber}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={variant}>{label}</Badge>
          {expanded ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </div>
      </button>
      <div
        className={cn(
          "overflow-hidden transition-all duration-200",
          expanded ? "max-h-96" : "max-h-0"
        )}
      >
        <CardContent className="border-t pt-3">
          <p className="text-muted-foreground text-sm whitespace-pre-wrap">
            {source.text}
          </p>
          {source.score !== null && source.score !== undefined && (
            <p className="text-muted-foreground mt-2 text-xs">
              Relevance score: {(source.score * 100).toFixed(1)}%
            </p>
          )}
        </CardContent>
      </div>
    </Card>
  );
}

interface SourcesListProps {
  sources: SourceNode[];
}

export function SourcesList({ sources }: SourcesListProps) {
  const [showAll, setShowAll] = useState(false);

  if (sources.length === 0) return null;

  const displayedSources = showAll ? sources : sources.slice(0, 3);
  const hasMore = sources.length > 3;

  return (
    <div className="mt-3 border-t pt-3">
      <p className="text-muted-foreground mb-2 text-xs font-medium uppercase">
        Sources ({sources.length})
      </p>
      {displayedSources.map((source, index) => (
        <SourceCard key={index} source={source} index={index} />
      ))}
      {hasMore && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-primary mt-1 text-sm hover:underline"
        >
          {showAll ? "Show less" : `Show ${sources.length - 3} more sources`}
        </button>
      )}
    </div>
  );
}
