import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  FileText,
  BookOpen,
  Paperclip,
  Book,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Badge } from "@/components/ui/badge";
import { cn, convertLatexDelimiters } from "@/lib/utils";
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

  // Extract metadata
  const filename =
    (source.metadata?.filename as string) ||
    (source.metadata?.file_name as string) ||
    `Source ${index + 1}`;
  const displayName = (source.metadata?.display_name as string) || filename;
  const sourceUrl = source.metadata?.source_url as string | undefined;
  const authors = source.metadata?.authors as string | undefined;
  const docType = source.metadata?.doc_type as string | undefined;
  const pageNumber = source.metadata?.page as number | undefined;

  const renderIcon = () => {
    const iconClassName = "text-muted-foreground h-3.5 w-3.5 shrink-0";
    switch (docType) {
      case "paper":
        return <FileText className={iconClassName} />;
      case "library_doc":
        return <BookOpen className={iconClassName} />;
      case "uploaded_pdf":
        return <Paperclip className={iconClassName} />;
      case "book":
        return <Book className={iconClassName} />;
      default:
        return <FileText className={iconClassName} />;
    }
  };

  return (
    <div className="border-border mb-1.5 rounded-md border">
      <button
        onClick={() => setExpanded(!expanded)}
        className="hover:bg-muted/50 flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left transition-colors"
      >
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {renderIcon()}
          <div className="min-w-0 flex-1">
            {sourceUrl ? (
              <a
                href={sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary block truncate text-sm hover:underline"
                onClick={(e) => e.stopPropagation()}
              >
                {displayName}
              </a>
            ) : (
              <span className="text-foreground block truncate text-sm">
                {displayName}
              </span>
            )}
            {authors && (
              <span className="text-muted-foreground block truncate text-xs">
                {authors}
              </span>
            )}
          </div>
          {pageNumber !== undefined && (
            <span className="text-muted-foreground shrink-0 text-xs">
              p. {pageNumber}
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant={variant} className="text-xs">
            {label}
          </Badge>
          {expanded ? (
            <ChevronUp className="text-muted-foreground h-3.5 w-3.5" />
          ) : (
            <ChevronDown className="text-muted-foreground h-3.5 w-3.5" />
          )}
        </div>
      </button>
      <div
        className={cn(
          "overflow-hidden transition-all duration-200",
          expanded ? "max-h-96" : "max-h-0"
        )}
      >
        <div className="border-border border-t px-2.5 py-2">
          <div className="chat-markdown text-muted-foreground max-w-none text-xs leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeHighlight, rehypeKatex]}
            >
              {convertLatexDelimiters(source.text)}
            </ReactMarkdown>
          </div>
          {source.score !== null && source.score !== undefined && (
            <p className="text-muted-foreground mt-1.5 text-xs">
              Relevance: {(source.score * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

interface SourcesListProps {
  sources: SourceNode[];
}

export function SourcesList({ sources }: SourcesListProps) {
  const [collapsed, setCollapsed] = useState(true);

  if (sources.length === 0) return null;

  // Calculate confidence statistics
  const scores = sources
    .map((s) => s.score)
    .filter((score): score is number => score !== null && score !== undefined);

  const stats =
    scores.length > 0
      ? {
          max: Math.max(...scores),
          min: Math.min(...scores),
          mean: scores.reduce((a, b) => a + b, 0) / scores.length,
        }
      : null;

  return (
    <div className="mt-2 border-t pt-2">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="text-muted-foreground hover:text-foreground mb-1.5 flex w-full items-center justify-between gap-2 text-xs transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium tracking-wide uppercase">
            Sources ({sources.length})
          </span>
          {stats && (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | Max: {(stats.max * 100).toFixed(0)}% | Min: {(stats.min * 100).toFixed(0)}
              % | Avg: {(stats.mean * 100).toFixed(0)}%
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
          collapsed ? "max-h-0" : "max-h-[2000px]"
        )}
      >
        {sources.map((source, index) => (
          <SourceCard key={index} source={source} index={index} />
        ))}
      </div>
    </div>
  );
}
