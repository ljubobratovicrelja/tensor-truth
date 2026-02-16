import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  FileText,
  BookOpen,
  Paperclip,
  Book,
  HelpCircle,
  AlertTriangle,
  Globe,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn, convertLatexDelimiters } from "@/lib/utils";
import type { RetrievalMetrics, SourceNode } from "@/api/types";

// ============================================================================
// Metric Tooltip Helpers
// ============================================================================

interface MetricTooltip {
  title: string;
  description: string;
  interpretation: string;
}

function getMedianTooltip(median: number | null): MetricTooltip {
  const percentage = median !== null ? median * 100 : null;
  const base = {
    title: "Median Relevance Score",
    description:
      "The middle value of all source relevance scores. Half the sources score above this, half below.",
  };

  if (percentage === null) {
    return { ...base, interpretation: "No score data available." };
  }
  if (percentage >= 70) {
    return {
      ...base,
      interpretation:
        "Excellent match. The retrieval system found highly relevant sources for your query.",
    };
  }
  if (percentage >= 50) {
    return {
      ...base,
      interpretation:
        "Good match. Sources are reasonably relevant and should provide useful context.",
    };
  }
  if (percentage >= 30) {
    return {
      ...base,
      interpretation:
        "Moderate match. The sources have mixed relevance—consider refining your query for better results.",
    };
  }
  return {
    ...base,
    interpretation:
      "Weak match. Sources may only be tangentially related. Try rephrasing your question or selecting different modules.",
  };
}

function getIQRTooltip(iqr: number | null): MetricTooltip {
  const percentage = iqr !== null ? iqr * 100 : null;
  const base = {
    title: "Interquartile Range (IQR)",
    description:
      "Measures the spread between the 25th and 75th percentile scores. Lower values indicate more consistent source quality.",
  };

  if (percentage === null) {
    return { ...base, interpretation: "No score data available." };
  }
  if (percentage <= 10) {
    return {
      ...base,
      interpretation:
        "Very consistent. All retrieved sources have similar relevance levels—uniform quality throughout.",
    };
  }
  if (percentage <= 25) {
    return {
      ...base,
      interpretation:
        "Fairly consistent. Source scores are reasonably uniform with minor variation.",
    };
  }
  if (percentage <= 40) {
    return {
      ...base,
      interpretation:
        "Variable quality. Source relevance varies considerably—top sources are much better than lower ones.",
    };
  }
  return {
    ...base,
    interpretation:
      "Highly variable. Large spread in source quality—rely more heavily on the top-ranked sources.",
  };
}

function getHighConfidenceTooltip(ratio: number): MetricTooltip {
  const percentage = ratio * 100;
  const base = {
    title: "High Confidence Ratio",
    description:
      "Percentage of sources scoring ≥70%. These are strong matches with high relevance to your query.",
  };

  if (percentage >= 80) {
    return {
      ...base,
      interpretation:
        "Outstanding. The vast majority of sources are highly relevant to your query.",
    };
  }
  if (percentage >= 50) {
    return {
      ...base,
      interpretation:
        "Good quality pool. Most sources are confident matches that should inform the response well.",
    };
  }
  if (percentage >= 20) {
    return {
      ...base,
      interpretation:
        "Mixed results. Some strong sources found, but many are moderate matches.",
    };
  }
  return {
    ...base,
    interpretation:
      "Few top-tier matches. Most sources fall into moderate or low confidence ranges.",
  };
}

function getMediumConfidenceTooltip(ratio: number): MetricTooltip {
  const percentage = ratio * 100;
  const base = {
    title: "Medium Confidence Ratio",
    description:
      "Percentage of sources scoring between 40-70%. These are reasonable matches with moderate relevance.",
  };

  if (percentage >= 80) {
    return {
      ...base,
      interpretation:
        "Most sources are moderately relevant. Decent quality, but few standout matches.",
    };
  }
  if (percentage >= 50) {
    return {
      ...base,
      interpretation:
        "Balanced distribution. A healthy mix of confidence levels across your sources.",
    };
  }
  if (percentage >= 20) {
    return {
      ...base,
      interpretation:
        "Sources are polarized—mostly either high or low confidence with few in the middle.",
    };
  }
  return {
    ...base,
    interpretation:
      "Very few moderate matches. Sources tend strongly toward high or low confidence.",
  };
}

function getLowConfidenceTooltip(ratio: number): MetricTooltip {
  const percentage = ratio * 100;
  const base = {
    title: "Low Confidence Ratio",
    description:
      "Percentage of sources scoring <40%. These are weak matches that may only be tangentially related.",
  };

  if (percentage <= 10) {
    return {
      ...base,
      interpretation:
        "Excellent filtering. Very few weak sources made it through—strong overall retrieval quality.",
    };
  }
  if (percentage <= 30) {
    return {
      ...base,
      interpretation:
        "Good filtering. Some weaker sources included for coverage, but they won't dominate the context.",
    };
  }
  if (percentage <= 50) {
    return {
      ...base,
      interpretation:
        "Moderate noise. A significant portion of sources have low confidence—results may be diluted.",
    };
  }
  return {
    ...base,
    interpretation:
      "High noise level. Many sources have low relevance—consider narrowing your query scope.",
  };
}

function getSourceTypesTooltip(types: number): MetricTooltip {
  const base = {
    title: "Source Types",
    description:
      "Number of distinct document categories (papers, books, library docs, etc.) represented in the results.",
  };

  if (types === 1) {
    return {
      ...base,
      interpretation:
        "Single source type. Focused but narrow—all context comes from one document category.",
    };
  }
  if (types <= 3) {
    return {
      ...base,
      interpretation:
        "Moderate diversity. Multiple document types provide different perspectives on your query.",
    };
  }
  return {
    ...base,
    interpretation:
      "High diversity. Broad coverage across many document types—comprehensive multi-perspective context.",
  };
}

function getEntropyTooltip(entropy: number | null): MetricTooltip {
  const base = {
    title: "Source Entropy",
    description:
      "Information-theoretic measure of how evenly distributed the sources are. Higher values mean more balanced representation.",
  };

  if (entropy === null) {
    return { ...base, interpretation: "No entropy data available." };
  }
  if (entropy < 1) {
    return {
      ...base,
      interpretation:
        "Concentrated sources. Results heavily favor a few documents—deep but narrow coverage.",
    };
  }
  if (entropy < 2) {
    return {
      ...base,
      interpretation:
        "Moderate distribution. Sources are reasonably spread across documents with some clustering.",
    };
  }
  return {
    ...base,
    interpretation:
      "Well-distributed. Sources are evenly spread across many documents—broad, balanced coverage.",
  };
}

function getChunksTooltip(chunks: number): MetricTooltip {
  const base = {
    title: "Total Chunks",
    description:
      "Number of text passages retrieved to form the context. More chunks provide broader coverage but increase response time.",
  };

  if (chunks <= 5) {
    return {
      ...base,
      interpretation:
        "Minimal context. Few passages retrieved—responses will be concise but may miss relevant details.",
    };
  }
  if (chunks <= 15) {
    return {
      ...base,
      interpretation:
        "Moderate context. Good balance between coverage and efficiency for most queries.",
    };
  }
  if (chunks <= 30) {
    return {
      ...base,
      interpretation:
        "Rich context. Extensive coverage that should capture nuanced or complex topics well.",
    };
  }
  return {
    ...base,
    interpretation:
      "Very large context. Maximum coverage—ideal for comprehensive questions but may slow responses.",
  };
}

function getCharsPerChunkTooltip(avgChars: number): MetricTooltip {
  const base = {
    title: "Average Chunk Size",
    description:
      "Mean character count per retrieved passage. Affects how much context each chunk provides.",
  };

  if (avgChars < 200) {
    return {
      ...base,
      interpretation:
        "Very short chunks. Highly focused snippets—precise but may lack surrounding context.",
    };
  }
  if (avgChars < 500) {
    return {
      ...base,
      interpretation:
        "Short chunks. Good for precise retrieval where specific sentences matter most.",
    };
  }
  if (avgChars < 1000) {
    return {
      ...base,
      interpretation:
        "Medium chunks. Balanced size that preserves paragraph-level context.",
    };
  }
  return {
    ...base,
    interpretation:
      "Large chunks. Substantial passages that provide rich surrounding context per source.",
  };
}

// ============================================================================
// MetricItem Component
// ============================================================================

interface MetricItemProps {
  label: string;
  value: string;
  tooltip: MetricTooltip;
}

function MetricItem({ label, value, tooltip }: MetricItemProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="hover:bg-muted/50 -ml-1 flex w-fit cursor-help items-center gap-1 rounded px-1 py-0.5 transition-colors">
          <span>{label ? `${label}: ${value}` : value}</span>
          <HelpCircle className="text-muted-foreground/50 h-3 w-3" />
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        <div className="space-y-1.5">
          <div className="font-medium">{tooltip.title}</div>
          <div className="text-muted-foreground text-xs">{tooltip.description}</div>
          <div className="border-border border-t pt-1.5 text-xs">
            {tooltip.interpretation}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

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

  // Extract metadata
  const rawFilename =
    (source.metadata?.filename as string) ||
    (source.metadata?.file_name as string) ||
    `Source ${index + 1}`;
  // Strip internal doc ID prefix (e.g. "pdf_544414ce_elms-eye-tracking" → "elms-eye-tracking")
  const filename = rawFilename.replace(/^(pdf|doc|url)_[a-f0-9]{7,8}_/, "");
  const displayName = (source.metadata?.display_name as string) || filename;
  const sourceUrl = source.metadata?.source_url as string | undefined;
  const authors = source.metadata?.authors as string | undefined;
  const docType = source.metadata?.doc_type as string | undefined;
  const pageNumber = source.metadata?.page as number | undefined;
  const fetchStatus = source.metadata?.fetch_status as string | undefined;
  const fetchError = source.metadata?.fetch_error as string | undefined;

  // Determine badge based on fetch_status (for web sources) or score
  const getBadgeInfo = (): {
    variant: "default" | "secondary" | "destructive" | "outline";
    label: string;
  } => {
    // For web sources, use fetch_status for failures, confidence for success
    if (docType === "web" && fetchStatus) {
      if (fetchStatus === "failed") {
        return { variant: "destructive", label: "Failed" };
      }
      if (fetchStatus === "skipped") {
        return { variant: "secondary", label: "Skipped" };
      }
      if (fetchStatus === "success") {
        // For successful web fetches with relevance scores, show confidence level
        if (source.score !== null && source.score !== undefined) {
          return getConfidenceBadgeVariant(source.score);
        }
        return { variant: "default", label: "Fetched" };
      }
    }
    // Fall back to confidence-based badge
    return getConfidenceBadgeVariant(source.score);
  };

  const { variant, label } = getBadgeInfo();

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
      case "web":
        return <Globe className={iconClassName} />;
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
          {/* Show fetch error/skip reason for failed/skipped web sources */}
          {docType === "web" && fetchError && (
            <p
              className={cn(
                "mb-2 text-xs",
                fetchStatus === "failed"
                  ? "text-destructive"
                  : "text-muted-foreground italic"
              )}
            >
              {fetchError}
            </p>
          )}
          {/* Show content if available */}
          {source.text && (
            <div className="chat-markdown text-muted-foreground max-h-96 max-w-none overflow-y-auto text-xs leading-relaxed">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                  rehypeHighlight,
                  [rehypeKatex, { throwOnError: false, strict: false }],
                ]}
              >
                {convertLatexDelimiters(source.text)}
              </ReactMarkdown>
            </div>
          )}
          {/* Show relevance score when available */}
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

function MetricsPanel({ metrics }: { metrics: RetrievalMetrics }) {
  const { score_distribution, diversity, coverage, quality } = metrics;

  return (
    <div className="bg-muted/30 mb-3 rounded-md p-3">
      <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
        {/* Score Distribution */}
        <div>
          <div className="mb-1 font-medium">Distribution</div>
          <div className="text-muted-foreground space-y-0.5">
            {score_distribution.median !== null && (
              <MetricItem
                label="Median"
                value={`${(score_distribution.median * 100).toFixed(1)}%`}
                tooltip={getMedianTooltip(score_distribution.median)}
              />
            )}
            {score_distribution.iqr !== null && (
              <MetricItem
                label="IQR"
                value={`${(score_distribution.iqr * 100).toFixed(1)}%`}
                tooltip={getIQRTooltip(score_distribution.iqr)}
              />
            )}
          </div>
        </div>

        {/* Quality */}
        <div>
          <div className="mb-1 font-medium">Quality</div>
          <div className="text-muted-foreground space-y-0.5">
            {(() => {
              const high = quality.high_confidence_ratio;
              const low = quality.low_confidence_ratio;
              const medium = Math.max(0, 1 - high - low);
              return (
                <>
                  <MetricItem
                    label="High"
                    value={`${(high * 100).toFixed(0)}%`}
                    tooltip={getHighConfidenceTooltip(high)}
                  />
                  <MetricItem
                    label="Med"
                    value={`${(medium * 100).toFixed(0)}%`}
                    tooltip={getMediumConfidenceTooltip(medium)}
                  />
                  <MetricItem
                    label="Low"
                    value={`${(low * 100).toFixed(0)}%`}
                    tooltip={getLowConfidenceTooltip(low)}
                  />
                </>
              );
            })()}
          </div>
        </div>

        {/* Diversity */}
        <div>
          <div className="mb-1 font-medium">Diversity</div>
          <div className="text-muted-foreground space-y-0.5">
            <MetricItem
              label=""
              value={`${diversity.source_types} types`}
              tooltip={getSourceTypesTooltip(diversity.source_types)}
            />
            {diversity.source_entropy !== null && (
              <MetricItem
                label="Entropy"
                value={diversity.source_entropy.toFixed(2)}
                tooltip={getEntropyTooltip(diversity.source_entropy)}
              />
            )}
          </div>
        </div>

        {/* Coverage */}
        <div>
          <div className="mb-1 font-medium">Coverage</div>
          <div className="text-muted-foreground space-y-0.5">
            <MetricItem
              label=""
              value={`${coverage.total_chunks} chunks`}
              tooltip={getChunksTooltip(coverage.total_chunks)}
            />
            <MetricItem
              label=""
              value={`${coverage.avg_chunk_length.toFixed(0)} chars/chunk`}
              tooltip={getCharsPerChunkTooltip(coverage.avg_chunk_length)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Helper: compute web search stats for a set of web sources
// ============================================================================

function computeWebStats(webSources: SourceNode[]) {
  return {
    fetched: webSources.filter((s) => s.metadata?.fetch_status === "success").length,
    failed: webSources.filter(
      (s) =>
        s.metadata?.fetch_status === "failed" || s.metadata?.fetch_status === "skipped"
    ).length,
    totalChars: webSources.reduce(
      (sum, s) => sum + ((s.metadata?.content_chars as number) || 0),
      0
    ),
  };
}

// ============================================================================
// Helper: compute score stats for a set of sources
// ============================================================================

function computeScoreStats(sources: SourceNode[]) {
  const scores = sources
    .map((s) => s.score)
    .filter((score): score is number => score !== null && score !== undefined);

  if (scores.length === 0) return null;
  return {
    max: Math.max(...scores),
    min: Math.min(...scores),
    mean: scores.reduce((a, b) => a + b, 0) / scores.length,
  };
}

// ============================================================================
// Helper: sort web sources by score descending
// ============================================================================

function sortByScoreDesc(sources: SourceNode[]) {
  return [...sources].sort((a, b) => {
    const scoreA = a.score ?? -1;
    const scoreB = b.score ?? -1;
    return scoreB - scoreA;
  });
}

// ============================================================================
// SourceSection: renders a group of source cards with an optional header
// ============================================================================

interface SourceSectionProps {
  icon: React.ReactNode;
  title: string;
  sources: SourceNode[];
  metrics?: RetrievalMetrics | null;
  isWebSection?: boolean;
}

function SourceSection({
  icon,
  title,
  sources,
  metrics,
  isWebSection,
}: SourceSectionProps) {
  const sortedSources = isWebSection ? sortByScoreDesc(sources) : sources;
  const webStats = isWebSection ? computeWebStats(sources) : null;
  const webScoreStats = isWebSection
    ? computeScoreStats(sources.filter((s) => s.metadata?.fetch_status === "success"))
    : null;

  return (
    <div className="mb-3 last:mb-0">
      <div className="text-muted-foreground mb-1.5 flex items-center gap-1.5 text-xs">
        {icon}
        <span className="font-medium">
          {title} ({sources.length})
        </span>
        {webStats && (
          <span className="text-muted-foreground/70 font-normal">
            | {webStats.fetched} fetched
            {webStats.failed > 0 && <> | {webStats.failed} failed</>}
            {webStats.totalChars > 0 && (
              <> | ~{Math.round(webStats.totalChars / 4).toLocaleString()} tokens</>
            )}
            {webScoreStats && (
              <>
                {" "}
                | Relevance: {(webScoreStats.min * 100).toFixed(0)}%-
                {(webScoreStats.max * 100).toFixed(0)}%
              </>
            )}
          </span>
        )}
      </div>
      {metrics && <MetricsPanel metrics={metrics} />}
      {sortedSources.map((source, index) => (
        <SourceCard key={index} source={source} index={index} />
      ))}
    </div>
  );
}

// ============================================================================
// SourcesList Component
// ============================================================================

interface SourcesListProps {
  sources: SourceNode[];
  metrics?: RetrievalMetrics | null;
  confidenceLevel?: string;
}

export function SourcesList({ sources, metrics, confidenceLevel }: SourcesListProps) {
  const [collapsed, setCollapsed] = useState(true);

  // If no sources and no metrics, nothing to show (no RAG was attempted)
  if (sources.length === 0 && !metrics) return null;

  // RAG was active but no sources passed confidence thresholds
  if (sources.length === 0 && metrics) {
    return (
      <div className="mt-2 border-t pt-2">
        <div className="text-muted-foreground flex items-start gap-2 text-xs">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-yellow-500" />
          <div>
            <span className="font-medium">No sources met confidence thresholds.</span>{" "}
            <span className="text-muted-foreground/70">
              This response was generated without RAG context. Consider lowering the
              confidence threshold in session settings if you want to include
              lower-scoring sources.
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Partition sources into RAG and web groups
  const ragSources = sources.filter((s) => s.metadata?.doc_type !== "web");
  const webSources = sources.filter((s) => s.metadata?.doc_type === "web");
  const isMixed = ragSources.length > 0 && webSources.length > 0;

  // For single-type rendering, detect if all sources are web
  const isWebSearch = !isMixed && webSources.length > 0;

  // Sort web-only sources by score (for single-type web rendering)
  const sortedSources = isWebSearch ? sortByScoreDesc(sources) : sources;

  // Calculate stats for the header summary line
  const webStats = isWebSearch ? computeWebStats(sources) : null;

  // For score stats, use successfully fetched sources for web, all for RAG
  const sourcesForStats = isWebSearch
    ? sources.filter((s) => s.metadata?.fetch_status === "success")
    : sources;
  const stats = computeScoreStats(sourcesForStats);

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
          {confidenceLevel === "low" && (
            <Tooltip>
              <TooltipTrigger asChild>
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-yellow-500" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                Low confidence in retrieved sources — response may rely on general
                knowledge.
              </TooltipContent>
            </Tooltip>
          )}
          {isMixed ? (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | {ragSources.length} knowledge base, {webSources.length} web
            </span>
          ) : webStats ? (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | {webStats.fetched} fetched
              {webStats.failed > 0 && <> | {webStats.failed} failed</>}
              {webStats.totalChars > 0 && (
                <> | ~{Math.round(webStats.totalChars / 4).toLocaleString()} tokens</>
              )}
              {stats && (
                <>
                  {" "}
                  | Relevance: {(stats.min * 100).toFixed(0)}%-
                  {(stats.max * 100).toFixed(0)}%
                </>
              )}
            </span>
          ) : metrics ? (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | Avg: {((metrics.score_distribution.mean || 0) * 100).toFixed(0)}% | Range:{" "}
              {((metrics.score_distribution.min || 0) * 100).toFixed(0)}%-
              {((metrics.score_distribution.max || 0) * 100).toFixed(0)}%
              {metrics.score_distribution.std && (
                <> | σ: {(metrics.score_distribution.std * 100).toFixed(1)}%</>
              )}{" "}
              | {metrics.diversity.unique_sources} docs | ~
              {metrics.coverage.estimated_tokens} tokens
            </span>
          ) : stats ? (
            <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
              | Max: {(stats.max * 100).toFixed(0)}% | Min: {(stats.min * 100).toFixed(0)}
              % | Avg: {(stats.mean * 100).toFixed(0)}%
            </span>
          ) : null}
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
        {isMixed ? (
          <>
            <SourceSection
              icon={<BookOpen className="h-3.5 w-3.5" />}
              title="Knowledge Base Results"
              sources={ragSources}
              metrics={metrics}
            />
            <SourceSection
              icon={<Globe className="h-3.5 w-3.5" />}
              title="Web Results"
              sources={webSources}
              isWebSection
            />
          </>
        ) : (
          <>
            {/* Single-type: metrics panel for RAG only */}
            {metrics && !isWebSearch && <MetricsPanel metrics={metrics} />}

            {/* Source cards */}
            {sortedSources.map((source, index) => (
              <SourceCard key={index} source={source} index={index} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
