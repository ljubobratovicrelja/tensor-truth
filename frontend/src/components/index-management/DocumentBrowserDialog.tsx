import { useState, useMemo } from "react";
import { ArrowLeft, ExternalLink, FileText, Search } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useSourceDocuments, useDocumentContent } from "@/hooks";
import type { SourceDocumentInfo } from "@/api/types";

interface DocumentBrowserDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  embeddingModelId: string;
  moduleName: string;
  moduleDisplayName: string;
}

export function DocumentBrowserDialog({
  open,
  onOpenChange,
  embeddingModelId,
  moduleName,
  moduleDisplayName,
}: DocumentBrowserDialogProps) {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const { data: docsData, isLoading: docsLoading } = useSourceDocuments(
    embeddingModelId,
    moduleName,
    open
  );

  const filteredDocs = useMemo(() => {
    if (!docsData?.documents) return [];
    if (!filter) return docsData.documents;
    const lower = filter.toLowerCase();
    return docsData.documents.filter(
      (d) =>
        d.file_name.toLowerCase().includes(lower) ||
        (d.title && d.title.toLowerCase().includes(lower))
    );
  }, [docsData, filter]);

  const handleBack = () => {
    setSelectedDocId(null);
  };

  const handleClose = (value: boolean) => {
    if (!value) {
      setSelectedDocId(null);
      setFilter("");
    }
    onOpenChange(value);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="flex h-[100dvh] w-full flex-col md:h-[85vh] md:w-[80vw] md:max-w-6xl">
        <DialogHeader>
          <div className="flex items-center gap-2 pr-6">
            {selectedDocId && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={handleBack}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
            <DialogTitle className="truncate">
              {selectedDocId ? "Document Viewer" : `Documents — ${moduleDisplayName}`}
            </DialogTitle>
            {docsData && !selectedDocId && (
              <Badge variant="secondary" className="shrink-0">
                {filteredDocs.length}
                {filteredDocs.length !== docsData.documents.length &&
                  ` / ${docsData.documents.length}`}
              </Badge>
            )}
          </div>
        </DialogHeader>

        {selectedDocId ? (
          <DocumentViewer
            embeddingModelId={embeddingModelId}
            moduleName={moduleName}
            docId={selectedDocId}
          />
        ) : (
          <DocumentList
            documents={filteredDocs}
            loading={docsLoading}
            filter={filter}
            onFilterChange={setFilter}
            onSelect={setSelectedDocId}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

// ── Document List ────────────────────────────────────────────────────

function DocumentList({
  documents,
  loading,
  filter,
  onFilterChange,
  onSelect,
}: {
  documents: SourceDocumentInfo[];
  loading: boolean;
  filter: string;
  onFilterChange: (v: string) => void;
  onSelect: (docId: string) => void;
}) {
  return (
    <>
      <div className="relative">
        <Search className="text-muted-foreground absolute top-2.5 left-2.5 h-4 w-4" />
        <Input
          placeholder="Filter documents..."
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          className="pl-9"
        />
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-1 p-1">
          {loading &&
            Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full rounded-lg" />
            ))}
          {!loading && documents.length === 0 && (
            <p className="text-muted-foreground py-8 text-center text-sm">
              No documents found.
            </p>
          )}
          {!loading &&
            documents.map((doc) => (
              <button
                key={doc.doc_id}
                onClick={() => onSelect(doc.doc_id)}
                className="hover:bg-muted/80 flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors"
              >
                <FileText className="text-muted-foreground h-4 w-4 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{doc.file_name}</p>
                  {doc.title && doc.title !== doc.file_name && (
                    <p className="text-muted-foreground truncate text-xs">{doc.title}</p>
                  )}
                </div>
                <div className="text-muted-foreground flex shrink-0 items-center gap-3 text-xs">
                  {doc.file_size != null && <span>{formatFileSize(doc.file_size)}</span>}
                  <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                    {doc.chunk_count} chunks
                  </Badge>
                </div>
              </button>
            ))}
        </div>
      </ScrollArea>
    </>
  );
}

// ── Document Viewer ──────────────────────────────────────────────────

function DocumentViewer({
  embeddingModelId,
  moduleName,
  docId,
}: {
  embeddingModelId: string;
  moduleName: string;
  docId: string;
}) {
  const { data, isLoading } = useDocumentContent(
    embeddingModelId,
    moduleName,
    docId,
    true
  );

  if (isLoading) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-1">
        <Skeleton className="h-6 w-1/3" />
        <Skeleton className="min-h-0 flex-1" />
      </div>
    );
  }

  if (!data) {
    return (
      <p className="text-muted-foreground py-8 text-center text-sm">
        Document not found.
      </p>
    );
  }

  const meta = data.metadata as Record<string, string | undefined>;
  const fileName = meta.file_name || "Unknown";
  const title = meta.display_name || meta.title;
  const sourceUrl = meta.source_url;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{fileName}</p>
          {title && title !== fileName && (
            <p className="text-muted-foreground truncate text-xs">{title}</p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant="secondary" className="text-[10px]">
            {data.chunks.length} chunks
            {data.chunks.length > 0 && (
              <> &middot; {Math.max(...data.chunks.map((c) => c.level)) + 1} levels</>
            )}
          </Badge>
          {sourceUrl && (
            <a
              href={sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
        </div>
      </div>

      {/* Legend */}
      {data.chunks.length > 0 && (
        <ChunkLegend maxLevel={Math.max(...data.chunks.map((c) => c.level))} />
      )}

      {/* Text with chunk highlighting */}
      <ScrollArea className="min-h-0 flex-1 rounded-md border">
        <pre className="p-4 text-sm whitespace-pre-wrap">
          <ChunkedText text={data.text} chunks={data.chunks} />
        </pre>
      </ScrollArea>
    </div>
  );
}

// ── Hierarchical Chunk Renderer ──────────────────────────────────────

// Each level gets a pair of alternating background colors + a label color for the legend
const LEVEL_STYLES: {
  colors: [string, string];
  swatches: [string, string];
}[] = [
  {
    colors: [
      "bg-gray-200/70 dark:bg-gray-700/50",
      "bg-stone-200/70 dark:bg-stone-700/50",
    ],
    swatches: ["bg-gray-300 dark:bg-gray-600", "bg-stone-300 dark:bg-stone-600"],
  },
  {
    colors: [
      "bg-blue-200/60 dark:bg-blue-800/50",
      "bg-amber-200/60 dark:bg-amber-800/50",
    ],
    swatches: ["bg-blue-300 dark:bg-blue-700", "bg-amber-300 dark:bg-amber-700"],
  },
  {
    colors: [
      "bg-emerald-200/60 dark:bg-emerald-800/50",
      "bg-rose-200/60 dark:bg-rose-800/50",
    ],
    swatches: ["bg-emerald-300 dark:bg-emerald-700", "bg-rose-300 dark:bg-rose-700"],
  },
  {
    colors: [
      "bg-violet-200/60 dark:bg-violet-800/50",
      "bg-orange-200/60 dark:bg-orange-800/50",
    ],
    swatches: ["bg-violet-300 dark:bg-violet-700", "bg-orange-300 dark:bg-orange-700"],
  },
  {
    colors: ["bg-sky-200/60 dark:bg-sky-800/50", "bg-pink-200/60 dark:bg-pink-800/50"],
    swatches: ["bg-sky-300 dark:bg-sky-700", "bg-pink-300 dark:bg-pink-700"],
  },
];

// Convenience accessor for the color pairs
const LEVEL_COLORS: [string, string][] = LEVEL_STYLES.map((s) => s.colors);

function ChunkLegend({ maxLevel }: { maxLevel: number }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px]">
      {maxLevel > 0 && (
        <div className="flex items-center gap-1">
          <span className="border-border/60 inline-block h-0 w-4 border-t-2 border-dashed" />
          <span className="text-muted-foreground">L0 boundary</span>
        </div>
      )}
      {Array.from({ length: maxLevel + 1 }, (_, level) => {
        const style = LEVEL_STYLES[Math.min(level, LEVEL_STYLES.length - 1)];
        const label = level === 0 ? "L0 (top)" : `L${level}`;
        return (
          <div key={level} className="flex items-center gap-1">
            <span className={`inline-block h-3 w-2 rounded-l-sm ${style.swatches[0]}`} />
            <span
              className={`-ml-px inline-block h-3 w-2 rounded-r-sm ${style.swatches[1]}`}
            />
            <span className="text-muted-foreground">{label}</span>
          </div>
        );
      })}
    </div>
  );
}

interface ChunkWithLevel {
  chunk_id: string;
  start: number;
  end: number;
  level: number;
}

function ChunkedText({ text, chunks }: { text: string; chunks: ChunkWithLevel[] }) {
  if (chunks.length === 0) return <>{text}</>;

  // Find leaf chunks: those with no deeper chunk fully contained within them
  const leaves = chunks.filter(
    (c) => !chunks.some((d) => d.level > c.level && d.start >= c.start && d.end <= c.end)
  );
  leaves.sort((a, b) => a.start - b.start);

  // L0 boundary positions (for separator lines between top-level chunks)
  const l0Chunks = chunks.filter((c) => c.level === 0).sort((a, b) => a.start - b.start);
  const l0Starts = new Set(l0Chunks.slice(1).map((c) => c.start));

  // Track alternation counter per level — resets at L0 boundaries
  const altCounters = new Map<number, number>();

  const elements: React.ReactNode[] = [];
  let pos = 0;

  for (let i = 0; i < leaves.length; i++) {
    const leaf = leaves[i];
    const effectiveStart = Math.max(leaf.start, pos);

    // Skip fully overlapped leaves
    if (effectiveStart >= leaf.end) continue;

    // Check for L0 separator between pos and this leaf
    for (const l0Start of l0Starts) {
      if (l0Start > pos && l0Start <= effectiveStart) {
        elements.push(
          <span
            key={`sep-${l0Start}`}
            className="border-border/60 my-1 block border-t-2 border-dashed"
          />
        );
        // Reset all alternation counters at L0 boundary
        altCounters.clear();
        l0Starts.delete(l0Start);
        break;
      }
    }

    // Gap before this leaf (uncovered text)
    if (effectiveStart > pos) {
      elements.push(<span key={`gap-${i}`}>{text.slice(pos, effectiveStart)}</span>);
    }

    // Get alternating color for this leaf's level
    const level = leaf.level;
    const altIdx = altCounters.get(level) ?? 0;
    altCounters.set(level, altIdx + 1);

    const colors = LEVEL_COLORS[Math.min(level, LEVEL_COLORS.length - 1)];
    const colorClass = colors[altIdx % 2];

    elements.push(
      <span key={leaf.chunk_id} className={colorClass} title={chunkTooltip(leaf)}>
        {text.slice(effectiveStart, leaf.end)}
      </span>
    );

    pos = leaf.end;
  }

  // Trailing text
  if (pos < text.length) {
    elements.push(<span key="tail">{text.slice(pos)}</span>);
  }

  return <>{elements}</>;
}

function chunkTooltip(chunk: ChunkWithLevel): string {
  const truncId =
    chunk.chunk_id.length > 12 ? chunk.chunk_id.slice(0, 8) + "..." : chunk.chunk_id;
  return `L${chunk.level} chunk ${truncId} [${chunk.start}–${chunk.end}]`;
}

// ── Helpers ──────────────────────────────────────────────────────────

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
