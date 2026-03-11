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

      {/* Text with chunk highlighting */}
      <ScrollArea className="min-h-0 flex-1 rounded-md border">
        <pre className="p-4 text-sm whitespace-pre-wrap">
          <ChunkedText text={data.text} chunks={data.chunks} />
        </pre>
      </ScrollArea>
    </div>
  );
}

// ── Chunked Text Renderer ────────────────────────────────────────────

const CHUNK_COLORS = [
  "bg-blue-100/40 dark:bg-blue-900/20",
  "bg-amber-100/40 dark:bg-amber-900/20",
] as const;

function ChunkedText({
  text,
  chunks,
}: {
  text: string;
  chunks: { chunk_id: string; start: number; end: number }[];
}) {
  if (chunks.length === 0) return <>{text}</>;

  const elements: React.ReactNode[] = [];
  let lastEnd = 0;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    // Gap between chunks (if any)
    if (chunk.start > lastEnd) {
      elements.push(<span key={`gap-${i}`}>{text.slice(lastEnd, chunk.start)}</span>);
    }

    const colorClass = CHUNK_COLORS[i % 2];
    const truncId =
      chunk.chunk_id.length > 12 ? chunk.chunk_id.slice(0, 8) + "..." : chunk.chunk_id;

    elements.push(
      <span
        key={chunk.chunk_id}
        className={colorClass}
        title={`Chunk ${truncId} [${chunk.start}–${chunk.end}]`}
      >
        {text.slice(chunk.start, chunk.end)}
      </span>
    );

    lastEnd = chunk.end;
  }

  // Trailing text
  if (lastEnd < text.length) {
    elements.push(<span key="tail">{text.slice(lastEnd)}</span>);
  }

  return <>{elements}</>;
}

// ── Helpers ──────────────────────────────────────────────────────────

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
