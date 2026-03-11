import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface IndexMetadataViewProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  moduleName: string;
  metadata: Record<string, unknown> | null;
}

const LABEL_MAP: Record<string, string> = {
  num_embeddings: "Embeddings",
  embedding_dimension: "Embedding Dimension",
  num_documents: "Document Chunks",
  num_source_docs: "Source Documents",
  embedding_model_id: "Embedding Model",
  chunking_strategy: "Chunking Strategy",
  chunk_sizes: "Chunk Sizes",
  chunk_overlap: "Chunk Overlap",
  build_date: "Build Date",
};

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

export function IndexMetadataView({
  open,
  onOpenChange,
  moduleName,
  metadata,
}: IndexMetadataViewProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Index Metadata: {moduleName}</DialogTitle>
        </DialogHeader>
        {metadata && Object.keys(metadata).length > 0 ? (
          <div className="space-y-2 text-sm">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key} className="flex gap-2">
                <span className="text-muted-foreground min-w-[160px] text-xs font-medium">
                  {LABEL_MAP[key] || key}:
                </span>
                <span className="text-xs break-all">{formatValue(value)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">No metadata available.</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
