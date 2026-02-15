import { FileText, FileCode, Link, Trash2, RefreshCw, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useDocuments, useDeleteDocument, useReindexDocuments } from "@/hooks";
import type { ScopeType } from "@/api/types";
import { DocumentUploader } from "./DocumentUploader";

interface DocumentListProps {
  scopeId: string;
  scopeType: ScopeType;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function DocumentIcon({ filename }: { filename: string }) {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".md") || lower.endsWith(".markdown") || lower.endsWith(".txt")) {
    return <FileCode className="text-muted-foreground h-5 w-5 shrink-0" />;
  }
  if (lower.startsWith("url_")) {
    return <Link className="text-muted-foreground h-5 w-5 shrink-0" />;
  }
  return <FileText className="text-muted-foreground h-5 w-5 shrink-0" />;
}

export function DocumentList({ scopeId, scopeType }: DocumentListProps) {
  const { data, isLoading } = useDocuments(scopeId, scopeType);
  const deleteDoc = useDeleteDocument();
  const reindex = useReindexDocuments();

  const handleDelete = async (docId: string) => {
    try {
      await deleteDoc.mutateAsync({ scopeId, scopeType, docId });
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Delete failed";
      toast.error(`Failed to delete document: ${msg}`);
    }
  };

  const handleReindex = async () => {
    try {
      await reindex.mutateAsync({ scopeId, scopeType });
      toast.success("Documents reindexed");
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Reindex failed";
      toast.error(`Failed to reindex: ${msg}`);
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg">Documents</CardTitle>
        {data && data.documents.length > 0 && (
          <div className="flex items-center gap-2">
            {data.has_index ? (
              <Badge variant="secondary">Indexed</Badge>
            ) : (
              <Badge variant="outline">Not indexed</Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleReindex}
              disabled={reindex.isPending || data.documents.length === 0}
            >
              {reindex.isPending ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-1 h-4 w-4" />
              )}
              Reindex
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <DocumentUploader scopeId={scopeId} scopeType={scopeType} />

        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
          </div>
        ) : data?.documents.length === 0 ? (
          <p className="text-muted-foreground py-4 text-center text-sm">
            No documents uploaded yet
          </p>
        ) : (
          <div className="space-y-2">
            {data?.documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="flex items-center justify-between rounded-lg border p-3"
              >
                <div className="flex items-center gap-3 overflow-hidden">
                  <DocumentIcon filename={doc.filename} />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{doc.filename}</p>
                    <p className="text-muted-foreground text-xs">
                      {doc.page_count > 0 && <>{doc.page_count} pages &middot; </>}
                      {formatFileSize(doc.file_size)}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDelete(doc.doc_id)}
                  disabled={deleteDoc.isPending}
                  className="shrink-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
