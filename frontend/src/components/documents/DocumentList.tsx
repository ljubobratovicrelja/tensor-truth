import { useCallback } from "react";
import { FileText, FileCode, Link, Trash2, Loader2, Hammer, Check } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useDocuments, useDeleteDocument, useBuildIndex } from "@/hooks";
import { useTaskProgress } from "@/hooks/useTasks";
import type { ScopeType } from "@/api/types";
import { DocumentUploader } from "./DocumentUploader";
import { IndexingSettings } from "./IndexingSettings";

interface DocumentListProps {
  scopeId: string;
  scopeType: ScopeType;
  buildTaskId: string | null;
  onBuildTaskIdChange: (id: string | null) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function DocumentIcon({ docId, filename }: { docId: string; filename: string }) {
  if (docId.startsWith("url_")) {
    return <Link className="text-muted-foreground h-5 w-5 shrink-0" />;
  }
  const lower = filename.toLowerCase();
  if (lower.endsWith(".md") || lower.endsWith(".markdown") || lower.endsWith(".txt")) {
    return <FileCode className="text-muted-foreground h-5 w-5 shrink-0" />;
  }
  return <FileText className="text-muted-foreground h-5 w-5 shrink-0" />;
}

export function DocumentList({
  scopeId,
  scopeType,
  buildTaskId,
  onBuildTaskIdChange,
}: DocumentListProps) {
  const { data, isLoading } = useDocuments(scopeId, scopeType);
  const deleteDoc = useDeleteDocument();
  const buildIdx = useBuildIndex();

  const onTaskError = useCallback(
    (task: { error?: string | null }) => {
      onBuildTaskIdChange(null);
      toast.error(`Build failed: ${task.error ?? "Unknown error"}`);
    },
    [onBuildTaskIdChange]
  );

  const taskQuery = useTaskProgress(buildTaskId, {
    onError: onTaskError,
  });

  const isBuilding =
    buildIdx.isPending ||
    (buildTaskId != null &&
      taskQuery.data?.status !== "completed" &&
      taskQuery.data?.status !== "error");

  const progress = taskQuery.data?.progress ?? 0;
  const stage = taskQuery.data?.stage ?? "";

  const unindexedCount = data?.unindexed_count ?? 0;

  const handleDelete = async (docId: string) => {
    try {
      await deleteDoc.mutateAsync({ scopeId, scopeType, docId });
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Delete failed";
      toast.error(`Failed to delete document: ${msg}`);
    }
  };

  const handleBuildIndex = async () => {
    try {
      const result = await buildIdx.mutateAsync({ scopeId, scopeType });
      onBuildTaskIdChange(result.task_id);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Build failed";
      toast.error(`Failed to build index: ${msg}`);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header row with build action */}
      {data && data.documents.length > 0 && (
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground text-sm">
            {data.documents.length} document{data.documents.length !== 1 ? "s" : ""}
          </span>
          <div className="flex shrink-0 items-center gap-1.5">
            {unindexedCount > 0 ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 text-xs"
                onClick={handleBuildIndex}
                disabled={isBuilding}
              >
                {isBuilding ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Hammer className="h-3.5 w-3.5" />
                )}
                Build Index ({unindexedCount})
              </Button>
            ) : (
              <Badge variant="secondary" className="gap-1">
                <Check className="h-3 w-3" />
                Indexed
              </Badge>
            )}
          </div>
        </div>
      )}

      <DocumentUploader scopeId={scopeId} scopeType={scopeType} />

      {scopeType === "project" && <IndexingSettings projectId={scopeId} />}

      {/* Progress bar — prominent placement */}
      {isBuilding && (
        <div className="bg-muted/50 space-y-2 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <Loader2 className="text-primary h-4 w-4 animate-spin" />
            <p className="text-sm font-medium">{stage || "Starting build..."}</p>
          </div>
          <div className="bg-muted h-2 overflow-hidden rounded-full">
            <div
              className="bg-primary h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-muted-foreground text-xs">{progress}% complete</p>
        </div>
      )}

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
                <DocumentIcon docId={doc.doc_id} filename={doc.filename} />
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="truncate text-sm font-medium">{doc.filename}</p>
                    {doc.is_indexed === false && (
                      <span className="bg-primary/10 text-primary shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium">
                        New
                      </span>
                    )}
                  </div>
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
    </div>
  );
}
