import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { FileText, FileCode, Link, Trash2, Loader2, Hammer, Check } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { QUERY_KEYS } from "@/lib/constants";
import { useDocuments, useDeleteDocument, useBuildIndex } from "@/hooks";
import { useTaskProgress } from "@/hooks/useTasks";
import type { ScopeType } from "@/api/types";
import { DocumentUploader } from "./DocumentUploader";
import { IndexingSettings } from "./IndexingSettings";

interface DocumentListProps {
  scopeId: string;
  scopeType: ScopeType;
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

export function DocumentList({ scopeId, scopeType }: DocumentListProps) {
  const queryClient = useQueryClient();
  const { data, isLoading } = useDocuments(scopeId, scopeType);
  const deleteDoc = useDeleteDocument();
  const buildIdx = useBuildIndex();

  const [buildTaskId, setBuildTaskId] = useState<string | null>(null);

  const onTaskComplete = useCallback(() => {
    setBuildTaskId(null);
    queryClient.invalidateQueries({
      queryKey: QUERY_KEYS.documents(scopeType, scopeId),
    });
    if (scopeType === "project") {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.project(scopeId),
      });
    }
    toast.success("Index built successfully");
  }, [queryClient, scopeType, scopeId]);

  const onTaskError = useCallback((task: { error?: string | null }) => {
    setBuildTaskId(null);
    toast.error(`Build failed: ${task.error ?? "Unknown error"}`);
  }, []);

  const taskQuery = useTaskProgress(buildTaskId, {
    onComplete: onTaskComplete,
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
      setBuildTaskId(result.task_id);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Build failed";
      toast.error(`Failed to build index: ${msg}`);
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center gap-2">
        <CardTitle className="text-lg">Documents</CardTitle>
        {data && data.documents.length > 0 && (
          <div className="ml-auto flex shrink-0 items-center gap-1.5">
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
        )}
      </CardHeader>
      <CardContent className="space-y-4 pb-14">
        <DocumentUploader scopeId={scopeId} scopeType={scopeType} />

        {scopeType === "project" && <IndexingSettings projectId={scopeId} />}

        {isBuilding && (
          <div className="space-y-1.5">
            {stage && (
              <p className="text-muted-foreground text-center text-xs">{stage}</p>
            )}
            <div className="bg-muted h-1.5 overflow-hidden rounded-full">
              <div
                className="bg-primary h-full rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
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
      </CardContent>
    </Card>
  );
}
