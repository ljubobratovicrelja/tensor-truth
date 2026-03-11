import { useState, useCallback } from "react";
import { FileUp, Loader2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { QUERY_KEYS } from "@/lib/constants";
import { useTaskProgress } from "@/hooks/useTasks";
import type { ScopeType } from "@/api/types";
import { DocumentPanel } from "./DocumentPanel";

interface DocumentDialogProps {
  scopeId: string;
  scopeType: ScopeType;
}

export function DocumentDialog({ scopeId, scopeType }: DocumentDialogProps) {
  const queryClient = useQueryClient();
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

  const onTaskError = useCallback(() => {
    setBuildTaskId(null);
  }, []);

  const taskQuery = useTaskProgress(buildTaskId, {
    onComplete: onTaskComplete,
    onError: onTaskError,
  });

  const isBuilding =
    buildTaskId != null &&
    taskQuery.data?.status !== "completed" &&
    taskQuery.data?.status !== "error";

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          title={isBuilding ? "Building indexes..." : "Manage documents"}
        >
          {isBuilding ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <FileUp className="mr-2 h-4 w-4" />
          )}
          Documents
        </Button>
      </DialogTrigger>
      <DialogContent className="flex max-h-[80vh] max-w-2xl flex-col">
        <DialogHeader>
          <DialogTitle>Manage Documents</DialogTitle>
        </DialogHeader>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <DocumentPanel
            scopeId={scopeId}
            scopeType={scopeType}
            buildTaskId={buildTaskId}
            onBuildTaskIdChange={setBuildTaskId}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
