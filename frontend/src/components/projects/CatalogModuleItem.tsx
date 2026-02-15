import { Check, Loader2, AlertCircle, Trash2, RotateCw } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useTaskProgress, useAddCatalogModule } from "@/hooks";
import { generateDisplayName } from "@/lib/moduleUtils";
import { QUERY_KEYS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface CatalogModuleItemProps {
  projectId: string;
  moduleName: string;
  status: string;
  taskId?: string | null;
  onRemove: (name: string) => void;
  isRemoving: boolean;
}

export function CatalogModuleItem({
  projectId,
  moduleName,
  status,
  taskId,
  onRemove,
  isRemoving,
}: CatalogModuleItemProps) {
  const queryClient = useQueryClient();
  const addModule = useAddCatalogModule();
  const displayName = generateDisplayName(moduleName);

  const { data: task } = useTaskProgress(
    status === "building" && taskId ? taskId : null,
    {
      onComplete: () => {
        toast.success(`Module "${displayName}" indexed successfully`);
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.project(projectId) });
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.modules });
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.embeddingModels });
      },
      onError: (t) => {
        toast.error(
          `Module "${displayName}" build failed: ${t.error || "Unknown error"}`
        );
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.project(projectId) });
      },
    }
  );

  const handleRetry = async () => {
    try {
      await addModule.mutateAsync({ projectId, moduleName });
      toast.success(`Retrying build for "${displayName}"`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Retry failed";
      toast.error(`Failed to retry module: ${msg}`);
    }
  };

  // Building state
  if (status === "building" && taskId) {
    const progress = task?.progress ?? 0;
    const stage = task?.stage ?? "Queued";

    return (
      <li className="space-y-1 rounded px-1.5 py-1">
        <div className="flex items-center gap-2 text-xs">
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-yellow-500" />
          <span className="truncate">{displayName}</span>
          <span className="text-muted-foreground ml-auto shrink-0 text-[10px]">
            {Math.round(progress)}%
          </span>
        </div>
        <Progress value={progress} className="h-1" />
        <p className="text-muted-foreground truncate text-[10px]">{stage}</p>
      </li>
    );
  }

  // Error state
  if (status === "error") {
    return (
      <li className="flex items-center gap-2 rounded px-1.5 py-1 text-xs">
        <AlertCircle className="h-3.5 w-3.5 shrink-0 text-red-500" />
        <span className="truncate">{displayName}</span>
        <span
          className={cn("ml-auto shrink-0 text-[10px]", "text-red-600 dark:text-red-400")}
        >
          error
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 shrink-0"
          onClick={handleRetry}
          disabled={addModule.isPending}
          title={`Retry ${moduleName}`}
        >
          <RotateCw className={cn("h-3 w-3", addModule.isPending && "animate-spin")} />
        </Button>
      </li>
    );
  }

  // Indexed state
  if (status === "indexed") {
    return (
      <li className="flex items-center gap-2 rounded px-1.5 py-1 text-xs">
        <Check className="h-3.5 w-3.5 shrink-0 text-green-500" />
        <span className="truncate">{displayName}</span>
        <span
          className={cn(
            "ml-auto shrink-0 text-[10px]",
            "text-green-600 dark:text-green-400"
          )}
        >
          indexed
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 shrink-0"
          onClick={() => onRemove(moduleName)}
          disabled={isRemoving}
          title={`Remove ${moduleName}`}
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </li>
    );
  }

  // Pending / other state
  return (
    <li className="flex items-center gap-2 rounded px-1.5 py-1 text-xs">
      <div className="h-3.5 w-3.5 shrink-0 rounded-full bg-gray-400" />
      <span className="truncate">{displayName}</span>
      <span className="text-muted-foreground ml-auto shrink-0 text-[10px]">{status}</span>
    </li>
  );
}
