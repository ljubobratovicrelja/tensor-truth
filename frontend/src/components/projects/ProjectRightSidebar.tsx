import { useCallback } from "react";
import { X, Check, Loader2, AlertCircle, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useProject, useUpdateProject, useRemoveCatalogModule } from "@/hooks";
import { useUIStore } from "@/stores";
import { ProjectConfigPanel } from "./ProjectConfigPanel";
import { DocumentPanel } from "@/components/documents";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface ProjectRightSidebarProps {
  projectId: string;
}

function ModuleStatusIndicator({ status }: { status: string }) {
  switch (status) {
    case "indexed":
      return <Check className="h-3.5 w-3.5 text-green-500" />;
    case "building":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-yellow-500" />;
    case "error":
      return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
    default:
      return <div className="h-3.5 w-3.5 rounded-full bg-gray-400" />;
  }
}

export function ProjectRightSidebar({ projectId }: ProjectRightSidebarProps) {
  const { data: project, isLoading } = useProject(projectId);
  const updateProject = useUpdateProject();
  const removeCatalogModule = useRemoveCatalogModule();
  const setRightSidebarOpen = useUIStore((state) => state.setRightSidebarOpen);

  const handleConfigUpdate = useCallback(
    (newConfig: Record<string, unknown>) => {
      updateProject.mutate(
        { projectId, data: { config: newConfig } },
        {
          onError: () => {
            toast.error("Failed to save project configuration");
          },
        }
      );
    },
    [projectId, updateProject]
  );

  const handleRemoveModule = async (moduleName: string) => {
    try {
      await removeCatalogModule.mutateAsync({ projectId, moduleName });
      toast.success(`Removed module: ${moduleName}`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Remove failed";
      toast.error(`Failed to remove module: ${msg}`);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4 p-4">
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-full" />
      </div>
    );
  }

  if (!project) {
    return <div className="text-muted-foreground p-4 text-sm">Project not found.</div>;
  }

  const moduleEntries = Object.entries(project.catalog_modules);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-border flex items-center justify-between border-b px-4 py-3">
        <h2 className="truncate text-sm font-semibold">{project.name}</h2>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={() => setRightSidebarOpen(false)}
          title="Close sidebar"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Scrollable content */}
      <ScrollArea className="flex-1">
        <div className="space-y-4 p-4">
          <ProjectConfigPanel project={project} onUpdate={handleConfigUpdate} />
          <Separator />

          {/* Catalog Modules */}
          {moduleEntries.length > 0 && (
            <div className="space-y-1.5">
              <h3 className="text-sm font-medium">Catalog Modules</h3>
              <ul className="space-y-1">
                {moduleEntries.map(([name, info]) => (
                  <li
                    key={name}
                    className="flex items-center gap-2 rounded px-1.5 py-1 text-xs"
                  >
                    <ModuleStatusIndicator status={info.status} />
                    <span className="truncate">{name}</span>
                    <span
                      className={cn(
                        "ml-auto shrink-0 text-[10px]",
                        info.status === "indexed" && "text-green-600 dark:text-green-400",
                        info.status === "building" &&
                          "text-yellow-600 dark:text-yellow-400",
                        info.status === "error" && "text-red-600 dark:text-red-400"
                      )}
                    >
                      {info.status}
                    </span>
                    {info.status !== "building" && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-5 w-5 shrink-0"
                        onClick={() => handleRemoveModule(name)}
                        disabled={removeCatalogModule.isPending}
                        title={`Remove ${name}`}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {moduleEntries.length > 0 && <Separator />}

          {/* Documents */}
          <DocumentPanel scopeId={projectId} scopeType="project" />
        </div>
      </ScrollArea>
    </div>
  );
}
