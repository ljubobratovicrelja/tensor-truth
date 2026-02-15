import { useCallback } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useProject, useUpdateProject, useRemoveCatalogModule } from "@/hooks";
import { useUIStore } from "@/stores";
import { ProjectConfigPanel } from "./ProjectConfigPanel";
import { CatalogModuleItem } from "./CatalogModuleItem";
import { CatalogModuleAddPicker } from "./CatalogModuleAddPicker";
import { DocumentPanel } from "@/components/documents";
import { toast } from "sonner";

interface ProjectRightSidebarProps {
  projectId: string;
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
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Catalog Modules</h3>
              <CatalogModuleAddPicker
                projectId={projectId}
                existingModules={moduleEntries.map(([name]) => name)}
              />
            </div>
            {moduleEntries.length > 0 ? (
              <ul className="space-y-1">
                {moduleEntries.map(([name, info]) => (
                  <CatalogModuleItem
                    key={name}
                    projectId={projectId}
                    moduleName={name}
                    status={info.status}
                    taskId={info.task_id}
                    onRemove={handleRemoveModule}
                    isRemoving={removeCatalogModule.isPending}
                  />
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground text-xs">
                No modules added yet. Click + to add one.
              </p>
            )}
          </div>

          <Separator />

          {/* Documents */}
          <DocumentPanel scopeId={projectId} scopeType="project" />
        </div>
      </ScrollArea>
    </div>
  );
}
