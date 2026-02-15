import { useCallback } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useProject, useUpdateProject } from "@/hooks";
import { useUIStore } from "@/stores";
import { ProjectConfigPanel } from "./ProjectConfigPanel";
import { ProjectKnowledgePanel } from "./ProjectKnowledgePanel";
import { toast } from "sonner";

interface ProjectRightSidebarProps {
  projectId: string;
}

export function ProjectRightSidebar({ projectId }: ProjectRightSidebarProps) {
  const { data: project, isLoading } = useProject(projectId);
  const updateProject = useUpdateProject();
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
          <ProjectKnowledgePanel project={project} />
        </div>
      </ScrollArea>
    </div>
  );
}
