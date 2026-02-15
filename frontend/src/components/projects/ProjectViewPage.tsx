import { useParams } from "react-router-dom";
import { FolderKanban } from "lucide-react";
import { useProject } from "@/hooks";
import { formatRelativeTime } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

export function ProjectViewPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: project, isLoading, error } = useProject(projectId ?? null);

  if (isLoading) {
    return (
      <div className="mx-auto w-full max-w-2xl space-y-4 p-6">
        <Skeleton className="h-8 w-1/2" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/3" />
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center">
        <p>Project not found</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-auto">
      <div className="mx-auto w-full max-w-2xl space-y-6 p-6">
        <div className="flex items-start gap-3">
          <FolderKanban className="text-muted-foreground mt-1 h-6 w-6 shrink-0" />
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold">{project.name}</h1>
            {project.description && (
              <p className="text-muted-foreground">{project.description}</p>
            )}
            <p className="text-muted-foreground text-xs">
              Created {formatRelativeTime(project.created_at)} &middot; Updated{" "}
              {formatRelativeTime(project.updated_at)}
            </p>
          </div>
        </div>

        <div className="border-border space-y-3 rounded-lg border p-4">
          <h2 className="text-sm font-medium">Project Details</h2>
          <div className="text-muted-foreground space-y-1 text-sm">
            <p>Documents: {project.documents.length}</p>
            <p>Sessions: {project.session_ids.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
