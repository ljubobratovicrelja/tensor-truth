import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Plus, Search, FolderKanban, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useProjects, useDeleteProject } from "@/hooks";
import { formatRelativeTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Skeleton } from "@/components/ui/skeleton";

export function ProjectsListPage() {
  const { data, isLoading, error } = useProjects();
  const deleteProject = useDeleteProject();
  const [search, setSearch] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const navigate = useNavigate();

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteProject.mutateAsync(deleteTarget.id);
      toast.success(`Deleted project "${deleteTarget.name}"`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Delete failed";
      toast.error(`Failed to delete project: ${msg}`);
    }
    setDeleteTarget(null);
  };

  const filteredProjects = data?.projects.filter((project) => {
    const query = search.toLowerCase();
    return (
      project.name.toLowerCase().includes(query) ||
      project.description.toLowerCase().includes(query)
    );
  });

  return (
    <div className="flex h-full flex-col overflow-auto">
      <div className="mx-auto w-full max-w-5xl space-y-6 p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Projects</h1>
          <Button asChild>
            <Link to="/projects/new">
              <Plus className="mr-2 h-4 w-4" />
              New Project
            </Link>
          </Button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
          <Input
            placeholder="Search projects..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Card key={i} className="p-0">
                <CardHeader>
                  <Skeleton className="h-5 w-3/4" />
                  <Skeleton className="h-4 w-full" />
                </CardHeader>
                <CardFooter>
                  <Skeleton className="h-3 w-1/3" />
                </CardFooter>
              </Card>
            ))}
          </div>
        ) : error ? (
          <div className="text-muted-foreground py-12 text-center">
            <p>Failed to load projects. Please try again.</p>
          </div>
        ) : filteredProjects && filteredProjects.length > 0 ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filteredProjects.map((project) => (
              <ContextMenu key={project.project_id}>
                <ContextMenuTrigger asChild>
                  <Card
                    className="hover:border-foreground/20 cursor-pointer transition-colors"
                    onClick={() => navigate(`/projects/${project.project_id}`)}
                  >
                    <CardHeader>
                      <CardTitle className="truncate text-base">{project.name}</CardTitle>
                      {project.description && (
                        <CardDescription className="line-clamp-2">
                          {project.description}
                        </CardDescription>
                      )}
                    </CardHeader>
                    <CardFooter>
                      <span className="text-muted-foreground text-xs">
                        Updated {formatRelativeTime(project.updated_at)}
                      </span>
                    </CardFooter>
                  </Card>
                </ContextMenuTrigger>
                <ContextMenuContent>
                  <ContextMenuItem
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteTarget({
                        id: project.project_id,
                        name: project.name,
                      });
                    }}
                    className="text-destructive focus:text-destructive"
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete
                  </ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            ))}
          </div>
        ) : (
          <div className="py-16 text-center">
            <FolderKanban className="text-muted-foreground mx-auto mb-4 h-12 w-12" />
            <h2 className="text-lg font-medium">
              {search ? "No matching projects" : "No projects yet"}
            </h2>
            <p className="text-muted-foreground mt-1 text-sm">
              {search
                ? "Try adjusting your search terms."
                : "Create your first project to organize documents and chats."}
            </p>
          </div>
        )}
      </div>

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deleteTarget?.name}"? This will also
              delete all its sessions and documents. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
