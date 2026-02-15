import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Plus, Search, FolderKanban } from "lucide-react";
import { useProjects } from "@/hooks";
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
import { Skeleton } from "@/components/ui/skeleton";

export function ProjectsListPage() {
  const { data, isLoading, error } = useProjects();
  const [search, setSearch] = useState("");
  const navigate = useNavigate();

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
              <Card
                key={project.project_id}
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
            {!search && (
              <Button asChild className="mt-4">
                <Link to="/projects/new">
                  <Plus className="mr-2 h-4 w-4" />
                  Create Project
                </Link>
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
