import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useCreateProject } from "@/hooks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export function ProjectsNewPage() {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const createProject = useCreateProject();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    try {
      const result = await createProject.mutateAsync({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      toast.success("Project created");
      navigate(`/projects/${result.project_id}`);
    } catch (err) {
      console.error("Failed to create project:", err);
      toast.error("Failed to create project");
    }
  };

  return (
    <div className="flex h-full flex-col overflow-auto">
      <div className="mx-auto w-full max-w-lg space-y-6 p-6">
        <h1 className="text-2xl font-semibold">New Project</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="project-name" className="text-sm leading-none font-medium">
              Name
            </label>
            <Input
              id="project-name"
              placeholder="My project"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <label
              htmlFor="project-description"
              className="text-sm leading-none font-medium"
            >
              Description
            </label>
            <Textarea
              id="project-description"
              placeholder="What is this project about?"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>

          <div className="flex gap-3">
            <Button type="button" variant="outline" onClick={() => navigate("/projects")}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || createProject.isPending}>
              {createProject.isPending ? "Creating..." : "Create Project"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
