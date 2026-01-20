import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useModules, useModels, useCreateSession } from "@/hooks";

export function NewSessionDialog() {
  const [open, setOpen] = useState(false);
  const [selectedModule, setSelectedModule] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");

  const { data: modulesData, isLoading: modulesLoading } = useModules();
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const createSession = useCreateSession();
  const navigate = useNavigate();

  const handleCreate = async () => {
    try {
      const result = await createSession.mutateAsync({
        modules: selectedModule ? [selectedModule] : undefined,
        params: selectedModel ? { model: selectedModel } : undefined,
      });
      navigate(`/chat/${result.session_id}`);
      setOpen(false);
      setSelectedModule("");
      setSelectedModel("");
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="w-full" variant="outline">
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Chat Session</DialogTitle>
          <DialogDescription>
            Select a knowledge module and model for your chat session.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="module">Knowledge Module</Label>
            <Select value={selectedModule} onValueChange={setSelectedModule}>
              <SelectTrigger id="module">
                <SelectValue placeholder="Select a module (optional)" />
              </SelectTrigger>
              <SelectContent>
                {modulesLoading ? (
                  <SelectItem value="loading" disabled>
                    Loading...
                  </SelectItem>
                ) : modulesData?.modules.length === 0 ? (
                  <SelectItem value="none" disabled>
                    No modules available
                  </SelectItem>
                ) : (
                  modulesData?.modules.map((module) => (
                    <SelectItem key={module.name} value={module.name}>
                      {module.name}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-2">
            <Label htmlFor="model">Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger id="model">
                <SelectValue placeholder="Select a model (optional)" />
              </SelectTrigger>
              <SelectContent>
                {modelsLoading ? (
                  <SelectItem value="loading" disabled>
                    Loading...
                  </SelectItem>
                ) : modelsData?.models.length === 0 ? (
                  <SelectItem value="none" disabled>
                    No models available
                  </SelectItem>
                ) : (
                  modelsData?.models.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={createSession.isPending}>
            {createSession.isPending ? "Creating..." : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
