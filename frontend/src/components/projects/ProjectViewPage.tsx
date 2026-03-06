import { useState, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Send, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useProject, useModels, useConfig, useCreateProjectSession } from "@/hooks";
import { useChatStore } from "@/stores";
import { ModuleSelector } from "@/components/chat/ModuleSelector";
import { SessionSettingsPanel } from "@/components/config/SessionSettingsPanel";

export function ProjectViewPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: project, isLoading, error } = useProject(projectId ?? null);
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: config } = useConfig();
  const createProjectSession = useCreateProjectSession();
  const navigate = useNavigate();
  const setPendingMessage = useChatStore((state) => state.setPendingUserMessage);

  const [message, setMessage] = useState("");
  const [selectedModules, setSelectedModules] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [sessionParams, setSessionParams] = useState<Record<string, unknown>>({});
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Derive effective model: user selection, project config, or system default
  const projectModel = (project?.config?.model as string) || "";
  const effectiveModel = selectedModel || projectModel || config?.llm.default_model || "";

  const handleSubmit = async () => {
    const text = message.trim();
    if (!text || isSubmitting || !projectId) return;

    setIsSubmitting(true);
    try {
      const params: Record<string, unknown> = { ...sessionParams, model: effectiveModel };

      const result = await createProjectSession.mutateAsync({
        projectId,
        data: {
          modules: selectedModules.length > 0 ? selectedModules : undefined,
          params,
        },
      });

      // Set the pending message so it shows immediately in chat
      setPendingMessage(text);

      // Navigate to the new chat session within the project
      navigate(`/projects/${projectId}/chat/${result.session_id}?autoSend=true`);
    } catch (err) {
      console.error("Failed to create project session:", err);
      toast.error("Failed to start chat session");
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSend = message.trim().length > 0 && !isSubmitting;

  if (isLoading) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-4">
        <div className="w-full max-w-2xl space-y-4">
          <Skeleton className="mx-auto h-8 w-1/2" />
          <Skeleton className="mx-auto h-4 w-3/4" />
          <Skeleton className="h-32 w-full rounded-2xl" />
        </div>
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

  const lockedModules = Object.keys(project.catalog_modules);

  return (
    <div className="flex h-full flex-col items-center justify-center px-4 pb-[env(safe-area-inset-bottom)]">
      <div className="w-full max-w-2xl space-y-8">
        {/* Project heading */}
        <div className="text-center">
          <h1 className="text-foreground text-3xl font-semibold tracking-tight">
            {project.name}
          </h1>
          {project.description && (
            <p className="text-muted-foreground mt-2 text-base">{project.description}</p>
          )}
        </div>

        {/* Chat Input */}
        <div className="space-y-4">
          <div className="bg-muted/50 border-input relative overflow-visible rounded-2xl border shadow-sm">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about this project..."
              disabled={isSubmitting}
              className={cn(
                "min-h-[100px] w-full resize-none bg-transparent px-4 pt-4 pb-16 text-base",
                "placeholder:text-muted-foreground focus:outline-none",
                "disabled:cursor-not-allowed disabled:opacity-50"
              )}
            />

            {/* Bottom toolbar */}
            <div className="absolute right-3 bottom-3 left-3 flex items-center justify-between">
              {/* Left side - RAG config */}
              <div className="flex items-center gap-1">
                {/* Module selector with locked project modules */}
                <ModuleSelector
                  selectedModules={selectedModules}
                  onModulesChange={setSelectedModules}
                  disabled={isSubmitting}
                  lockedModules={lockedModules}
                  projectDocuments={project.documents}
                />

                {/* Session settings */}
                <SessionSettingsPanel
                  currentParams={sessionParams}
                  disabled={isSubmitting}
                  onChange={setSessionParams}
                  hideSystemPrompt
                />

                {/* Model selector */}
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
                    <Bot className="h-3.5 w-3.5" />
                    <span className="text-xs">{effectiveModel || "Model"}</span>
                  </SelectTrigger>
                  <SelectContent position="popper" side="top" className="max-h-[300px]">
                    {modelsLoading ? (
                      <SelectItem value="loading" disabled>
                        Loading...
                      </SelectItem>
                    ) : (
                      modelsData?.models
                        .slice()
                        .sort((a, b) => a.name.localeCompare(b.name))
                        .map((model) => (
                          <SelectItem key={model.name} value={model.name}>
                            {model.name}
                          </SelectItem>
                        ))
                    )}
                  </SelectContent>
                </Select>
              </div>

              {/* Right side - send button */}
              <Button
                onClick={handleSubmit}
                disabled={!canSend}
                size="icon"
                className="h-9 w-9 rounded-xl"
                title="Start chat"
              >
                {isSubmitting ? (
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>

          <p className="text-muted-foreground text-center text-xs">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>

        {/* Encouraging text */}
        <p className="text-muted-foreground text-center text-sm">
          Start a conversation to explore your project knowledge base.
          {lockedModules.length > 0 &&
            ` ${lockedModules.length} module${lockedModules.length === 1 ? "" : "s"} loaded.`}
          {project.documents.length > 0 &&
            ` ${project.documents.length} document${project.documents.length === 1 ? "" : "s"} available.`}
        </p>
      </div>
    </div>
  );
}
