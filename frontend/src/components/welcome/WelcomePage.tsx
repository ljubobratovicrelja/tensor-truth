import { useState, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import { Send, Bot, FolderKanban, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger } from "@/components/ui/select";
import { ModelSelectContent } from "@/components/chat/ModelSelectContent";
import { cn } from "@/lib/utils";
import {
  useModels,
  useCreateSession,
  useProjects,
  useConfig,
  useCommandDetection,
  useThinking,
  thinkingToParam,
} from "@/hooks";
import { useChatStore } from "@/stores";
import { ModuleSelector } from "@/components/chat/ModuleSelector";
import { CommandAutocomplete } from "@/components/chat/CommandAutocomplete";
import { ThinkingSelect } from "@/components/chat/ThinkingSelect";
import { SessionSettingsPanel } from "@/components/config";
import type { CommandDefinition } from "@/types/commands";

export function WelcomePage() {
  const [message, setMessage] = useState("");
  const [selectedModules, setSelectedModules] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [sessionParams, setSessionParams] = useState<Record<string, unknown>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [autocompleteHasResults, setAutocompleteHasResults] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: projectsData } = useProjects();
  const { data: config } = useConfig();
  const createSession = useCreateSession();
  const navigate = useNavigate();
  const setPendingMessage = useChatStore((state) => state.setPendingUserMessage);
  const detection = useCommandDetection(message);

  // Show autocomplete only if command detected AND no space after command name
  const commandEndPos =
    detection.commandPosition + (detection.commandName?.length || 0) + 1;
  const hasSpaceAfterCommand = message.charAt(commandEndPos) === " ";
  const showAutocomplete = detection.hasCommand && !hasSpaceAfterCommand;

  // Derive effective model: user selection, config default, or first available
  const effectiveModel =
    selectedModel || config?.llm.default_model || modelsData?.models[0]?.name || "";

  const {
    thinking,
    thinkingSupport,
    handleModelChange: handleThinkingModelChange,
    setThinking,
  } = useThinking({ modelsData, effectiveModel });

  const handleModelSelect = (model: string) => {
    setSelectedModel(model);
    handleThinkingModelChange(model);
  };

  const handleSubmit = async (promptText?: string) => {
    const text = promptText ?? message.trim();
    if (!text || isSubmitting) return;

    setIsSubmitting(true);
    try {
      const thinkingValue = thinkingToParam(thinking);
      const params = {
        ...sessionParams,
        model: effectiveModel,
        ...(thinkingValue !== undefined && { thinking: thinkingValue }),
      };

      const result = await createSession.mutateAsync({
        modules: selectedModules.length > 0 ? selectedModules : undefined,
        params,
      });

      // Set the pending message so it shows immediately in chat
      setPendingMessage(text);

      // Navigate to the new chat session
      navigate(`/chat/${result.session_id}?autoSend=true`);
    } catch (error) {
      console.error("Failed to create session:", error);
      toast.error("Failed to start chat session");
      setIsSubmitting(false);
    }
  };

  const handleCommandSelect = (command: CommandDefinition) => {
    // Replace the current command with the selected one
    if (detection.commandPosition >= 0) {
      const before = message.slice(0, detection.commandPosition);
      const after = message.slice(
        detection.commandPosition + (detection.commandName?.length || 0) + 1
      );
      const commandName = command.usage.split(" ")[0];
      setMessage(`${before}${commandName} ${after}`.trim() + " ");
    } else {
      const commandName = command.usage.split(" ")[0];
      setMessage(`${message} ${commandName} `);
    }
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Let autocomplete handle navigation keys if it's open and has results
    if (
      showAutocomplete &&
      autocompleteHasResults &&
      ["Enter", "Tab", "ArrowUp", "ArrowDown", "Escape"].includes(e.key)
    ) {
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSend = message.trim().length > 0 && !isSubmitting && !!effectiveModel;

  // Projects data
  const projects = projectsData?.projects ?? [];
  const recentProjects = projects.slice(0, 4);
  const hasMoreProjects = projects.length > 4;

  return (
    <div className="flex h-full flex-col items-center justify-center px-4 pb-[env(safe-area-inset-bottom)]">
      <div className="w-full max-w-2xl space-y-8">
        {/* Logo and Greeting */}
        <div className="text-center">
          <div className="mb-4 flex items-center justify-center gap-3">
            <img src="/logo.png" alt="TensorTruth" className="h-24 w-24" />
          </div>
          <h1 className="text-foreground text-3xl font-semibold tracking-tight">
            What would you like to know?
          </h1>
          <p className="text-muted-foreground mt-2 text-base">
            Ask questions about your documents with AI-powered retrieval
          </p>
        </div>

        {/* Chat Input */}
        <div className="space-y-4">
          <div className="bg-muted/50 border-input relative flex flex-col overflow-visible rounded-2xl border shadow-sm transition-all duration-500">
            {/* Command Autocomplete */}
            <CommandAutocomplete
              input={message}
              isOpen={showAutocomplete}
              onSelect={handleCommandSelect}
              onHasResultsChange={setAutocompleteHasResults}
              onClose={() => {
                if (detection.commandPosition >= 0) {
                  const before = message.slice(0, detection.commandPosition);
                  const after = message.slice(
                    detection.commandPosition + (detection.commandName?.length || 0) + 1
                  );
                  setMessage((before + after).trim());
                }
                textareaRef.current?.focus();
              }}
            />

            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your documents..."
              disabled={isSubmitting}
              className={cn(
                "min-h-[100px] w-full resize-none bg-transparent px-4 pt-4 pb-2 text-base",
                "placeholder:text-muted-foreground focus:outline-none",
                "disabled:cursor-not-allowed disabled:opacity-50"
              )}
            />

            {/* Bottom toolbar */}
            <div className="flex items-center justify-between px-3 pb-3">
              {/* Left side - RAG config */}
              <div className="flex items-center gap-1">
                {/* Module selector */}
                <ModuleSelector
                  selectedModules={selectedModules}
                  onModulesChange={setSelectedModules}
                  disabled={isSubmitting}
                  embeddingModel={sessionParams.embedding_model as string | undefined}
                />

                {/* Session settings */}
                <SessionSettingsPanel
                  currentParams={sessionParams}
                  onChange={setSessionParams}
                  disabled={isSubmitting}
                />

                {/* Model selector */}
                <Select value={selectedModel} onValueChange={handleModelSelect}>
                  <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
                    <Bot className="h-3.5 w-3.5" />
                    <span className="text-xs">{effectiveModel || "No model"}</span>
                  </SelectTrigger>
                  <ModelSelectContent
                    models={modelsData?.models ?? []}
                    isLoading={modelsLoading}
                    position="popper"
                    side="top"
                    className="!max-h-[300px]"
                  />
                </Select>
                {thinkingSupport.thinking && (
                  <ThinkingSelect
                    value={thinking}
                    onValueChange={setThinking}
                    disabled={isSubmitting}
                    supportsLevels={thinkingSupport.levels}
                  />
                )}
              </div>

              {/* Right side - send button */}
              <Button
                onClick={() => handleSubmit()}
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

          {!modelsLoading && modelsData?.models.length === 0 && (
            <p className="text-destructive text-center text-xs">
              No Ollama models available. Start Ollama and pull a model to begin.
            </p>
          )}
          <p className="text-muted-foreground text-center text-xs">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>

        {/* Projects Quick Access */}
        <div className="space-y-3">
          {projects.length > 0 ? (
            <>
              <p className="text-muted-foreground text-center text-sm">Your projects</p>
              <div className="flex flex-wrap items-center justify-center gap-2">
                {recentProjects.map((project) => (
                  <Button
                    key={project.project_id}
                    variant="outline"
                    size="sm"
                    asChild
                    className="gap-2 rounded-full"
                  >
                    <Link to={`/projects/${project.project_id}`}>
                      <FolderKanban className="h-3.5 w-3.5" />
                      {project.name}
                    </Link>
                  </Button>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  asChild
                  className="gap-2 rounded-full"
                >
                  <Link to="/projects/new">
                    <Plus className="h-3.5 w-3.5" />
                    New Project
                  </Link>
                </Button>
              </div>
              {hasMoreProjects && (
                <p className="text-center">
                  <Link
                    to="/projects"
                    className="text-muted-foreground hover:text-foreground text-xs underline underline-offset-2 transition-colors"
                  >
                    View all projects
                  </Link>
                </p>
              )}
            </>
          ) : (
            <>
              <p className="text-muted-foreground text-center text-sm">
                Organize your work with Projects
              </p>
              <div className="flex justify-center">
                <Button
                  variant="outline"
                  size="sm"
                  asChild
                  className="gap-2 rounded-full"
                >
                  <Link to="/projects/new">
                    <Plus className="h-3.5 w-3.5" />
                    New Project
                  </Link>
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
