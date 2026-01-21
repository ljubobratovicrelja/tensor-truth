import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Send, Bot, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useModels, useCreateSession, useFavoritePresets, useConfig } from "@/hooks";
import { useChatStore } from "@/stores";
import { ModuleSelector } from "@/components/chat/ModuleSelector";
import { SessionSettingsPanel } from "@/components/config";
import type { PresetInfo } from "@/api/types";

export function WelcomePage() {
  const [message, setMessage] = useState("");
  const [selectedModules, setSelectedModules] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [sessionParams, setSessionParams] = useState<Record<string, unknown>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: favoritesData } = useFavoritePresets();
  const { data: config } = useConfig();
  const createSession = useCreateSession();
  const navigate = useNavigate();
  const setPendingMessage = useChatStore((state) => state.setPendingUserMessage);

  const handleSubmit = async (promptText?: string, preset?: PresetInfo) => {
    const text = promptText ?? message.trim();
    if (!text || isSubmitting) return;

    setIsSubmitting(true);
    try {
      // Use preset config or selected options
      const presetModules = preset?.config?.modules as string[] | undefined;
      const presetModel = preset?.config?.model as string | undefined;

      const effectiveModules = selectedModules.length > 0 ? selectedModules : undefined;
      const effectiveModel =
        selectedModel && selectedModel !== "__none__" ? selectedModel : undefined;

      // Build params: start with session settings, add model if selected
      const baseParams =
        Object.keys(sessionParams).length > 0 ? { ...sessionParams } : {};
      const params = preset?.config
        ? { ...preset.config, model: presetModel }
        : effectiveModel
          ? { ...baseParams, model: effectiveModel }
          : Object.keys(baseParams).length > 0
            ? baseParams
            : undefined;

      const result = await createSession.mutateAsync({
        modules: presetModules ?? effectiveModules,
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePresetClick = (preset: PresetInfo) => {
    // Use the preset's description as a starting prompt, or a default greeting
    const description = preset.config?.description as string | undefined;
    const prompt = description
      ? `I'd like help with: ${description}`
      : `Start a ${preset.name} session`;
    handleSubmit(prompt, preset);
  };

  const canSend = message.trim().length > 0 && !isSubmitting;

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
          <div className="bg-muted/50 border-input relative rounded-2xl border shadow-sm">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your documents..."
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
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
                    <Bot className="h-3.5 w-3.5" />
                    <span className="text-xs">
                      {selectedModel && selectedModel !== "__none__"
                        ? selectedModel
                        : config?.models.default_rag_model || "Model"}
                    </span>
                  </SelectTrigger>
                  <SelectContent position="popper" side="top" className="max-h-[300px]">
                    <SelectItem value="__none__">
                      <span className="text-muted-foreground">
                        Default ({config?.models.default_rag_model || "..."})
                      </span>
                    </SelectItem>
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

          <p className="text-muted-foreground text-center text-xs">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>

        {/* Favorite Presets */}
        {favoritesData && favoritesData.presets.length > 0 && (
          <div className="space-y-3">
            <p className="text-muted-foreground text-center text-sm">
              Quick start with a preset
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {favoritesData.presets.map((preset) => (
                <Button
                  key={preset.name}
                  variant="outline"
                  size="sm"
                  onClick={() => handlePresetClick(preset)}
                  disabled={isSubmitting}
                  className="gap-2 rounded-full"
                  title={preset.config?.description as string | undefined}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  {preset.name}
                </Button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
