import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Send, Bot, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useModels, useCreateSession, usePresets, useConfig } from "@/hooks";
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
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: presetsData } = usePresets();
  const { data: config } = useConfig();
  const createSession = useCreateSession();
  const navigate = useNavigate();
  const setPendingMessage = useChatStore((state) => state.setPendingUserMessage);

  // Derive effective model: user selection or config default
  const effectiveModel = selectedModel || config?.models.default_rag_model || "";

  const handleSubmit = async (promptText?: string) => {
    const text = promptText ?? message.trim();
    if (!text || isSubmitting) return;

    setIsSubmitting(true);
    try {
      const params = { ...sessionParams, model: effectiveModel };

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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePresetClick = (preset: PresetInfo) => {
    const config = preset.config || {};

    // Apply modules from preset
    if (Array.isArray(config.modules)) {
      setSelectedModules(config.modules as string[]);
    }

    // Apply model from preset
    if (config.model) {
      setSelectedModel(config.model as string);
    }

    // Build session params from preset config
    const newParams: Record<string, unknown> = {};
    const paramKeys = [
      "temperature",
      "context_window",
      "reranker_model",
      "reranker_top_n",
      "confidence_cutoff",
      "confidence_cutoff_hard",
      "system_prompt",
      "embedding_model",
    ];
    paramKeys.forEach((key) => {
      if (config[key] !== undefined) newParams[key] = config[key];
    });
    setSessionParams(newParams);

    // Track active preset & trigger animation
    setActivePreset(preset.name);
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 800);
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
          <div
            className={cn(
              "bg-muted/50 border-input relative rounded-2xl border shadow-sm transition-all duration-500",
              isAnimating && "preset-glow"
            )}
          >
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

        {/* Presets */}
        {presetsData && presetsData.presets.length > 0 && (
          <div className="space-y-3">
            <p className="text-muted-foreground text-center text-sm">
              Quick start with a preset
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {presetsData.presets.map((preset) => (
                <Button
                  key={preset.name}
                  variant={activePreset === preset.name ? "default" : "outline"}
                  size="sm"
                  onClick={() => handlePresetClick(preset)}
                  disabled={isSubmitting}
                  className={cn(
                    "gap-2 rounded-full transition-all duration-300",
                    activePreset === preset.name && "ring-primary/50 scale-105 ring-2"
                  )}
                  title={preset.config?.description as string | undefined}
                >
                  <Sparkles
                    className={cn(
                      "h-3.5 w-3.5 transition-all",
                      activePreset === preset.name && "text-yellow-400"
                    )}
                  />
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
