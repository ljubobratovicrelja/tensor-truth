import { useState, useEffect } from "react";
import { Settings, Loader2, HelpCircle, Plus, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import {
  useConfig,
  useUpdateSession,
  useModels,
  useEmbeddingModels,
  useRerankers,
  useAddReranker,
  useRestartEngine,
} from "@/hooks";
import { toast } from "sonner";

function HelpTooltip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <HelpCircle className="text-muted-foreground ml-1 inline h-3.5 w-3.5 cursor-help" />
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        {text}
      </TooltipContent>
    </Tooltip>
  );
}

const DEVICE_OPTIONS = ["cpu", "cuda", "mps"];
const LLM_DEVICE_OPTIONS = ["cpu", "gpu"];
const CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

interface SessionSettingsPanelProps {
  sessionId?: string;
  currentParams?: Record<string, unknown>;
  disabled?: boolean;
  onChange?: (params: Record<string, unknown>) => void;
}

export function SessionSettingsPanel({
  sessionId,
  currentParams = {},
  disabled = false,
  onChange,
}: SessionSettingsPanelProps) {
  const [open, setOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const { data: config } = useConfig();
  const { data: modelsData } = useModels();
  const { data: embeddingModelsData } = useEmbeddingModels();
  const { data: rerankersData } = useRerankers();
  const addReranker = useAddReranker();
  const updateSession = useUpdateSession();
  const restartEngine = useRestartEngine();

  // Add reranker dialog state
  const [addRerankerOpen, setAddRerankerOpen] = useState(false);
  const [newRerankerModel, setNewRerankerModel] = useState("");

  // Form state - initialized from session params or config defaults
  const [temperature, setTemperature] = useState<number>(0.7);
  const [contextWindow, setContextWindow] = useState<number>(8192);
  const [maxTokens, setMaxTokens] = useState<number>(2048);
  const [rerankerModel, setRerankerModel] = useState<string>("");
  const [rerankerTopN, setRerankerTopN] = useState<number>(5);
  const [confidenceCutoff, setConfidenceCutoff] = useState<number>(0.3);
  const [confidenceCutoffHard, setConfidenceCutoffHard] = useState<number>(0.1);
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [ragDevice, setRagDevice] = useState<string>("cpu");
  const [llmDevice, setLlmDevice] = useState<string>("gpu");
  const [embeddingModel, setEmbeddingModel] = useState<string>("");
  const [availableDevices, setAvailableDevices] = useState<string[]>(DEVICE_OPTIONS);
  const [maxHistoryTurns, setMaxHistoryTurns] = useState<number>(3);
  const [memoryTokenLimit, setMemoryTokenLimit] = useState<number>(4000);
  const [routerModel, setRouterModel] = useState<string>("");
  const [functionAgentModel, setFunctionAgentModel] = useState<string>("");

  // Fetch available devices from backend
  useEffect(() => {
    import("@/api/config").then(({ getAvailableDevices }) => {
      getAvailableDevices()
        .then(setAvailableDevices)
        .catch(() => {
          // Fallback to default options if fetch fails
          setAvailableDevices(DEVICE_OPTIONS);
        });
    });
  }, []);

  // Reset form when dialog opens
  useEffect(() => {
    if (open && config) {
      setTemperature(
        (currentParams.temperature as number) ?? config.ui.default_temperature
      );
      setContextWindow(
        (currentParams.context_window as number) ?? config.ui.default_context_window
      );
      setMaxTokens((currentParams.max_tokens as number) ?? config.ui.default_max_tokens);
      setRerankerModel(
        (currentParams.reranker_model as string) ?? config.rag.default_reranker
      );
      setRerankerTopN(
        (currentParams.reranker_top_n as number) ?? config.ui.default_top_n
      );
      setConfidenceCutoff(
        (currentParams.confidence_cutoff as number) ??
          config.ui.default_confidence_threshold
      );
      setConfidenceCutoffHard(
        (currentParams.confidence_cutoff_hard as number) ??
          config.ui.default_confidence_cutoff_hard
      );
      setSystemPrompt((currentParams.system_prompt as string) ?? "");
      setRagDevice((currentParams.rag_device as string) ?? config.rag.default_device);
      setLlmDevice((currentParams.llm_device as string) ?? "gpu");
      setEmbeddingModel(
        (currentParams.embedding_model as string) ?? config.rag.default_embedding_model
      );
      setMaxHistoryTurns(
        (currentParams.max_history_turns as number) ?? config.rag.max_history_turns
      );
      setMemoryTokenLimit(
        (currentParams.memory_token_limit as number) ?? config.rag.memory_token_limit
      );
      setRouterModel((currentParams.router_model as string) ?? config.agent.router_model);
      setFunctionAgentModel(
        (currentParams.function_agent_model as string) ??
          config.agent.function_agent_model
      );
    }
  }, [open, config, currentParams]);

  const handleSave = async () => {
    const newParams: Record<string, unknown> = {
      ...currentParams,
      temperature,
      context_window: contextWindow,
      max_tokens: maxTokens,
      reranker_model: rerankerModel,
      reranker_top_n: rerankerTopN,
      confidence_cutoff: confidenceCutoff,
      confidence_cutoff_hard: confidenceCutoffHard,
      rag_device: ragDevice,
      llm_device: llmDevice,
      embedding_model: embeddingModel,
      max_history_turns: maxHistoryTurns,
      memory_token_limit: memoryTokenLimit,
      router_model: routerModel,
      function_agent_model: functionAgentModel,
    };

    // Only include system_prompt if non-empty
    if (systemPrompt.trim()) {
      newParams.system_prompt = systemPrompt.trim();
    } else {
      delete newParams.system_prompt;
    }

    // If onChange provided (welcome page mode), just call it
    if (onChange) {
      onChange(newParams);
      setOpen(false);
      return;
    }

    // Otherwise update existing session via API
    if (!sessionId) return;

    setIsSaving(true);
    try {
      await updateSession.mutateAsync({
        sessionId,
        data: { params: newParams },
      });
      setOpen(false);
    } catch {
      toast.error("Failed to save session settings");
    } finally {
      setIsSaving(false);
    }
  };

  const handleRestartEngine = async () => {
    try {
      await restartEngine.mutateAsync();
      toast.success("Engine restarted successfully");
    } catch {
      toast.error("Failed to restart engine");
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          disabled={disabled}
          title="Session settings"
        >
          <Settings className="h-3.5 w-3.5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[85vh] max-w-lg overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Session Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Generation Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Generation</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Temperature</Label>
                  <span className="text-muted-foreground text-sm">{temperature}</span>
                </div>
                <Slider
                  value={[temperature]}
                  onValueChange={([v]) => setTemperature(v)}
                  min={0}
                  max={2}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>Context Window</Label>
                <Select
                  value={String(contextWindow)}
                  onValueChange={(v) => setContextWindow(Number(v))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select context window" />
                  </SelectTrigger>
                  <SelectContent>
                    {CONTEXT_WINDOW_OPTIONS.map((size) => (
                      <SelectItem key={size} value={String(size)}>
                        {size.toLocaleString()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Max Tokens</Label>
                  <span className="text-muted-foreground text-sm">{maxTokens}</span>
                </div>
                <Slider
                  value={[maxTokens]}
                  onValueChange={([v]) => setMaxTokens(v)}
                  min={256}
                  max={16384}
                  step={256}
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Retrieval Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Retrieval</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>
                  Embedding Model
                  <HelpTooltip text="The embedding model used for vector search. Only models with built indexes are available." />
                </Label>
                <Select value={embeddingModel} onValueChange={setEmbeddingModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select embedding model" />
                  </SelectTrigger>
                  <SelectContent>
                    {embeddingModelsData?.models.map((model) => (
                      <SelectItem
                        key={model.model_id}
                        value={model.model_name || model.model_id}
                      >
                        {model.model_id} ({model.index_count} indexes)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>
                  Reranker Model
                  <HelpTooltip text="Cross-encoder model that re-scores retrieved documents for better relevance ranking." />
                </Label>
                <div className="flex gap-2">
                  <Select value={rerankerModel} onValueChange={setRerankerModel}>
                    <SelectTrigger className="flex-1">
                      <SelectValue placeholder="Select reranker" />
                    </SelectTrigger>
                    <SelectContent>
                      {rerankersData?.models.map((model) => (
                        <SelectItem key={model.model} value={model.model}>
                          {model.model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Dialog open={addRerankerOpen} onOpenChange={setAddRerankerOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="icon" title="Add custom reranker">
                        <Plus className="h-4 w-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-md">
                      <DialogHeader>
                        <DialogTitle>Add Reranker Model</DialogTitle>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label>HuggingFace Model Path</Label>
                          <Input
                            value={newRerankerModel}
                            onChange={(e) => setNewRerankerModel(e.target.value)}
                            placeholder="e.g., BAAI/bge-reranker-v2-m3"
                          />
                          <p className="text-muted-foreground text-xs">
                            Enter the full HuggingFace model path. The model will be
                            validated before adding.
                          </p>
                        </div>
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="outline"
                            onClick={() => {
                              setAddRerankerOpen(false);
                              setNewRerankerModel("");
                            }}
                          >
                            Cancel
                          </Button>
                          <Button
                            onClick={() => {
                              if (!newRerankerModel.trim()) return;
                              addReranker.mutate(newRerankerModel.trim(), {
                                onSuccess: (response) => {
                                  if (response.status === "added") {
                                    toast.success(`Reranker "${response.model}" added`);
                                    setRerankerModel(
                                      response.model || newRerankerModel.trim()
                                    );
                                    setAddRerankerOpen(false);
                                    setNewRerankerModel("");
                                  } else {
                                    toast.error(
                                      response.error || "Failed to add reranker"
                                    );
                                  }
                                },
                                onError: (error) => {
                                  toast.error(`Failed to add reranker: ${error.message}`);
                                },
                              });
                            }}
                            disabled={addReranker.isPending || !newRerankerModel.trim()}
                          >
                            {addReranker.isPending ? (
                              <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Validating...
                              </>
                            ) : (
                              "Add"
                            )}
                          </Button>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Top N Sources</Label>
                  <span className="text-muted-foreground text-sm">{rerankerTopN}</span>
                </div>
                <Slider
                  value={[rerankerTopN]}
                  onValueChange={([v]) => setRerankerTopN(v)}
                  min={1}
                  max={20}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Confidence Threshold</Label>
                  <span className="text-muted-foreground text-sm">
                    {confidenceCutoff.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[confidenceCutoff]}
                  onValueChange={([v]) => setConfidenceCutoff(v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Hard Cutoff</Label>
                  <span className="text-muted-foreground text-sm">
                    {confidenceCutoffHard.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[confidenceCutoffHard]}
                  onValueChange={([v]) => setConfidenceCutoffHard(v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Chat History Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Chat History</h3>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>
                    Max History Turns
                    <HelpTooltip text="Number of conversation turns to include in the prompt. 1 turn = 1 user query + 1 assistant response. Lower values = faster responses and lower cost." />
                  </Label>
                  <span className="text-muted-foreground text-sm">{maxHistoryTurns}</span>
                </div>
                <Slider
                  value={[maxHistoryTurns]}
                  onValueChange={([v]) => setMaxHistoryTurns(v)}
                  min={0}
                  max={10}
                  step={1}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>
                    Memory Token Limit
                    <HelpTooltip text="Maximum tokens stored in chat memory buffer. Acts as a safety backstop - if total history exceeds this, oldest messages are dropped. Usually the message count limit above is what matters." />
                  </Label>
                  <span className="text-muted-foreground text-sm">
                    {memoryTokenLimit.toLocaleString()}
                  </span>
                </div>
                <Slider
                  value={[Math.min(memoryTokenLimit, contextWindow)]}
                  onValueChange={([v]) => setMemoryTokenLimit(v)}
                  min={1000}
                  max={contextWindow}
                  step={1000}
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Agent Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Agent</h3>
            <div className="space-y-2">
              <Label>
                Reasoning Model
                <HelpTooltip text="Model used by routing agents for step-by-step decisions (e.g., search, fetch, summarize). Smaller, fast models work well here." />
              </Label>
              <Select value={routerModel} onValueChange={setRouterModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {modelsData?.models.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>
                Function Model
                <HelpTooltip text="Model used by function agents that call tools autonomously via LLM tool-calling. Needs a model with good tool-use support." />
              </Label>
              <Select value={functionAgentModel} onValueChange={setFunctionAgentModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {modelsData?.models.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          {/* Hardware Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Hardware</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>RAG Device</Label>
                <Select value={ragDevice} onValueChange={setRagDevice}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select device" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableDevices.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option.toUpperCase()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>LLM Device</Label>
                <Select value={llmDevice} onValueChange={setLlmDevice}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select device" />
                  </SelectTrigger>
                  <SelectContent>
                    {LLM_DEVICE_OPTIONS.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option.toUpperCase()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleRestartEngine}
                disabled={restartEngine.isPending}
                className="w-full"
              >
                {restartEngine.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Restarting...
                  </>
                ) : (
                  <>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Restart Engine
                  </>
                )}
              </Button>
              <p className="text-muted-foreground text-xs">
                Clears GPU/MPS memory and restarts the RAG pipeline
              </p>
            </div>
          </div>

          <Separator />

          {/* System Prompt Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Custom Instructions</h3>
            <div className="space-y-2">
              <Label>System Prompt</Label>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="Add custom instructions for this chat session..."
                className="min-h-[100px] resize-y"
              />
            </div>
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={isSaving}>
              {isSaving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save"
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
