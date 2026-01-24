import { useState, useEffect } from "react";
import { Settings, Loader2, HelpCircle, RefreshCw, Plus } from "lucide-react";
import { toast } from "sonner";
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
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  useConfig,
  useUpdateConfig,
  useModels,
  useEmbeddingModels,
  useRerankers,
  useAddReranker,
  useReinitializeIndexes,
  useStartupStatus,
} from "@/hooks";
import type { ConfigResponse } from "@/api/types";

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

interface ConfigFormProps {
  config: ConfigResponse;
  onSave: (updates: Record<string, unknown>) => Promise<void>;
  isSaving: boolean;
}

const DEVICE_OPTIONS = ["cpu", "cuda", "mps"];

const CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

const REINITIALIZE_START_KEY = "tensortruth-reinitialize-start";

function ConfigForm({ config, onSave, isSaving }: ConfigFormProps) {
  const { data: modelsData } = useModels();
  const { data: embeddingModelsData } = useEmbeddingModels();
  const { data: rerankersData } = useRerankers();
  const addReranker = useAddReranker();
  const reinitializeIndexes = useReinitializeIndexes();

  // Add reranker dialog state
  const [addRerankerOpen, setAddRerankerOpen] = useState(false);
  const [newRerankerModel, setNewRerankerModel] = useState("");

  // Reinitialization progress tracking
  const [isReinitializing, setIsReinitializing] = useState(false);
  const [reinitializeStartTime, setReinitializeStartTime] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Poll startup status every 1s when reinitializing, otherwise don't poll
  const pollingInterval = isReinitializing ? 1000 : false;
  const { data: startupStatus } = useStartupStatus(
    pollingInterval ? { pollingInterval } : undefined
  );

  // Models
  const [ragModel, setRagModel] = useState(config.models.default_rag_model);

  // Generation
  const [temperature, setTemperature] = useState(config.ui.default_temperature);
  const [contextWindow, setContextWindow] = useState(config.ui.default_context_window);
  const [maxTokens, setMaxTokens] = useState(config.ui.default_max_tokens);

  // Retrieval
  const [reranker, setReranker] = useState(config.rag.default_reranker);
  const [topN, setTopN] = useState(config.ui.default_top_n);
  const [confidenceThreshold, setConfidenceThreshold] = useState(
    config.ui.default_confidence_threshold
  );
  const [confidenceCutoffHard, setConfidenceCutoffHard] = useState(
    config.ui.default_confidence_cutoff_hard
  );

  // Hardware
  const [device, setDevice] = useState(config.rag.default_device);
  const [availableDevices, setAvailableDevices] = useState<string[]>(DEVICE_OPTIONS);

  // Embedding Model
  const [embeddingModel, setEmbeddingModel] = useState(
    config.rag.default_embedding_model
  );

  // History Cleaning
  const [historyCleaningEnabled, setHistoryCleaningEnabled] = useState(
    config.history_cleaning.enabled
  );
  const [removeEmojis, setRemoveEmojis] = useState(config.history_cleaning.remove_emojis);
  const [removeFillerPhrases, setRemoveFillerPhrases] = useState(
    config.history_cleaning.remove_filler_phrases
  );
  const [normalizeWhitespace, setNormalizeWhitespace] = useState(
    config.history_cleaning.normalize_whitespace
  );
  const [collapseNewlines, setCollapseNewlines] = useState(
    config.history_cleaning.collapse_newlines
  );

  // History Size (turns = user query + assistant response pairs)
  const [maxHistoryTurns, setMaxHistoryTurns] = useState(config.rag.max_history_turns);
  const [memoryTokenLimit, setMemoryTokenLimit] = useState(config.rag.memory_token_limit);

  // Web Search
  const [ddgMaxResults, setDdgMaxResults] = useState(config.web_search.ddg_max_results);
  const [maxPagesToFetch, setMaxPagesToFetch] = useState(
    config.web_search.max_pages_to_fetch
  );
  const [rerankTitleThreshold, setRerankTitleThreshold] = useState(
    config.web_search.rerank_title_threshold
  );
  const [rerankContentThreshold, setRerankContentThreshold] = useState(
    config.web_search.rerank_content_threshold
  );
  const [maxSourceContextPct, setMaxSourceContextPct] = useState(
    config.web_search.max_source_context_pct
  );
  const [inputContextPct, setInputContextPct] = useState(
    config.web_search.input_context_pct
  );

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

  // Restore reinitialize state from localStorage on mount
  useEffect(() => {
    if (!startupStatus) return;

    const storedReinitializeStart = localStorage.getItem(REINITIALIZE_START_KEY);
    if (storedReinitializeStart && !startupStatus.indexes_ok) {
      const startTime = parseInt(storedReinitializeStart, 10);
      setIsReinitializing(true);
      setReinitializeStartTime(startTime);
    } else if (startupStatus.indexes_ok) {
      localStorage.removeItem(REINITIALIZE_START_KEY);
    }
  }, [startupStatus]);

  // Update elapsed time every second while reinitializing
  useEffect(() => {
    if (!isReinitializing || !reinitializeStartTime) {
      setElapsedSeconds(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsedSeconds(Math.round((Date.now() - reinitializeStartTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isReinitializing, reinitializeStartTime]);

  // Detect when reinitialization completes
  useEffect(() => {
    if (isReinitializing && startupStatus?.indexes_ok) {
      setIsReinitializing(false);
      localStorage.removeItem(REINITIALIZE_START_KEY);
      const elapsed = reinitializeStartTime ? Date.now() - reinitializeStartTime : 0;
      const elapsedSeconds = Math.round(elapsed / 1000);
      toast.success(`Indexes reinitialized successfully! (${elapsedSeconds}s)`);
    }
  }, [isReinitializing, startupStatus?.indexes_ok, reinitializeStartTime]);

  const handleSave = async () => {
    await onSave({
      models_default_rag_model: ragModel,
      ui_default_temperature: temperature,
      ui_default_context_window: contextWindow,
      ui_default_max_tokens: maxTokens,
      ui_default_top_n: topN,
      ui_default_confidence_threshold: confidenceThreshold,
      ui_default_confidence_cutoff_hard: confidenceCutoffHard,
      rag_default_device: device,
      rag_default_embedding_model: embeddingModel,
      rag_default_reranker: reranker,
      history_cleaning_enabled: historyCleaningEnabled,
      history_cleaning_remove_emojis: removeEmojis,
      history_cleaning_remove_filler_phrases: removeFillerPhrases,
      history_cleaning_normalize_whitespace: normalizeWhitespace,
      history_cleaning_collapse_newlines: collapseNewlines,
      rag_max_history_turns: maxHistoryTurns,
      rag_memory_token_limit: memoryTokenLimit,
      web_search_ddg_max_results: ddgMaxResults,
      web_search_max_pages_to_fetch: maxPagesToFetch,
      web_search_rerank_title_threshold: rerankTitleThreshold,
      web_search_rerank_content_threshold: rerankContentThreshold,
      web_search_max_source_context_pct: maxSourceContextPct,
      web_search_input_context_pct: inputContextPct,
    });
  };

  const handleReinitialize = () => {
    if (
      !confirm(
        "This will delete all existing indexes and download fresh copies from HuggingFace Hub. This may take several minutes. Continue?"
      )
    ) {
      return;
    }

    const startTime = Date.now();
    setIsReinitializing(true);
    setReinitializeStartTime(startTime);
    localStorage.setItem(REINITIALIZE_START_KEY, startTime.toString());

    reinitializeIndexes.mutate(undefined, {
      onSuccess: (response) => {
        toast.info(response.message);
      },
      onError: (error) => {
        setIsReinitializing(false);
        localStorage.removeItem(REINITIALIZE_START_KEY);
        toast.error(`Failed to reinitialize indexes: ${error.message}`);
      },
    });
  };

  return (
    <div className="space-y-6">
      {/* Models Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Models</h3>
        <div className="space-y-3">
          <div className="space-y-2">
            <Label>
              RAG Model
              <HelpTooltip text="Primary model for retrieval-augmented generation. Used when answering questions with document context." />
            </Label>
            <Select value={ragModel} onValueChange={setRagModel}>
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
      </div>

      <Separator />

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
              <HelpTooltip text="Model used for vector embeddings. Changing this switches to a different set of indexes." />
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
                    {model.model_id} ({model.index_count} modules)
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
              <Select value={reranker} onValueChange={setReranker}>
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
                        placeholder="e.g., Qwen/Qwen3-Reranker-0.6B"
                      />
                      <p className="text-muted-foreground text-xs">
                        Enter the full HuggingFace model path. The model will be validated
                        before adding.
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
                                setReranker(response.model || newRerankerModel.trim());
                                setAddRerankerOpen(false);
                                setNewRerankerModel("");
                              } else {
                                toast.error(response.error || "Failed to add reranker");
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
              <span className="text-muted-foreground text-sm">{topN}</span>
            </div>
            <Slider
              value={[topN]}
              onValueChange={([v]) => setTopN(v)}
              min={1}
              max={20}
              step={1}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Confidence Threshold</Label>
              <span className="text-muted-foreground text-sm">
                {confidenceThreshold.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[confidenceThreshold]}
              onValueChange={([v]) => setConfidenceThreshold(v)}
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

      {/* Hardware Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Hardware</h3>
        <div className="space-y-2">
          <Label>RAG Device</Label>
          <Select value={device} onValueChange={setDevice}>
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
          <p className="text-muted-foreground text-xs">
            Device for embedding model and reranker
          </p>
        </div>
      </div>

      <Separator />

      {/* Chat History Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Chat History</h3>
        <p className="text-muted-foreground text-xs">
          Control how much conversation history is included in prompts.
        </p>
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

        <Separator className="my-4" />

        <h4 className="text-sm font-medium">History Cleaning</h4>
        <p className="text-muted-foreground text-xs">
          Reduce token usage by cleaning chat history before sending to the LLM.
        </p>
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Checkbox
              id="history-cleaning-enabled"
              checked={historyCleaningEnabled}
              onCheckedChange={(checked) => setHistoryCleaningEnabled(checked === true)}
            />
            <Label htmlFor="history-cleaning-enabled" className="cursor-pointer">
              Enable history cleaning
              <HelpTooltip text="Master switch for all history cleaning operations." />
            </Label>
          </div>

          <div className="ml-6 space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="remove-emojis"
                checked={removeEmojis}
                disabled={!historyCleaningEnabled}
                onCheckedChange={(checked) => setRemoveEmojis(checked === true)}
              />
              <Label
                htmlFor="remove-emojis"
                className={
                  historyCleaningEnabled
                    ? "cursor-pointer"
                    : "text-muted-foreground cursor-not-allowed"
                }
              >
                Remove emojis
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="remove-filler-phrases"
                checked={removeFillerPhrases}
                disabled={!historyCleaningEnabled}
                onCheckedChange={(checked) => setRemoveFillerPhrases(checked === true)}
              />
              <Label
                htmlFor="remove-filler-phrases"
                className={
                  historyCleaningEnabled
                    ? "cursor-pointer"
                    : "text-muted-foreground cursor-not-allowed"
                }
              >
                Remove filler phrases
                <HelpTooltip text="Removes common LLM pleasantries like 'Great question!' and 'Hope this helps!'" />
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="normalize-whitespace"
                checked={normalizeWhitespace}
                disabled={!historyCleaningEnabled}
                onCheckedChange={(checked) => setNormalizeWhitespace(checked === true)}
              />
              <Label
                htmlFor="normalize-whitespace"
                className={
                  historyCleaningEnabled
                    ? "cursor-pointer"
                    : "text-muted-foreground cursor-not-allowed"
                }
              >
                Normalize whitespace
                <HelpTooltip text="Collapses multiple spaces into single spaces (preserves indentation)." />
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="collapse-newlines"
                checked={collapseNewlines}
                disabled={!historyCleaningEnabled}
                onCheckedChange={(checked) => setCollapseNewlines(checked === true)}
              />
              <Label
                htmlFor="collapse-newlines"
                className={
                  historyCleaningEnabled
                    ? "cursor-pointer"
                    : "text-muted-foreground cursor-not-allowed"
                }
              >
                Collapse excessive newlines
                <HelpTooltip text="Reduces 3+ consecutive newlines to 2." />
              </Label>
            </div>
          </div>
        </div>
      </div>

      <Separator />

      {/* Web Search Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Web Search</h3>
        <p className="text-muted-foreground text-xs">
          Configure web search behavior and quality thresholds for /web commands.
        </p>

        {/* Search Limits */}
        <div className="bg-muted/30 space-y-3 rounded-lg border p-3">
          <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Search Limits
          </h4>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Max Search Results
                <HelpTooltip text="Maximum results to fetch from DuckDuckGo. Higher values = more candidates to choose from." />
              </Label>
              <span className="text-muted-foreground text-sm">{ddgMaxResults}</span>
            </div>
            <Slider
              value={[ddgMaxResults]}
              onValueChange={([v]) => setDdgMaxResults(v)}
              min={5}
              max={20}
              step={1}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Max Pages to Fetch
                <HelpTooltip text="Maximum pages to download and process. Higher values = more comprehensive results." />
              </Label>
              <span className="text-muted-foreground text-sm">{maxPagesToFetch}</span>
            </div>
            <Slider
              value={[maxPagesToFetch]}
              onValueChange={([v]) => setMaxPagesToFetch(v)}
              min={1}
              max={10}
              step={1}
            />
          </div>
        </div>

        {/* Relevance Thresholds */}
        <div className="bg-muted/30 space-y-3 rounded-lg border p-3">
          <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Relevance Thresholds
          </h4>
          <p className="text-muted-foreground text-xs">
            Sources below these thresholds are rejected. Lower = more lenient.
          </p>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Title Threshold
                <HelpTooltip text="Minimum relevance score for search result titles/snippets (0-50%). Sources below this are not fetched." />
              </Label>
              <span className="text-muted-foreground text-sm">
                {(rerankTitleThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[rerankTitleThreshold]}
              onValueChange={([v]) => setRerankTitleThreshold(v)}
              min={0}
              max={0.5}
              step={0.05}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Content Threshold
                <HelpTooltip text="Minimum relevance score for fetched page content (0-50%). Sources below this are excluded from summary." />
              </Label>
              <span className="text-muted-foreground text-sm">
                {(rerankContentThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[rerankContentThreshold]}
              onValueChange={([v]) => setRerankContentThreshold(v)}
              min={0}
              max={0.5}
              step={0.05}
            />
          </div>
        </div>

        {/* Context Fitting */}
        <div className="bg-muted/30 space-y-3 rounded-lg border p-3">
          <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Context Fitting
          </h4>
          <p className="text-muted-foreground text-xs">
            Control how source content is distributed within the context window.
          </p>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Max Source Context
                <HelpTooltip text="Maximum % of context window for a single source. Prevents one source from dominating." />
              </Label>
              <span className="text-muted-foreground text-sm">
                {(maxSourceContextPct * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[maxSourceContextPct]}
              onValueChange={([v]) => setMaxSourceContextPct(v)}
              min={0.05}
              max={0.3}
              step={0.01}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">
                Input Context
                <HelpTooltip text="% of context window for input (sources). Rest is reserved for LLM output." />
              </Label>
              <span className="text-muted-foreground text-sm">
                {(inputContextPct * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[inputContextPct]}
              onValueChange={([v]) => setInputContextPct(v)}
              min={0.4}
              max={0.8}
              step={0.05}
            />
          </div>
        </div>
      </div>

      <Separator />

      {/* Maintenance Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Maintenance</h3>
        <div className="space-y-3">
          <div className="bg-muted/50 rounded-lg border p-4">
            <div className="mb-3 flex items-start gap-2">
              <RefreshCw className="text-muted-foreground mt-0.5 h-4 w-4" />
              <div className="flex-1">
                <h4 className="text-sm font-medium">Reinitialize Indexes</h4>
                <p className="text-muted-foreground mt-1 text-xs">
                  Delete all existing vector indexes and download fresh copies from
                  HuggingFace Hub. Use this to fix corrupted indexes or update to the
                  latest version. (~3.9GB download)
                </p>
              </div>
            </div>
            <Button
              onClick={handleReinitialize}
              disabled={isReinitializing}
              variant="outline"
              size="sm"
              className="w-full"
            >
              {isReinitializing ? (
                <>
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  Reinitializing...
                </>
              ) : (
                <>
                  <RefreshCw className="mr-2 h-3 w-3" />
                  Reinitialize Indexes
                </>
              )}
            </Button>
            {isReinitializing && (
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    Reinitialization in progress...
                  </span>
                  <span className="text-muted-foreground font-mono">
                    {elapsedSeconds}s
                  </span>
                </div>
                <div className="bg-secondary relative h-1.5 overflow-hidden rounded-full">
                  <div className="from-primary/50 via-primary to-primary/50 absolute inset-0 animate-pulse bg-gradient-to-r" />
                  <div className="via-primary/30 animate-shimmer absolute inset-0 bg-gradient-to-r from-transparent to-transparent" />
                </div>
                <p className="text-muted-foreground text-xs">
                  Deleting old indexes and downloading fresh copies. This may take several
                  minutes.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : (
            "Save Changes"
          )}
        </Button>
      </div>
    </div>
  );
}

export function ConfigPanel() {
  const { data: config, isLoading: configLoading } = useConfig();
  const updateConfig = useUpdateConfig();
  const [open, setOpen] = useState(false);

  const handleSave = async (updates: Record<string, unknown>) => {
    try {
      await updateConfig.mutateAsync(updates);
      setOpen(false);
    } catch (error) {
      console.error("Failed to save config:", error);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[85vh] max-w-lg overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        {configLoading || !config ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : (
          <ConfigForm
            key={open ? "open" : "closed"}
            config={config}
            onSave={handleSave}
            isSaving={updateConfig.isPending}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
