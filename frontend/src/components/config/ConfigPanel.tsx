import { useState, useEffect } from "react";
import { Settings, Loader2, HelpCircle } from "lucide-react";
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
import { useConfig, useUpdateConfig, useModels } from "@/hooks";
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

const RERANKER_OPTIONS = [
  "BAAI/bge-reranker-v2-m3",
  "BAAI/bge-reranker-base",
  "cross-encoder/ms-marco-MiniLM-L-6-v2",
];

const DEVICE_OPTIONS = ["cpu", "cuda", "mps"];

const CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

function ConfigForm({ config, onSave, isSaving }: ConfigFormProps) {
  const { data: modelsData } = useModels();

  // Models
  const [ragModel, setRagModel] = useState(config.models.default_rag_model);
  const [fallbackModel, setFallbackModel] = useState(
    config.models.default_fallback_model
  );

  // Generation
  const [temperature, setTemperature] = useState(config.ui.default_temperature);
  const [contextWindow, setContextWindow] = useState(config.ui.default_context_window);
  const [maxTokens, setMaxTokens] = useState(config.ui.default_max_tokens);

  // Retrieval
  const [reranker, setReranker] = useState(config.ui.default_reranker);
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

  const handleSave = async () => {
    await onSave({
      models_default_rag_model: ragModel,
      models_default_fallback_model: fallbackModel,
      ui_default_temperature: temperature,
      ui_default_context_window: contextWindow,
      ui_default_max_tokens: maxTokens,
      ui_default_reranker: reranker,
      ui_default_top_n: topN,
      ui_default_confidence_threshold: confidenceThreshold,
      ui_default_confidence_cutoff_hard: confidenceCutoffHard,
      rag_default_device: device,
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
          <div className="space-y-2">
            <Label>
              Fallback Model
              <HelpTooltip text="Backup model used when the primary model is unavailable or Ollama fails to list models." />
            </Label>
            <Select value={fallbackModel} onValueChange={setFallbackModel}>
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
              Reranker Model
              <HelpTooltip text="Cross-encoder model that re-scores retrieved documents for better relevance ranking." />
            </Label>
            <Select value={reranker} onValueChange={setReranker}>
              <SelectTrigger>
                <SelectValue placeholder="Select reranker" />
              </SelectTrigger>
              <SelectContent>
                {RERANKER_OPTIONS.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
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
