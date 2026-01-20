import { useState } from "react";
import { Settings, Loader2 } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { useConfig, useUpdateConfig, useModels } from "@/hooks";
import type { ConfigResponse } from "@/api/types";

interface ConfigFormProps {
  config: ConfigResponse;
  onSave: (updates: Record<string, unknown>) => Promise<void>;
  isSaving: boolean;
}

function ConfigForm({ config, onSave, isSaving }: ConfigFormProps) {
  const { data: modelsData } = useModels();

  const [temperature, setTemperature] = useState(config.ui.default_temperature);
  const [contextWindow, setContextWindow] = useState(config.ui.default_context_window);
  const [maxTokens, setMaxTokens] = useState(config.ui.default_max_tokens);
  const [topN, setTopN] = useState(config.ui.default_top_n);
  const [ragModel, setRagModel] = useState(config.models.default_rag_model);
  const [fallbackModel, setFallbackModel] = useState(
    config.models.default_fallback_model
  );

  const handleSave = async () => {
    await onSave({
      ui_default_temperature: temperature,
      ui_default_context_window: contextWindow,
      ui_default_max_tokens: maxTokens,
      ui_default_top_n: topN,
      models_default_rag_model: ragModel,
      models_default_fallback_model: fallbackModel,
    });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Models</h3>
        <div className="space-y-3">
          <div className="space-y-2">
            <Label>RAG Model</Label>
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
            <Label>Fallback Model</Label>
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

      <div className="space-y-4">
        <h3 className="text-sm font-medium">Generation Parameters</h3>
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
            <div className="flex items-center justify-between">
              <Label>Context Window</Label>
              <span className="text-muted-foreground text-sm">{contextWindow}</span>
            </div>
            <Slider
              value={[contextWindow]}
              onValueChange={([v]) => setContextWindow(v)}
              min={2048}
              max={32768}
              step={1024}
            />
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
              max={8192}
              step={256}
            />
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
      <DialogContent className="max-w-lg">
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
