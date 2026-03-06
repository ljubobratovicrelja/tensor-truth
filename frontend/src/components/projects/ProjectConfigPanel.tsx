import { useState, useEffect, useRef, useCallback } from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useModels, useConfig } from "@/hooks";
import type { ProjectResponse } from "@/api/types";

const CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

interface ProjectConfigPanelProps {
  project: ProjectResponse;
  onUpdate: (config: Record<string, unknown>) => void;
}

export function ProjectConfigPanel({ project, onUpdate }: ProjectConfigPanelProps) {
  const { data: modelsData } = useModels();
  const { data: config } = useConfig();

  const configRef = useRef(project.config);
  useEffect(() => {
    configRef.current = project.config;
  }, [project.config]);

  // Derive current values from project config with fallbacks
  const systemPrompt = (project.config.system_prompt as string) ?? "";
  const serverModel = (project.config.model as string) ?? "";
  const temperature =
    (project.config.temperature as number) ?? config?.llm.default_temperature ?? 0.7;
  const contextWindow =
    (project.config.context_window as number) ??
    config?.llm.default_context_window ??
    8192;

  // Debounced system prompt
  const [localPrompt, setLocalPrompt] = useState(systemPrompt);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Local model state so selection is instant (not waiting for server roundtrip)
  const [localModel, setLocalModel] = useState(serverModel);

  // Sync local prompt when project changes externally
  useEffect(() => {
    setLocalPrompt(systemPrompt);
  }, [systemPrompt]);

  // Sync local model when server value changes
  useEffect(() => {
    setLocalModel(serverModel);
  }, [serverModel]);

  const handlePromptChange = useCallback(
    (value: string) => {
      setLocalPrompt(value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        const updated = { ...configRef.current };
        if (value.trim()) {
          updated.system_prompt = value.trim();
        } else {
          delete updated.system_prompt;
        }
        onUpdate(updated);
      }, 1000);
    },
    [onUpdate]
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleImmediateChange = useCallback(
    (field: string, value: unknown) => {
      onUpdate({ ...configRef.current, [field]: value });
    },
    [onUpdate]
  );

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">Configuration</h3>

      {/* System Prompt */}
      <div className="space-y-2">
        <Label className="text-xs">System Prompt</Label>
        <Textarea
          value={localPrompt}
          onChange={(e) => handlePromptChange(e.target.value)}
          placeholder="Custom instructions for this project..."
          className="min-h-[80px] resize-y text-xs"
        />
      </div>

      {/* Model */}
      <div className="space-y-2">
        <Label className="text-xs">Model</Label>
        <Select
          value={localModel || undefined}
          onValueChange={(v) => {
            setLocalModel(v);
            handleImmediateChange("model", v);
          }}
        >
          <SelectTrigger className="text-xs">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {modelsData?.models.map((m) => (
              <SelectItem key={m.name} value={m.name} className="text-xs">
                {m.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Temperature */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Temperature</Label>
          <span className="text-muted-foreground text-xs">{temperature}</span>
        </div>
        <Slider
          value={[temperature]}
          onValueChange={([v]) => handleImmediateChange("temperature", v)}
          min={0}
          max={2}
          step={0.1}
        />
      </div>

      {/* Context Window */}
      <div className="space-y-2">
        <Label className="text-xs">Context Window</Label>
        <Select
          value={String(contextWindow)}
          onValueChange={(v) => handleImmediateChange("context_window", Number(v))}
        >
          <SelectTrigger className="text-xs">
            <SelectValue placeholder="Select context window" />
          </SelectTrigger>
          <SelectContent>
            {CONTEXT_WINDOW_OPTIONS.map((size) => (
              <SelectItem key={size} value={String(size)} className="text-xs">
                {size.toLocaleString()}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
