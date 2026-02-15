import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Settings2, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { QUERY_KEYS } from "@/lib/constants";
import { useIndexingConfig, useUpdateIndexingConfig } from "@/hooks";
import type { IndexingConfig } from "@/api/types";

const CHUNK_PRESETS: Record<string, number[]> = {
  papers: [2048, 512, 256],
  books: [3072, 768, 384],
  custom: [],
};

const PRESET_LABELS: Record<string, string> = {
  papers: "Papers (2048 / 512 / 256)",
  books: "Books (3072 / 768 / 384)",
  custom: "Custom",
};

function presetFromSizes(sizes: number[]): string {
  for (const [key, preset] of Object.entries(CHUNK_PRESETS)) {
    if (key === "custom") continue;
    if (preset.length === sizes.length && preset.every((v, i) => v === sizes[i])) {
      return key;
    }
  }
  return "custom";
}

interface IndexingSettingsProps {
  projectId: string;
}

export function IndexingSettings({ projectId }: IndexingSettingsProps) {
  const { data: config, dataUpdatedAt } = useIndexingConfig(projectId);
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border">
      <button
        onClick={() => setOpen(!open)}
        className="hover:bg-muted/50 flex w-full items-center gap-2 px-3 py-2 text-sm font-medium transition-colors"
      >
        <Settings2 className="text-muted-foreground h-4 w-4" />
        <span>Indexing Settings</span>
        {open ? (
          <ChevronDown className="text-muted-foreground ml-auto h-4 w-4" />
        ) : (
          <ChevronRight className="text-muted-foreground ml-auto h-4 w-4" />
        )}
      </button>

      {open && (
        <IndexingSettingsForm key={dataUpdatedAt} projectId={projectId} config={config} />
      )}
    </div>
  );
}

function IndexingSettingsForm({
  projectId,
  config,
}: {
  projectId: string;
  config: IndexingConfig | undefined;
}) {
  const queryClient = useQueryClient();
  const updateConfig = useUpdateIndexingConfig();

  const initialSizes = config?.chunk_sizes ?? [2048, 512, 256];
  const [preset, setPreset] = useState(() => presetFromSizes(initialSizes));
  const [customSizes, setCustomSizes] = useState(() => [...initialSizes]);
  const [conversionMethod, setConversionMethod] = useState<"marker" | "direct">(
    () => config?.conversion_method ?? "marker"
  );
  const [dirty, setDirty] = useState(false);

  const handlePresetChange = (value: string) => {
    setPreset(value);
    if (value !== "custom") {
      setCustomSizes([...CHUNK_PRESETS[value]]);
    }
    setDirty(true);
  };

  const handleSizeChange = (index: number, value: string) => {
    const num = parseInt(value, 10);
    if (isNaN(num) || num <= 0) return;
    setCustomSizes((prev) => {
      const next = [...prev];
      next[index] = num;
      return next;
    });
    setPreset("custom");
    setDirty(true);
  };

  const handleConversionChange = (value: string) => {
    setConversionMethod(value as "marker" | "direct");
    setDirty(true);
  };

  const handleSave = async () => {
    const sizes = preset === "custom" ? customSizes : CHUNK_PRESETS[preset];
    try {
      await updateConfig.mutateAsync({
        projectId,
        data: {
          chunk_sizes: sizes,
          conversion_method: conversionMethod,
        },
      });
      setDirty(false);
      // Invalidate docs so UI reflects index deletion (all docs become unindexed)
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents("project", projectId),
      });
      toast.success("Indexing settings saved");
    } catch {
      toast.error("Failed to save settings");
    }
  };

  return (
    <div className="space-y-3 border-t px-3 pt-2 pb-3">
      <div className="space-y-1.5">
        <Label className="text-xs">Chunk Size Preset</Label>
        <Select value={preset} onValueChange={handlePresetChange}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(PRESET_LABELS).map(([key, label]) => (
              <SelectItem key={key} value={key} className="text-xs">
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {preset === "custom" && (
        <div className="grid grid-cols-3 gap-2">
          {["Large", "Medium", "Small"].map((label, i) => (
            <div key={label} className="space-y-1">
              <Label className="text-muted-foreground text-[10px]">{label}</Label>
              <Input
                type="number"
                className="h-7 text-xs"
                value={customSizes[i] ?? ""}
                onChange={(e) => handleSizeChange(i, e.target.value)}
                min={64}
                step={64}
              />
            </div>
          ))}
        </div>
      )}

      <div className="space-y-1.5">
        <Label className="text-xs">PDF Conversion</Label>
        <Select value={conversionMethod} onValueChange={handleConversionChange}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="marker" className="text-xs">
              Marker-PDF (better quality)
            </SelectItem>
            <SelectItem value="direct" className="text-xs">
              Direct PDF (faster, lower quality)
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      {dirty && (
        <Button
          size="sm"
          className="h-7 w-full text-xs"
          onClick={handleSave}
          disabled={updateConfig.isPending}
        >
          {updateConfig.isPending ? "Saving..." : "Save Settings"}
        </Button>
      )}
    </div>
  );
}
