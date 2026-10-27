import { Brain } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";

interface ThinkingSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  /** Reasoning-effort values this model's endpoint accepted (probed). */
  levels?: string[];
}

const LEVEL_LABELS: Record<string, string> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
};

export function formatThinkingValue(value: string): string {
  return LEVEL_LABELS[value] ?? value.charAt(0).toUpperCase() + value.slice(1);
}

export function ThinkingSelect({
  value,
  onValueChange,
  disabled,
  levels = [],
}: ThinkingSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
        <Brain className="h-3.5 w-3.5" />
        <span className="text-xs">Think: {formatThinkingValue(value)}</span>
      </SelectTrigger>
      <SelectContent position="popper" side="top">
        <SelectItem value="auto">Auto</SelectItem>
        <SelectItem value="off">Off</SelectItem>
        <SelectItem value="on">On</SelectItem>
        {levels.map((level) => (
          <SelectItem key={level} value={level}>
            {formatThinkingValue(level)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
