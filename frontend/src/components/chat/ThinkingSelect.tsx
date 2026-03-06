import { Brain } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";

interface ThinkingSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  supportsLevels?: boolean;
}

export function ThinkingSelect({
  value,
  onValueChange,
  disabled,
  supportsLevels,
}: ThinkingSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
        <Brain className="h-3.5 w-3.5" />
        <span className="text-xs">
          {value === "auto" ? "Think" : value.charAt(0).toUpperCase() + value.slice(1)}
        </span>
      </SelectTrigger>
      <SelectContent position="popper" side="top">
        <SelectItem value="auto">Auto</SelectItem>
        <SelectItem value="off">Off</SelectItem>
        <SelectItem value="on">On</SelectItem>
        {supportsLevels && (
          <>
            <SelectItem value="low">Low</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="high">High</SelectItem>
          </>
        )}
      </SelectContent>
    </Select>
  );
}
