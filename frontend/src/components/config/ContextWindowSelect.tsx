import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CONTEXT_WINDOW_OPTIONS, formatContextWindow } from "@/lib/contextWindow";

interface ContextWindowSelectProps {
  /** Currently selected context window, in tokens. */
  value: number;
  onValueChange: (value: number) => void;
  triggerClassName?: string;
  itemClassName?: string;
  disabled?: boolean;
}

/**
 * Context-window dropdown shared by the config, session and project panels.
 *
 * Renders every option from {@link CONTEXT_WINDOW_OPTIONS} so the three
 * entry points cannot drift apart.
 */
export function ContextWindowSelect({
  value,
  onValueChange,
  triggerClassName,
  itemClassName,
  disabled,
}: ContextWindowSelectProps) {
  return (
    <Select
      value={String(value)}
      onValueChange={(v) => onValueChange(Number(v))}
      disabled={disabled}
    >
      <SelectTrigger className={triggerClassName}>
        <SelectValue placeholder="Select context window" />
      </SelectTrigger>
      <SelectContent>
        {CONTEXT_WINDOW_OPTIONS.map((size) => (
          <SelectItem key={size} value={String(size)} className={itemClassName}>
            {formatContextWindow(size)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
