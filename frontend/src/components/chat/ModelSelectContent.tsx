import { Cpu, MessageSquare } from "lucide-react";
import {
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
} from "@/components/ui/select";
import type { ModelInfo } from "@/api/types";

interface ModelSelectContentProps {
  models: ModelInfo[];
  isLoading?: boolean;
  /**
   * Extra items rendered before the grouped model list — e.g. a "Default" entry
   * or greyed-out "not installed" entries.
   */
  extraItems?: React.ReactNode;
  position?: "popper" | "item-aligned";
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export function ModelSelectContent({
  models,
  isLoading,
  extraItems,
  position = "item-aligned",
  side,
  className,
}: ModelSelectContentProps) {
  const sorted = models.slice().sort((a, b) => a.name.localeCompare(b.name));
  const agentic = sorted.filter((m) => m.capabilities?.includes("tools"));
  const standard = sorted.filter((m) => !m.capabilities?.includes("tools"));

  return (
    <SelectContent
      position={position}
      side={side}
      className={className}
      hideScrollButtons
    >
      {extraItems}
      {isLoading ? (
        <SelectItem value="loading" disabled>
          Loading...
        </SelectItem>
      ) : (
        <>
          {agentic.length > 0 && (
            <SelectGroup>
              <SelectLabel className="mb-1 flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium">
                <Cpu className="h-3 w-3" />
                <span>Agentic</span>
              </SelectLabel>
              {agentic.map((model) => (
                <SelectItem key={model.name} value={model.name}>
                  {model.name}
                </SelectItem>
              ))}
            </SelectGroup>
          )}
          {agentic.length > 0 && standard.length > 0 && <SelectSeparator />}
          {standard.length > 0 && (
            <SelectGroup>
              <SelectLabel className="mb-1 flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium">
                <MessageSquare className="h-3 w-3" />
                <span>Standard</span>
              </SelectLabel>
              {standard.map((model) => (
                <SelectItem key={model.name} value={model.name}>
                  {model.name}
                </SelectItem>
              ))}
            </SelectGroup>
          )}
          {models.length === 0 && (
            <SelectItem value="__no_models__" disabled>
              No models available
            </SelectItem>
          )}
        </>
      )}
    </SelectContent>
  );
}
