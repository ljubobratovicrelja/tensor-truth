import { Cpu, MessageSquare, Server } from "lucide-react";
import {
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
} from "@/components/ui/select";
import type { ModelInfo } from "@/api/types";

/**
 * Encode a provider_id + model name into a single Select value string.
 * Format: "provider_id::model_name"
 */
export function encodeModelValue(providerId: string, modelName: string): string {
  return `${providerId}::${modelName}`;
}

/**
 * Decode a composite Select value back to { providerId, modelName }.
 * If the value has no "::" separator, assumes "ollama" as provider.
 */
export function decodeModelValue(value: string): {
  providerId: string;
  modelName: string;
} {
  const idx = value.indexOf("::");
  if (idx === -1) {
    return { providerId: "ollama", modelName: value };
  }
  return {
    providerId: value.slice(0, idx),
    modelName: value.slice(idx + 2),
  };
}

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
  // Group models by provider
  const providerIds = [...new Set(models.map((m) => m.provider_id || "ollama"))];
  const multiProvider = providerIds.length > 1;

  if (multiProvider) {
    // Multi-provider: group by provider, then agentic/standard within each
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
            {providerIds.map((pid, pidIdx) => {
              const providerModels = models
                .filter((m) => (m.provider_id || "ollama") === pid)
                .sort((a, b) => a.name.localeCompare(b.name));
              const agentic = providerModels.filter((m) =>
                m.capabilities?.includes("tools")
              );
              const standard = providerModels.filter(
                (m) => !m.capabilities?.includes("tools")
              );

              return (
                <div key={pid}>
                  {pidIdx > 0 && <SelectSeparator />}
                  <SelectGroup>
                    <SelectLabel className="mb-1 flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium">
                      <Server className="h-3 w-3" />
                      <span>{pid}</span>
                    </SelectLabel>
                    {agentic.map((model) => (
                      <SelectItem
                        key={encodeModelValue(pid, model.name)}
                        value={encodeModelValue(pid, model.name)}
                      >
                        <span className="flex items-center gap-1.5">
                          <Cpu className="h-3 w-3 opacity-50" />
                          {model.display_name || model.name}
                        </span>
                      </SelectItem>
                    ))}
                    {standard.map((model) => (
                      <SelectItem
                        key={encodeModelValue(pid, model.name)}
                        value={encodeModelValue(pid, model.name)}
                      >
                        {model.display_name || model.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </div>
              );
            })}
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

  // Single provider: use original agentic/standard grouping
  const sorted = models.slice().sort((a, b) => a.name.localeCompare(b.name));
  const agentic = sorted.filter((m) => m.capabilities?.includes("tools"));
  const standard = sorted.filter((m) => !m.capabilities?.includes("tools"));
  const pid = providerIds[0] || "ollama";

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
                <SelectItem
                  key={encodeModelValue(pid, model.name)}
                  value={encodeModelValue(pid, model.name)}
                >
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
                <SelectItem
                  key={encodeModelValue(pid, model.name)}
                  value={encodeModelValue(pid, model.name)}
                >
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
