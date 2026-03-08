import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  loadLlamaCppModel,
  unloadLlamaCppModel,
  loadOllamaModel,
  unloadOllamaModel,
} from "@/api/system";

export function useModelActions() {
  const [actionsInFlight, setActionsInFlight] = useState<Set<string>>(new Set());
  const queryClient = useQueryClient();

  const makeKey = (providerId: string, modelName: string) =>
    `${providerId}::${modelName}`;

  const handleLoadModel = useCallback(
    async (providerId: string, providerType: string, modelName: string) => {
      const key = makeKey(providerId, modelName);
      setActionsInFlight((prev) => new Set(prev).add(key));
      try {
        if (providerType === "llama_cpp") {
          await loadLlamaCppModel(modelName, providerId);
        } else if (providerType === "ollama") {
          await loadOllamaModel(modelName);
        }
        await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
      } catch (e) {
        console.error(`Failed to load model ${modelName}:`, e);
      } finally {
        setActionsInFlight((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
      }
    },
    [queryClient]
  );

  const handleUnloadModel = useCallback(
    async (providerId: string, providerType: string, modelName: string) => {
      const key = makeKey(providerId, modelName);
      setActionsInFlight((prev) => new Set(prev).add(key));
      try {
        if (providerType === "llama_cpp") {
          await unloadLlamaCppModel(modelName, providerId);
        } else if (providerType === "ollama") {
          await unloadOllamaModel(modelName);
        }
        await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
      } catch (e) {
        console.error(`Failed to unload model ${modelName}:`, e);
      } finally {
        setActionsInFlight((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
      }
    },
    [queryClient]
  );

  return { actionsInFlight, handleLoadModel, handleUnloadModel };
}
