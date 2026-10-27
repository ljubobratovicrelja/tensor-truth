import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { getModelReasoningLevels } from "@/api/providers";
import type { ModelsResponse } from "@/api/types";

const COOKIE_NAME = "thinking_preference";

function readThinkingCookie(): string {
  const match = document.cookie.match(/(?:^|; )thinking_preference=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : "auto";
}

function writeThinkingCookie(value: string): void {
  document.cookie = `${COOKIE_NAME}=${encodeURIComponent(value)}; max-age=${60 * 60 * 24 * 365}; path=/`;
}

interface ThinkingSupport {
  thinking: boolean;
  /** Reasoning-effort values the model's endpoint actually accepts. */
  levels: string[];
}

interface UseThinkingOptions {
  modelsData: ModelsResponse | undefined;
  effectiveModel: string;
}

interface UseThinkingReturn {
  thinking: string;
  thinkingSupport: ThinkingSupport;
  /** Call when model changes to auto-reset thinking if incompatible. */
  handleModelChange: (model: string) => void;
  /** Set thinking value and get the corresponding param value for session params. */
  setThinking: (value: string) => void;
  /** Convert current thinking UI value to a session param value. */
  thinkingParam: boolean | string | undefined;
}

/** Compute the session-param value for a given thinking UI string. */
export function thinkingToParam(value: string): boolean | string | undefined {
  if (value === "auto") return undefined;
  if (value === "off") return false;
  if (value === "on") return true;
  return value; // "low" | "medium" | "high"
}

/** Derive the thinking UI string from a session param value. */
export function paramToThinking(value: unknown): string {
  if (value === undefined || value === null) return "auto";
  if (value === false) return "off";
  if (value === true) return "on";
  return value as string;
}

function getThinkingSupport(
  modelsData: ModelsResponse | undefined,
  effectiveModel: string
): { thinking: boolean; staticLevels: string[] } {
  if (!modelsData?.models?.length) return { thinking: false, staticLevels: [] };
  const model = modelsData.models.find((m) => m.name === effectiveModel);
  const caps = model?.capabilities ?? [];
  return {
    thinking: caps.includes("thinking"),
    // Ollama models report level support statically; OpenAI-compatible models
    // are probed (see useReasoningLevels) because the endpoint decides.
    staticLevels: caps.includes("thinking_levels")
      ? ["low", "medium", "high"]
      : (model?.reasoning_levels ?? []),
  };
}

/**
 * Ask the backend which reasoning levels the selected model accepts.
 *
 * The backend pings the endpoint once per candidate level and caches the
 * answer; this hook only runs for OpenAI-compatible models that advertise
 * thinking support and have not been probed already.
 */
export function useReasoningLevels(
  modelsData: ModelsResponse | undefined,
  effectiveModel: string
): string[] {
  const model = modelsData?.models?.find((m) => m.name === effectiveModel);
  const staticLevels = getThinkingSupport(modelsData, effectiveModel).staticLevels;
  const canProbe =
    !!model &&
    model.provider_type === "openai_compatible" &&
    (model.capabilities ?? []).includes("thinking") &&
    staticLevels.length === 0;

  const { data } = useQuery({
    queryKey: ["reasoning-levels", model?.provider_id ?? "", effectiveModel],
    queryFn: () => getModelReasoningLevels(model!.provider_id, effectiveModel, false),
    enabled: canProbe,
    staleTime: 60 * 60 * 1000,
    retry: false,
  });

  if (staticLevels.length > 0) return staticLevels;
  return data?.levels ?? [];
}

/**
 * Hook for managing thinking state in pre-session pages (WelcomePage, ProjectViewPage).
 * Returns thinking state, support detection, and handlers.
 */
export function useThinking({
  modelsData,
  effectiveModel,
}: UseThinkingOptions): UseThinkingReturn {
  const [thinking, setThinkingState] = useState(() => readThinkingCookie());

  const probedLevels = useReasoningLevels(modelsData, effectiveModel);
  const thinkingSupport = useMemo<ThinkingSupport>(() => {
    const base = getThinkingSupport(modelsData, effectiveModel);
    return { thinking: base.thinking, levels: probedLevels };
  }, [modelsData, effectiveModel, probedLevels]);

  const handleModelChange = useCallback(
    (model: string) => {
      if (thinking === "auto") return;
      const support = getThinkingSupport(modelsData, model);
      if (
        !support.thinking ||
        (support.staticLevels.length === 0 &&
          ["minimal", "low", "medium", "high"].includes(thinking))
      ) {
        setThinkingState("auto");
        writeThinkingCookie("auto");
      }
    },
    [thinking, modelsData]
  );

  const setThinking = useCallback((value: string) => {
    setThinkingState(value);
    writeThinkingCookie(value);
  }, []);

  const thinkingParam = useMemo(() => thinkingToParam(thinking), [thinking]);

  return { thinking, thinkingSupport, handleModelChange, setThinking, thinkingParam };
}

/**
 * Compute thinking support for an active session (ChatInput).
 * For active sessions, thinking state lives in session params, not local state.
 */
export function useThinkingSupport(
  modelsData: ModelsResponse | undefined,
  effectiveModel: string
): ThinkingSupport {
  const probedLevels = useReasoningLevels(modelsData, effectiveModel);
  return useMemo(() => {
    const base = getThinkingSupport(modelsData, effectiveModel);
    return { thinking: base.thinking, levels: probedLevels };
  }, [modelsData, effectiveModel, probedLevels]);
}
