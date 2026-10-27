import { describe, it, expect, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useReasoningLevels, useThinkingSupport } from "./useThinking";
import type { ModelsResponse } from "@/api/types";

const mockGetLevels = vi.fn();

vi.mock("@/api/providers", () => ({
  getModelReasoningLevels: (...args: unknown[]) => mockGetLevels(...args),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const openaiModel = {
  name: "openai/gpt-5.5",
  provider_id: "openrouter",
  provider_type: "openai_compatible",
  display_name: "GPT-5.5",
  size: 0,
  modified_at: "",
  capabilities: ["tools", "thinking"],
};

const ollamaModel = {
  name: "qwen3:8b",
  provider_id: "ollama",
  provider_type: "ollama",
  display_name: "qwen3:8b",
  size: 1,
  modified_at: "",
  capabilities: ["tools", "thinking", "thinking_levels"],
};

function models(...list: (typeof openaiModel)[]): ModelsResponse {
  return { models: list };
}

describe("useReasoningLevels", () => {
  it("uses statically declared levels for Ollama models without probing", async () => {
    const data = models(ollamaModel);
    const { result } = renderHook(() => useReasoningLevels(data, "qwen3:8b"), {
      wrapper,
    });

    expect(result.current).toEqual(["low", "medium", "high"]);
    expect(mockGetLevels).not.toHaveBeenCalled();
  });

  it("probes an OpenAI-compatible model and returns accepted levels", async () => {
    mockGetLevels.mockResolvedValue({
      model: "openai/gpt-5.5",
      provider_id: "openrouter",
      levels: ["minimal", "low", "medium", "high"],
      accepted: ["none", "minimal", "low", "medium", "high"],
      reasoning: true,
      cached: false,
    });

    const data = models(openaiModel);
    const { result } = renderHook(() => useReasoningLevels(data, "openai/gpt-5.5"), {
      wrapper,
    });

    await waitFor(() =>
      expect(result.current).toEqual(["minimal", "low", "medium", "high"])
    );
    expect(mockGetLevels).toHaveBeenCalledWith("openrouter", "openai/gpt-5.5", false);
  });

  it("skips probing for models without thinking capability", () => {
    const plain = { ...openaiModel, capabilities: ["tools"] };
    const { result } = renderHook(() => useReasoningLevels(models(plain), plain.name), {
      wrapper,
    });
    expect(result.current).toEqual([]);
  });

  it("exposes the levels through the thinking-support hook", async () => {
    mockGetLevels.mockResolvedValue({
      model: "openai/gpt-5.5",
      provider_id: "openrouter",
      levels: ["low", "high"],
      accepted: ["low", "high"],
      reasoning: true,
      cached: true,
    });

    const { result } = renderHook(
      () => useThinkingSupport(models(openaiModel), "openai/gpt-5.5"),
      { wrapper }
    );

    await waitFor(() =>
      expect(result.current).toEqual({ thinking: true, levels: ["low", "high"] })
    );
  });
});
