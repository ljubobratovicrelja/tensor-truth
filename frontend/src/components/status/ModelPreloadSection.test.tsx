import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelPreloadSection } from "./ModelPreloadSection";
import type { ModelLoadState, ModelPreloadStatusResponse } from "@/api/system";

function state(overrides: Partial<ModelLoadState> = {}): ModelLoadState {
  return {
    kind: "embedder",
    state: "loading",
    model_name: "BAAI/bge-m3",
    device: "cuda",
    step: "loading weights",
    started_at: 100,
    finished_at: null,
    elapsed_seconds: 1.2,
    error: null,
    ...overrides,
  };
}

function status(
  embedder: Partial<ModelLoadState>,
  reranker: Partial<ModelLoadState>
): ModelPreloadStatusResponse {
  return {
    embedder: state({ kind: "embedder", ...embedder }),
    reranker: state({ kind: "reranker", ...reranker }),
    any_loading: false,
    all_ready: false,
  };
}

describe("ModelPreloadSection", () => {
  it("shows spinner and current step while loading", () => {
    const data = status(
      { state: "loading", step: "loading weights", elapsed_seconds: 2.4 },
      {
        state: "loading",
        step: "resolving device",
        model_name: "BAAI/bge-reranker-v2-m3",
      }
    );
    data.any_loading = true;

    render(<ModelPreloadSection status={data} />);

    expect(screen.getByText("Warming up")).toBeInTheDocument();
    expect(screen.getByText("Embedding model")).toBeInTheDocument();
    expect(screen.getByText("Reranker")).toBeInTheDocument();
    expect(screen.getByText("loading weights · 2.4s")).toBeInTheDocument();
    expect(screen.getByText("bge-m3")).toBeInTheDocument();
    expect(screen.getAllByTestId("model-state-spinner")).toHaveLength(2);
    expect(screen.getAllByText("CUDA").length).toBe(2);
  });

  it("shows model name, device and load time when ready", () => {
    const data = status(
      { state: "ready", step: "ready", elapsed_seconds: 3.7, finished_at: 103.7 },
      {
        state: "ready",
        step: "ready",
        model_name: "BAAI/bge-reranker-v2-m3",
        elapsed_seconds: 1.4,
        finished_at: 104.0,
      }
    );
    data.all_ready = true;

    render(<ModelPreloadSection status={data} />);

    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("bge-m3")).toBeInTheDocument();
    expect(screen.getByText("bge-reranker-v2-m3")).toBeInTheDocument();
    expect(screen.getByText("loaded in 3.7s")).toBeInTheDocument();
    expect(screen.getByText("loaded in 1.4s")).toBeInTheDocument();
    expect(screen.queryByTestId("model-state-spinner")).not.toBeInTheDocument();
  });

  it("shows a red error state with the failure message", () => {
    const data = status(
      { state: "error", step: "failed", error: "CUDA out of memory" },
      { state: "ready", elapsed_seconds: 2.0 }
    );

    render(<ModelPreloadSection status={data} />);

    expect(screen.getByText("Load failed")).toBeInTheDocument();
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("CUDA out of memory");
    expect(alert.className).toContain("text-destructive");
  });

  it("shows the idle state before preload is triggered", () => {
    const data = status({ state: "idle", step: "" }, { state: "idle", step: "" });

    render(<ModelPreloadSection status={data} />);

    expect(screen.getByText("Not loaded")).toBeInTheDocument();
    expect(screen.getAllByText("Not loaded yet")).toHaveLength(2);
  });

  it("renders a skeleton while the first status request is in flight", () => {
    const { container } = render(<ModelPreloadSection isLoading />);

    expect(container.querySelectorAll(".animate-pulse")).toHaveLength(2);
  });
});
