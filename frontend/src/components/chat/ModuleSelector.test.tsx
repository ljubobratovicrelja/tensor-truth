import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";

const useModules = vi.fn();
const useConfig = vi.fn();
const useEmbeddingModels = vi.fn();

vi.mock("@/hooks", () => ({
  useModules: () => useModules(),
  useConfig: () => useConfig(),
  useEmbeddingModels: () => useEmbeddingModels(),
}));

import { ModuleSelector } from "./ModuleSelector";

const MODULES = {
  data: {
    modules: [
      {
        name: "papers_dl",
        display_name: "DL Papers",
        doc_type: "paper",
        sort_order: 1,
      },
    ],
  },
  isLoading: false,
};

function renderSelector(props: Partial<ComponentProps<typeof ModuleSelector>> = {}) {
  const onModulesChange = vi.fn();
  const onModuleModeChange = vi.fn();
  render(
    <ModuleSelector
      selectedModules={[]}
      onModulesChange={onModulesChange}
      moduleMode="auto"
      onModuleModeChange={onModuleModeChange}
      {...props}
    />
  );
  return { onModulesChange, onModuleModeChange };
}

async function openPanel(user: ReturnType<typeof userEvent.setup>) {
  // The popover trigger is the first button rendered by the selector.
  await user.click(screen.getAllByRole("button")[0]);
  return screen.findByText("Auto — let the agent pick");
}

describe("ModuleSelector module mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useModules.mockReturnValue(MODULES);
    useConfig.mockReturnValue({
      data: {
        rag: { default_embedding_model: "sentence-transformers/all-MiniLM-L6-v2" },
      },
    });
    useEmbeddingModels.mockReturnValue({
      data: {
        models: [
          {
            model_id: "sentence-transformers/all-MiniLM-L6-v2",
            modules: ["papers_dl"],
          },
        ],
      },
    });
  });

  it("defaults to auto mode when no mode is passed", async () => {
    const user = userEvent.setup();
    render(
      <ModuleSelector
        selectedModules={[]}
        onModulesChange={vi.fn()}
        onModuleModeChange={vi.fn()}
      />
    );
    await user.click(screen.getAllByRole("button")[0]);
    const row = (await screen.findByText("Auto — let the agent pick")).closest("button");
    expect(row?.querySelector('[role="checkbox"]')).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  it("emits manual as soon as a module is selected", async () => {
    const user = userEvent.setup();
    const { onModuleModeChange } = renderSelector({ moduleMode: "manual" });
    await openPanel(user);

    await user.click(screen.getByText("DL Papers"));

    expect(onModuleModeChange).toHaveBeenCalledWith("manual");
  });

  it("emits auto when the selection is cleared", async () => {
    const user = userEvent.setup();
    const { onModuleModeChange, onModulesChange } = renderSelector({
      moduleMode: "manual",
      selectedModules: ["papers_dl"],
    });
    await openPanel(user);

    await user.click(screen.getByText("Clear all"));

    expect(onModuleModeChange).toHaveBeenCalledWith("auto");
    expect(onModulesChange).not.toHaveBeenCalled();
  });

  it("keeps module rows read-only in auto mode", async () => {
    const user = userEvent.setup();
    const { onModuleModeChange } = renderSelector({ moduleMode: "auto" });
    await openPanel(user);

    const row = screen.getByText("DL Papers").closest("button");
    expect(row).toBeDisabled();
    await user.click(screen.getByText("DL Papers"));
    expect(onModuleModeChange).not.toHaveBeenCalled();
  });
});
