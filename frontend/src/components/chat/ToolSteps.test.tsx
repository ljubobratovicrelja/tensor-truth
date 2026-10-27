import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ToolSteps } from "./ToolSteps";
import type { ToolStepWithStatus } from "./ToolSteps";
import type { StreamConfirmationRequest } from "@/api/types";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

describe("ToolSteps", () => {
  it("renders nothing when steps array is empty", () => {
    const { container } = render(<ToolSteps steps={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows correct step count in header", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: { q: "test" },
        output: "result",
        is_error: false,
        status: "completed",
      },
      {
        tool: "fetch",
        params: { url: "https://example.com" },
        output: "page content",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} />);

    expect(screen.getByText("Steps (2)")).toBeInTheDocument();
  });

  it("shows completed and failed counts in summary", () => {
    const steps: ToolStepWithStatus[] = [
      { tool: "search", params: {}, output: "ok", is_error: false, status: "completed" },
      { tool: "fetch", params: {}, output: "error", is_error: true, status: "failed" },
      { tool: "read", params: {}, output: "", is_error: false, status: "calling" },
    ];
    render(<ToolSteps steps={steps} />);

    expect(screen.getByText(/1 completed/)).toBeInTheDocument();
    expect(screen.getByText(/1 failed/)).toBeInTheDocument();
    expect(screen.getByText(/1 running/)).toBeInTheDocument();
  });

  it("shows Calling badge for in-progress step", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: { q: "test" },
        output: "",
        is_error: false,
        status: "calling",
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    expect(screen.getByText("Calling")).toBeInTheDocument();
  });

  it("shows Done badge for completed step", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: {},
        output: "results",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    expect(screen.getByText("Done")).toBeInTheDocument();
  });

  it("shows Failed badge for failed step", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: {},
        output: "error msg",
        is_error: true,
        status: "failed",
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("expands step to show output on click", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: {},
        output: "detailed output here",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    // Output should not be visible initially
    expect(screen.queryByText("detailed output here")).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(screen.getByText("search"));

    expect(screen.getByText("detailed output here")).toBeInTheDocument();
  });

  it("displays tool name for each step", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "resolve-library-id",
        params: {},
        output: "ok",
        is_error: false,
        status: "completed",
      },
      {
        tool: "get-library-docs",
        params: {},
        output: "docs",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    expect(screen.getByText("resolve-library-id")).toBeInTheDocument();
    expect(screen.getByText("get-library-docs")).toBeInTheDocument();
  });

  it("is collapsed by default when defaultOpen is false", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: {},
        output: "result",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} />);

    // The step card content should be hidden (inside collapsed div)
    expect(screen.getByText("Steps (1)")).toBeInTheDocument();
    // Tool name should exist but be inside a max-h-0 overflow-hidden container
    const toolName = screen.getByText("search");
    const collapsibleDiv = toolName.closest('[class*="max-h-0"]');
    expect(collapsibleDiv).not.toBeNull();
  });

  it("expands panel when header is clicked", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "search",
        params: {},
        output: "result",
        is_error: false,
        status: "completed",
      },
    ];
    render(<ToolSteps steps={steps} />);

    // Click the header to expand
    fireEvent.click(screen.getByText("Steps (1)"));

    // Now tool name should be inside max-h-[2000px] container
    const toolName = screen.getByText("search");
    const collapsibleDiv = toolName.closest('[class*="max-h-"]');
    expect(collapsibleDiv?.className).toContain("max-h-[2000px]");
  });

  it("renders a live progress bar and stage line for a step with progress", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "build_index",
        params: { scope: "session" },
        output: "",
        is_error: false,
        tool_id: "build_index",
        status: "calling",
        progress: {
          percent: 35,
          message: "Converting 1/2: Attention_Is_All_You_Need.pdf",
          stage: "converting",
        },
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "35");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(bar).toHaveStyle({ width: "35%" });
    expect(
      screen.getByText("Converting 1/2: Attention_Is_All_You_Need.pdf")
    ).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
  });

  it("does not render a progress bar once the step is no longer running", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "build_index",
        params: {},
        output: "done",
        is_error: false,
        status: "completed",
        progress: { percent: 100, message: "Indexed 2 documents." },
      },
    ];
    render(<ToolSteps steps={steps} defaultOpen />);

    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("pairs same-tool confirmations to their own steps positionally", () => {
    const steps: ToolStepWithStatus[] = [
      {
        tool: "add_arxiv_paper",
        params: { arxiv_id: "2511.08544" },
        output: "added",
        is_error: false,
        status: "completed",
      },
      {
        tool: "add_arxiv_paper",
        params: { arxiv_id: "2301.08243" },
        output: "",
        is_error: false,
        status: "calling",
      },
    ];
    const confirmations: StreamConfirmationRequest[] = [
      {
        confirmation_id: "id-1",
        tool_name: "add_arxiv_paper",
        action_type: "add_document",
        title: "Add arXiv paper 2511.08544",
        summary: "",
        details: { arxiv_id: "2511.08544" },
      },
      {
        confirmation_id: "id-2",
        tool_name: "add_arxiv_paper",
        action_type: "add_document",
        title: "Add arXiv paper 2301.08243",
        summary: "",
        details: { arxiv_id: "2301.08243" },
      },
    ];

    const queryClient = new QueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <ToolSteps steps={steps} defaultOpen confirmationRequests={confirmations} />
      </QueryClientProvider>
    );

    // Only the still-running step owns a card, and it owns the second request.
    expect(screen.getAllByText("Approve")).toHaveLength(1);
    expect(screen.getByText("Add arXiv paper 2301.08243")).toBeInTheDocument();
    expect(screen.queryByText("Add arXiv paper 2511.08544")).not.toBeInTheDocument();
  });
});
