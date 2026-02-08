import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ToolSteps } from "./ToolSteps";
import type { ToolStepWithStatus } from "./ToolSteps";

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
});
