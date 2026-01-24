import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentProgress } from "./AgentProgress";
import type { StreamAgentProgress } from "@/api/types";

describe("AgentProgress", () => {
  it("returns null when progress is null", () => {
    const { container } = render(<AgentProgress progress={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders searching phase with correct message", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "searching",
      message: "Searching DuckDuckGo",
      search_hits: 10,
    };
    render(<AgentProgress progress={progress} />);

    expect(screen.getByText("Searching DuckDuckGo - 10 results")).toBeInTheDocument();
  });

  it("renders fetching phase with progress counter", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "fetching",
      message: "Fetching pages",
      pages_target: 5,
      pages_fetched: 3,
    };
    render(<AgentProgress progress={progress} />);

    expect(screen.getByText("Fetching pages (3/5)")).toBeInTheDocument();
  });

  it("renders summarizing phase with message", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "summarizing",
      message: "Generating summary with gpt-4",
    };
    render(<AgentProgress progress={progress} />);

    expect(screen.getByText("Generating summary with gpt-4")).toBeInTheDocument();
  });

  it("renders complete phase", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "complete",
      message: "Search complete",
    };
    render(<AgentProgress progress={progress} />);

    expect(screen.getByText("Search complete")).toBeInTheDocument();
  });

  it("applies animation class for searching phase", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "searching",
      message: "Searching",
    };
    const { container } = render(<AgentProgress progress={progress} />);

    const icon = container.querySelector("svg");
    expect(icon?.getAttribute("class")).toContain("animate-bounce");
  });

  it("applies animation class for fetching phase", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "fetching",
      message: "Fetching",
      pages_target: 5,
      pages_fetched: 2,
    };
    const { container } = render(<AgentProgress progress={progress} />);

    const icon = container.querySelector("svg");
    expect(icon?.getAttribute("class")).toContain("animate-pulse");
  });

  it("does not apply animation for complete phase", () => {
    const progress: StreamAgentProgress = {
      type: "agent_progress",
      agent: "web_search",
      phase: "complete",
      message: "Done",
    };
    const { container } = render(<AgentProgress progress={progress} />);

    const icon = container.querySelector("svg");
    expect(icon?.getAttribute("class")).not.toContain("animate-");
  });
});
