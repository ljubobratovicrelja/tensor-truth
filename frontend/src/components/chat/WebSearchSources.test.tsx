import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { WebSearchSources } from "./WebSearchSources";
import type { WebSearchSource } from "@/api/types";

describe("WebSearchSources", () => {
  const mockSources: WebSearchSource[] = [
    {
      url: "https://example.com/article1",
      title: "Example Article 1",
      status: "success",
      snippet: "This is a preview snippet for article 1.",
    },
    {
      url: "https://example.com/article2",
      title: "Example Article 2",
      status: "failed",
      error: "Connection timeout",
    },
    {
      url: "https://example.com/article3",
      title: "Example Article 3",
      status: "skipped",
      error: "Duplicate content",
    },
  ];

  it("returns null when sources array is empty", () => {
    const { container } = render(<WebSearchSources sources={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows collapsed summary with total count", () => {
    render(<WebSearchSources sources={mockSources} />);

    expect(screen.getByText("Web Sources (3)")).toBeInTheDocument();
  });

  it("shows summary with fetched, failed, and skipped counts", () => {
    render(<WebSearchSources sources={mockSources} />);

    expect(screen.getByText(/1 fetched, 1 failed, 1 skipped/)).toBeInTheDocument();
  });

  it("starts in collapsed state", () => {
    render(<WebSearchSources sources={mockSources} />);

    // Source titles should not be visible when collapsed
    // (they're in a max-h-0 overflow-hidden container)
    const sourceList = screen.getByRole("button", { name: /Web Sources/ }).parentElement;
    const expandableDiv = sourceList?.querySelector(".max-h-0");
    expect(expandableDiv).toBeTruthy();
  });

  it("expands to show individual sources on click", async () => {
    const user = userEvent.setup();
    render(<WebSearchSources sources={mockSources} />);

    // Click to expand
    await user.click(screen.getByRole("button", { name: /Web Sources/ }));

    // Source titles should now be visible
    expect(screen.getByText("Example Article 1")).toBeInTheDocument();
    expect(screen.getByText("Example Article 2")).toBeInTheDocument();
    expect(screen.getByText("Example Article 3")).toBeInTheDocument();
  });

  it("shows correct status badges for each source", async () => {
    const user = userEvent.setup();
    render(<WebSearchSources sources={mockSources} />);

    await user.click(screen.getByRole("button", { name: /Web Sources/ }));

    expect(screen.getByText("Fetched")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Skipped")).toBeInTheDocument();
  });

  it("renders external links with correct attributes", async () => {
    const user = userEvent.setup();
    render(<WebSearchSources sources={mockSources} />);

    await user.click(screen.getByRole("button", { name: /Web Sources/ }));

    const links = screen.getAllByRole("link");
    expect(links[0]).toHaveAttribute("href", "https://example.com/article1");
    expect(links[0]).toHaveAttribute("target", "_blank");
    expect(links[0]).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("can expand individual source to show snippet", async () => {
    const user = userEvent.setup();
    render(<WebSearchSources sources={mockSources} />);

    // Expand the main list
    await user.click(screen.getByRole("button", { name: /Web Sources/ }));

    // Find and click the first source item button (not the link)
    const sourceButtons = screen
      .getAllByRole("button")
      .filter((btn) => btn.textContent?.includes("Example Article"));
    await user.click(sourceButtons[0]);

    // Snippet should now be visible
    expect(
      screen.getByText("This is a preview snippet for article 1.")
    ).toBeInTheDocument();
  });

  it("can expand individual source to show error", async () => {
    const user = userEvent.setup();
    render(<WebSearchSources sources={mockSources} />);

    // Expand the main list
    await user.click(screen.getByRole("button", { name: /Web Sources/ }));

    // Find and click the failed source button
    const sourceButtons = screen
      .getAllByRole("button")
      .filter((btn) => btn.textContent?.includes("Example Article 2"));
    await user.click(sourceButtons[0]);

    // Error should now be visible
    expect(screen.getByText("Connection timeout")).toBeInTheDocument();
  });

  it("handles sources with only successes", () => {
    const successOnlySources: WebSearchSource[] = [
      { url: "https://a.com", title: "A", status: "success" },
      { url: "https://b.com", title: "B", status: "success" },
    ];
    render(<WebSearchSources sources={successOnlySources} />);

    expect(screen.getByText("Web Sources (2)")).toBeInTheDocument();
    expect(screen.getByText(/2 fetched/)).toBeInTheDocument();
  });
});
