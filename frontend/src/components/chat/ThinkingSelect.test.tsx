import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThinkingSelect, formatThinkingValue } from "./ThinkingSelect";

describe("ThinkingSelect", () => {
  it("labels the trigger with the selected level", () => {
    render(<ThinkingSelect value="medium" onValueChange={() => {}} />);
    expect(screen.getByText("Think: Medium")).toBeInTheDocument();
  });

  it("labels a detected minimal level", () => {
    render(
      <ThinkingSelect value="minimal" onValueChange={() => {}} levels={["minimal"]} />
    );
    expect(screen.getByText("Think: Minimal")).toBeInTheDocument();
  });

  it("formats detected levels for display", () => {
    expect(formatThinkingValue("minimal")).toBe("Minimal");
    expect(formatThinkingValue("high")).toBe("High");
    expect(formatThinkingValue("on")).toBe("On");
    expect(formatThinkingValue("auto")).toBe("Auto");
  });
});
