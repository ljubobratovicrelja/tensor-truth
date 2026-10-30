import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import { ElaborationTrigger } from "./ElaborationTrigger";

const RECT = { top: 100, left: 100, right: 300, bottom: 120, width: 200, height: 20 };

describe("ElaborationTrigger", () => {
  it("renders nothing without a selection rect", () => {
    const { container } = render(<ElaborationTrigger rect={null} onOpen={() => {}} />);
    expect(container.querySelector("[data-elaboration-trigger]")).toBeNull();
  });

  it("is hit-testable when idle and inert while the selection is being dragged", () => {
    const { container, rerender } = render(
      <ElaborationTrigger rect={RECT} onOpen={() => {}} />
    );
    const button = container.querySelector<HTMLElement>("[data-elaboration-trigger]")!;
    expect(button.className).not.toContain("pointer-events-none");

    rerender(<ElaborationTrigger rect={RECT} onOpen={() => {}} interactive={false} />);
    expect(
      container.querySelector<HTMLElement>("[data-elaboration-trigger]")!.className
    ).toContain("pointer-events-none");
  });

  it("keeps the selection alive by preventing mousedown and opens on click", () => {
    const onOpen = vi.fn();
    const { container } = render(<ElaborationTrigger rect={RECT} onOpen={onOpen} />);
    const button = container.querySelector<HTMLElement>("[data-elaboration-trigger]")!;

    const mousedown = new MouseEvent("mousedown", { bubbles: true, cancelable: true });
    button.dispatchEvent(mousedown);
    expect(mousedown.defaultPrevented).toBe(true);

    button.click();
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});
