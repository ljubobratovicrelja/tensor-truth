import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";

import { CONTEXT_WINDOW_OPTIONS, formatContextWindow } from "@/lib/contextWindow";

import { ContextWindowSelect } from "./ContextWindowSelect";

// Radix Select relies on pointer capture + scrollIntoView, which jsdom lacks.
beforeAll(() => {
  if (!window.PointerEvent) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    window.PointerEvent = MouseEvent as any;
  }
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.setPointerCapture = () => {};
  Element.prototype.releasePointerCapture = () => {};
  Element.prototype.scrollIntoView = () => {};
});

function Harness({ initial }: { initial: number }) {
  const [value, setValue] = useState(initial);
  return (
    <div>
      <ContextWindowSelect value={value} onValueChange={setValue} />
      <output data-testid="selected">{value}</output>
    </div>
  );
}

function openDropdown() {
  fireEvent.click(screen.getByRole("combobox"));
}

describe("CONTEXT_WINDOW_OPTIONS", () => {
  it("keeps every previously selectable size", () => {
    for (const size of [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]) {
      expect(CONTEXT_WINDOW_OPTIONS).toContain(size);
    }
  });

  it("includes 512k and 1M token windows", () => {
    expect(CONTEXT_WINDOW_OPTIONS).toContain(524288);
    expect(CONTEXT_WINDOW_OPTIONS).toContain(1048576);
  });

  it("is sorted ascending", () => {
    const sorted = [...CONTEXT_WINDOW_OPTIONS].sort((a, b) => a - b);
    expect([...CONTEXT_WINDOW_OPTIONS]).toEqual(sorted);
  });

  it("formats large sizes without truncation", () => {
    expect(formatContextWindow(262144)).toBe("262,144");
    expect(formatContextWindow(524288)).toBe("524,288");
    expect(formatContextWindow(1048576)).toBe("1,048,576");
  });
});

describe("ContextWindowSelect", () => {
  it("renders every option, including 512k and 1M", () => {
    render(<Harness initial={8192} />);
    openDropdown();

    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(CONTEXT_WINDOW_OPTIONS.length);
    expect(screen.getByRole("option", { name: "524,288" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "1,048,576" })).toBeInTheDocument();
  });

  it.each([524288, 1048576])("selects %i tokens", (size) => {
    render(<Harness initial={8192} />);
    openDropdown();

    const option = screen.getByRole("option", {
      name: formatContextWindow(size),
    });
    fireEvent.pointerUp(option);
    fireEvent.click(option);

    expect(screen.getByTestId("selected")).toHaveTextContent(String(size));
  });
});
