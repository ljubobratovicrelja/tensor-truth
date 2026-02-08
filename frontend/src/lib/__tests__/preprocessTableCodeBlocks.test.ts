import { describe, test, expect } from "vitest";
import { preprocessTableCodeBlocks } from "../utils";

describe("preprocessTableCodeBlocks", () => {
  test("converts code fences with <br> in table cells to inline code with %%BR%%", () => {
    const input =
      "| Layer | Code |\n| --- | --- |\n| Conv | ```python<br>nn.Conv2d(3, 64)<br>nn.ReLU()``` |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toContain("`nn.Conv2d(3, 64)%%BR%%nn.ReLU()`");
    expect(result).not.toContain("```");
  });

  test("does not modify non-table content", () => {
    const input = "```python\nprint('hello')\n```";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toBe(input);
  });

  test("does not modify tables without code fences", () => {
    const input = "| Name | Value |\n| --- | --- |\n| foo | bar |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toBe(input);
  });

  test("handles multiple cells with code fences", () => {
    const input = "| A | B |\n| --- | --- |\n| ```js<br>x()``` | ```py<br>y()``` |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toContain("`x()`");
    expect(result).toContain("`y()`");
    expect(result).not.toContain("```");
  });

  test("handles <br/> and <BR> variants", () => {
    const input = "| Code |\n| --- |\n| ```python<br/>a = 1<BR>b = 2``` |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toContain("`a = 1%%BR%%b = 2`");
  });

  test("handles self-closing <br /> with space", () => {
    const input = "| Code |\n| --- |\n| ```js<br />foo()<br />bar()``` |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toContain("`foo()%%BR%%bar()`");
  });

  test("preserves table structure", () => {
    const input = "| Header |\n| --- |\n| ```python<br>code``` |";
    const result = preprocessTableCodeBlocks(input);
    expect(result).toMatch(/^\|.*\|$/m);
  });
});
