import { describe, it, expect } from "vitest";
import { splitMarkdown, EMPTY_RESULT } from "../markdownSplitter";

describe("markdownSplitter", () => {
  it("returns empty result for empty input", () => {
    expect(splitMarkdown("")).toEqual(EMPTY_RESULT);
  });

  it("single paragraph → 0 stable, 1 unstable", () => {
    const result = splitMarkdown("Hello world");
    expect(result.stableBlocks).toHaveLength(0);
    expect(result.unstableBlock).toContain("Hello world");
    expect(result.isIncompleteCodeFence).toBe(false);
  });

  it("two paragraphs → 1 stable, 1 unstable", () => {
    const result = splitMarkdown("First paragraph\n\nSecond paragraph");
    expect(result.stableBlocks).toHaveLength(1);
    expect(result.stableBlocks[0]).toContain("First paragraph");
    expect(result.unstableBlock).toContain("Second paragraph");
  });

  it("complete code block followed by paragraph → code stable, paragraph unstable", () => {
    const content = "```js\nconsole.log('hi');\n```\n\nAfter code";
    const result = splitMarkdown(content);
    expect(result.stableBlocks).toHaveLength(1);
    expect(result.stableBlocks[0]).toContain("console.log");
    expect(result.unstableBlock).toContain("After code");
    expect(result.isIncompleteCodeFence).toBe(false);
  });

  it("incomplete code block → isIncompleteCodeFence: true", () => {
    const content = "```python\nprint('hello')\n# more code";
    const result = splitMarkdown(content);
    expect(result.isIncompleteCodeFence).toBe(true);
  });

  it("complete code block → isIncompleteCodeFence: false", () => {
    const content = "```python\nprint('hello')\n```";
    const result = splitMarkdown(content);
    expect(result.isIncompleteCodeFence).toBe(false);
  });

  it("partial table renders in unstable block", () => {
    const content = "# Header\n\n| Col1 | Col2 |\n| --- | --- |\n| a | b |";
    const result = splitMarkdown(content);
    expect(result.stableBlocks).toHaveLength(1);
    expect(result.stableBlocks[0]).toContain("Header");
    // The table should be in the unstable block
    expect(result.unstableBlock).toContain("Col1");
    expect(result.unstableBlock).toContain("Col2");
  });

  it("nested list → single MDAST node in unstable", () => {
    const content = "- item 1\n  - sub item\n- item 2";
    const result = splitMarkdown(content);
    // A single list is one MDAST node
    expect(result.stableBlocks).toHaveLength(0);
    expect(result.unstableBlock).toContain("item 1");
    expect(result.unstableBlock).toContain("sub item");
    expect(result.unstableBlock).toContain("item 2");
  });

  it("mixed content: heading + paragraph + code + list", () => {
    const content = [
      "# Title",
      "",
      "Some text here.",
      "",
      "```js",
      "code()",
      "```",
      "",
      "- list item",
    ].join("\n");
    const result = splitMarkdown(content);
    // heading, paragraph, code block are stable; list is unstable
    expect(result.stableBlocks).toHaveLength(3);
    expect(result.stableBlocks[0]).toContain("Title");
    expect(result.stableBlocks[1]).toContain("Some text");
    expect(result.stableBlocks[2]).toContain("code()");
    expect(result.unstableBlock).toContain("list item");
  });

  it("heading alone → 0 stable, 1 unstable", () => {
    const result = splitMarkdown("# Hello");
    expect(result.stableBlocks).toHaveLength(0);
    expect(result.unstableBlock).toContain("Hello");
  });
});
