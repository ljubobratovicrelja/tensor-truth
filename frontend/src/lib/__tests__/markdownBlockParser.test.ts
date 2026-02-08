import { describe, test, expect } from "vitest";
import {
  parseMarkdownBlocks,
  getAllContent,
  finalizeState,
  initialParserState,
  type ParserState,
} from "../markdownBlockParser";

/**
 * Helper to simulate streaming text character by character or in chunks.
 */
function streamText(text: string, chunkSize: number | "char" = "char"): ParserState {
  let state = { ...initialParserState };
  const size = chunkSize === "char" ? 1 : chunkSize;

  for (let i = 0; i < text.length; i += size) {
    const token = text.slice(i, i + size);
    state = parseMarkdownBlocks(state, token);
  }

  return state;
}

/**
 * Helper to stream and finalize.
 */
function streamAndFinalize(
  text: string,
  chunkSize: number | "char" = "char"
): ParserState {
  return finalizeState(streamText(text, chunkSize));
}

describe("markdownBlockParser", () => {
  describe("Header Detection", () => {
    test("detects h1 header", () => {
      const state = streamAndFinalize("# Title\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("header");
      expect(state.completedBlocks[0].content).toBe("# Title\n");
      expect(state.completedBlocks[0].isComplete).toBe(true);
    });

    test("detects h2 header", () => {
      const state = streamAndFinalize("## Subtitle\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("header");
    });

    test("detects h6 header", () => {
      const state = streamAndFinalize("###### Deep Header\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("header");
    });

    test("header without newline stays pending until finalized", () => {
      const state = streamText("# Title");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.pendingBuffer).toBe("# Title");

      const finalized = finalizeState(state);
      expect(finalized.completedBlocks).toHaveLength(1);
    });

    test("header split across tokens", () => {
      let state = { ...initialParserState };
      state = parseMarkdownBlocks(state, "#");
      state = parseMarkdownBlocks(state, " Title");
      state = parseMarkdownBlocks(state, "\n");

      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("header");
      expect(state.completedBlocks[0].content).toBe("# Title\n");
    });

    test("# in middle of text is not header", () => {
      const state = streamAndFinalize("Use # for comments\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
    });

    test("multiple headers in sequence", () => {
      const state = streamAndFinalize("# H1\n## H2\n### H3\n");
      expect(state.completedBlocks).toHaveLength(3);
      expect(state.completedBlocks[0].type).toBe("header");
      expect(state.completedBlocks[1].type).toBe("header");
      expect(state.completedBlocks[2].type).toBe("header");
    });
  });

  describe("Code Block Detection", () => {
    test("detects complete code block", () => {
      const code = "```js\nconst x = 1;\n```\n";
      const state = streamAndFinalize(code);
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
      expect(state.completedBlocks[0].content).toBe(code);
      expect(state.completedBlocks[0].isComplete).toBe(true);
    });

    test("code block without language identifier", () => {
      const code = "```\ncode here\n```\n";
      const state = streamAndFinalize(code);
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
    });

    test("code block without closing stays pending", () => {
      const state = streamText("```js\ncode");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.inCodeBlock).toBe(true);
      expect(state.pendingBuffer).toBe("```js\ncode");
    });

    test("backticks split across tokens", () => {
      let state = { ...initialParserState };
      state = parseMarkdownBlocks(state, "``");
      state = parseMarkdownBlocks(state, "`js\ncode\n```");
      state = parseMarkdownBlocks(state, "\n");

      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
    });

    test("nested backticks in code block content", () => {
      const code = "```\nuse ``` for code\n```\n";
      const state = streamAndFinalize(code);
      // The first ``` at start of line after content will close the block
      expect(state.completedBlocks.length).toBeGreaterThanOrEqual(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
    });

    test("inline backticks in text are not code blocks", () => {
      const state = streamAndFinalize("Use `code` inline.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
    });

    test("code block with content after closing", () => {
      const state = streamAndFinalize("```\ncode\n```\nMore text.\n\n");
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("code_block");
      expect(state.completedBlocks[1].type).toBe("paragraph");
    });
  });

  describe("Math Block Detection (CRITICAL)", () => {
    test("detects display math \\[...\\]", () => {
      const math = "\\[x^2 + y^2 = z^2\\]\n";
      const state = streamAndFinalize(math);
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("math_display");
      expect(state.completedBlocks[0].content).toContain("\\[");
      expect(state.completedBlocks[0].content).toContain("\\]");
    });

    test("math delimiters split across tokens", () => {
      let state = { ...initialParserState };
      state = parseMarkdownBlocks(state, "\\");
      state = parseMarkdownBlocks(state, "[x^2");
      state = parseMarkdownBlocks(state, "\\");
      state = parseMarkdownBlocks(state, "]\n");

      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("math_display");
    });

    test("multiline display math", () => {
      const math = "\\[\n\\frac{a}{b}\n+ c\n\\]\n";
      const state = streamAndFinalize(math);
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("math_display");
    });

    test("display math without closing stays pending", () => {
      const state = streamText("\\[x^2");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.inMathBlock).toBe(true);
    });

    test("inline math \\(...\\) stays in paragraph", () => {
      const state = streamAndFinalize("The formula \\(x^2\\) is quadratic.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[0].content).toContain("\\(");
      expect(state.completedBlocks[0].content).toContain("\\)");
    });

    test("paragraph with inline math followed by display math", () => {
      const text = "Inline \\(a\\) here.\n\n\\[b^2\\]\n";
      const state = streamAndFinalize(text);
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[0].content).toContain("\\(a\\)");
      expect(state.completedBlocks[1].type).toBe("math_display");
      expect(state.completedBlocks[1].content).toContain("\\[b^2\\]");
    });

    test("multiple display math blocks", () => {
      const text = "\\[a\\]\n\\[b\\]\n";
      const state = streamAndFinalize(text);
      // When streaming char-by-char, consecutive math blocks may have
      // small intermediate blocks. What matters is content integrity.
      expect(getAllContent(state)).toBe(text);
      const mathBlocks = state.completedBlocks.filter((b) => b.type === "math_display");
      expect(mathBlocks).toHaveLength(2);
    });
  });

  describe("List Detection", () => {
    test("detects bullet list item with -", () => {
      const state = streamAndFinalize("- Item one\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("list_item");
    });

    test("detects bullet list item with *", () => {
      const state = streamAndFinalize("* Item one\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("list_item");
    });

    test("detects bullet list item with +", () => {
      const state = streamAndFinalize("+ Item one\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("list_item");
    });

    test("detects numbered list item", () => {
      const state = streamAndFinalize("1. Item one\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("list_item");
    });

    test("consecutive list items become separate blocks", () => {
      const state = streamAndFinalize("- A\n- B\n- C\n");
      expect(state.completedBlocks).toHaveLength(3);
      expect(state.completedBlocks[0].type).toBe("list_item");
      expect(state.completedBlocks[1].type).toBe("list_item");
      expect(state.completedBlocks[2].type).toBe("list_item");
    });

    test("indented list item (nested)", () => {
      const state = streamAndFinalize("  - Nested item\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("list_item");
    });

    test("list item without newline stays pending", () => {
      const state = streamText("- Item");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.pendingBuffer).toBe("- Item");
    });
  });

  describe("Paragraph Detection", () => {
    test("detects paragraph between blank lines", () => {
      const state = streamAndFinalize("Some text here.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
    });

    test("paragraph without trailing blank line stays pending", () => {
      const state = streamText("Some text here.");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.pendingBuffer).toBe("Some text here.");
    });

    test("paragraph ends when header starts", () => {
      const state = streamAndFinalize("Some text.\n# Header\n");
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[1].type).toBe("header");
    });

    test("paragraph ends when code block starts", () => {
      const state = streamAndFinalize("Some text.\n```\ncode\n```\n");
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[1].type).toBe("code_block");
    });

    test("paragraph ends when list starts", () => {
      const state = streamAndFinalize("Intro text.\n- Item\n");
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[1].type).toBe("list_item");
    });

    test("multi-line paragraph", () => {
      const state = streamAndFinalize("Line one.\nLine two.\nLine three.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[0].content).toContain("Line one.");
      expect(state.completedBlocks[0].content).toContain("Line two.");
      expect(state.completedBlocks[0].content).toContain("Line three.");
    });
  });

  describe("Token Streaming Simulation", () => {
    test("character-by-character streaming produces same content", () => {
      const fullText = "# Title\n\nParagraph.\n\n```js\ncode\n```\n";
      const state = streamAndFinalize(fullText, "char");
      expect(getAllContent(state)).toBe(fullText);
    });

    test("various chunk sizes produce same content", () => {
      const fullText = "# Title\n\nParagraph with \\(math\\).\n\n\\[display\\]\n- List\n";

      for (const chunkSize of [1, 2, 3, 5, 7, 10, 20]) {
        const state = streamAndFinalize(fullText, chunkSize);
        expect(getAllContent(state)).toBe(fullText);
      }
    });

    test("single token produces same content", () => {
      const fullText = "# Title\n\nParagraph.\n\n";
      let state = { ...initialParserState };
      state = parseMarkdownBlocks(state, fullText);
      state = finalizeState(state);
      expect(getAllContent(state)).toBe(fullText);
    });

    test("streaming header character by character", () => {
      const header = "# Hello World\n";
      const state = streamText(header, "char");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].content).toBe(header);
    });

    test("streaming code block character by character", () => {
      const code = "```python\nprint('hi')\n```\n";
      // Use streamAndFinalize to merge trailing newline
      const state = streamAndFinalize(code, "char");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
      expect(state.completedBlocks[0].content).toBe(code);
    });
  });

  describe("Integrity Tests", () => {
    test("concatenation preserves all characters", () => {
      const inputs = [
        "Simple text.\n\n",
        "# Header\n\nParagraph.\n\n",
        "```\ncode\n```\n",
        "\\[math\\]\n",
        "- List item\n",
        "Complex:\n# H1\nText \\(x\\) here.\n\n\\[y\\]\n```js\nx\n```\n- Item\n",
      ];

      for (const input of inputs) {
        const state = streamAndFinalize(input, "char");
        expect(getAllContent(state)).toBe(input);
      }
    });

    test("no character duplication", () => {
      const input =
        "# Title\n\nSome \\(x\\) text.\n\n\\[y = mx + b\\]\n```py\ncode\n```\n";
      const state = streamAndFinalize(input, "char");
      expect(getAllContent(state)).toHaveLength(input.length);
    });

    test("no character loss with random chunk sizes", () => {
      const input =
        "# Title\n\nParagraph one.\n\n## Sub\n\n- Item 1\n- Item 2\n\n```\ncode\n```\n";

      // Simulate random-ish chunking
      const chunkSizes = [3, 1, 7, 2, 4, 8, 1, 5, 3];
      let state = { ...initialParserState };
      let pos = 0;
      let chunkIdx = 0;

      while (pos < input.length) {
        const size = chunkSizes[chunkIdx % chunkSizes.length];
        const token = input.slice(pos, pos + size);
        state = parseMarkdownBlocks(state, token);
        pos += size;
        chunkIdx++;
      }

      state = finalizeState(state);
      expect(getAllContent(state)).toBe(input);
    });

    test("empty input produces empty state", () => {
      const state = streamAndFinalize("");
      expect(state.completedBlocks).toHaveLength(0);
      expect(state.pendingBuffer).toBe("");
    });
  });

  describe("Edge Cases", () => {
    test("handles backslash not followed by [ or ]", () => {
      const state = streamAndFinalize("Use \\n for newline.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
    });

    test("handles partial delimiter at end", () => {
      // Backslash at end - might be start of \[
      let state = streamText("Some text \\");
      expect(state.pendingBuffer).toBe("Some text \\");

      // Complete it as not math
      state = parseMarkdownBlocks(state, "n newline.\n\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("paragraph");
    });

    test("handles empty code block", () => {
      const state = streamAndFinalize("```\n```\n");
      expect(state.completedBlocks).toHaveLength(1);
      expect(state.completedBlocks[0].type).toBe("code_block");
    });

    test("handles single character tokens", () => {
      const text = "Hi\n\n";
      const state = streamAndFinalize(text, 1);
      expect(getAllContent(state)).toBe(text);
    });

    test("finalizeState handles incomplete code block", () => {
      const state = streamText("```js\nincomplete code");
      expect(state.inCodeBlock).toBe(true);

      const finalized = finalizeState(state);
      expect(finalized.completedBlocks).toHaveLength(1);
      expect(finalized.completedBlocks[0].type).toBe("code_block");
      expect(finalized.pendingBuffer).toBe("");
    });

    test("finalizeState handles incomplete math block", () => {
      const state = streamText("\\[incomplete math");
      expect(state.inMathBlock).toBe(true);

      const finalized = finalizeState(state);
      expect(finalized.completedBlocks).toHaveLength(1);
      expect(finalized.completedBlocks[0].type).toBe("math_display");
    });

    test("header after paragraph on same line segment", () => {
      // Paragraph ends, header starts on next line
      const state = streamAndFinalize("Text.\n# Header\n");
      expect(state.completedBlocks).toHaveLength(2);
      expect(state.completedBlocks[0].type).toBe("paragraph");
      expect(state.completedBlocks[1].type).toBe("header");
    });
  });

  describe("Code Blocks with Empty Lines (Bug Fix)", () => {
    test("code block with internal empty lines is not split (char-by-char)", () => {
      const content = `\`\`\`python
# 1D
BatchNorm1d = True

# 2D
BatchNorm2d = True

# 3D
BatchNorm3d = True
\`\`\`
`;
      const state = streamAndFinalize(content, "char");
      const codeBlocks = state.completedBlocks.filter((b) => b.type === "code_block");
      expect(codeBlocks).toHaveLength(1);
      expect(codeBlocks[0].content).toContain("# 2D");
      expect(codeBlocks[0].content).toContain("# 3D");
      expect(codeBlocks[0].isComplete).toBe(true);
    });

    test("exact session content: header then code block with empty lines", () => {
      const content = `## Normalization Layers

\`\`\`python
# 1D
nn.BatchNorm1d(num_features)
nn.InstanceNorm1d(num_features)
nn.LayerNorm(normalized_shape)

# 2D
nn.BatchNorm2d(num_features)
nn.InstanceNorm2d(num_features)

# 3D
nn.BatchNorm3d(num_features)
nn.InstanceNorm3d(num_features)
\`\`\`
`;
      const state = streamAndFinalize(content, "char");
      const types = state.completedBlocks.map((b) => b.type);
      expect(types).toContain("header");
      expect(types).toContain("code_block");

      // The code block must be a single block, not fragmented
      const codeBlocks = state.completedBlocks.filter((b) => b.type === "code_block");
      expect(codeBlocks).toHaveLength(1);
      expect(codeBlocks[0].content).toContain("# 1D");
      expect(codeBlocks[0].content).toContain("# 2D");
      expect(codeBlocks[0].content).toContain("# 3D");
    });

    test("code block after single-newline paragraph boundary is detected", () => {
      const content = "Some intro text.\n```js\nconst x = 1;\n\nconst y = 2;\n```\n";
      const state = streamAndFinalize(content, "char");
      expect(state.completedBlocks.some((b) => b.type === "paragraph")).toBe(true);
      const codeBlocks = state.completedBlocks.filter((b) => b.type === "code_block");
      expect(codeBlocks).toHaveLength(1);
      expect(codeBlocks[0].content).toContain("const x = 1;");
      expect(codeBlocks[0].content).toContain("const y = 2;");
    });

    test("regression: paragraph → double-newline → list still works", () => {
      const content = "Paragraph text.\n\n- List item one\n- List item two\n";
      const state = streamAndFinalize(content, "char");
      expect(state.completedBlocks[0].type).toBe("paragraph");
      const listItems = state.completedBlocks.filter((b) => b.type === "list_item");
      expect(listItems).toHaveLength(2);
    });

    test("code block with empty lines at various chunk sizes", () => {
      const content = "```\nline1\n\nline2\n\nline3\n```\n";
      for (const chunkSize of [1, 2, 3, 5, 7, 10, 20]) {
        const state = streamAndFinalize(content, chunkSize);
        const codeBlocks = state.completedBlocks.filter((b) => b.type === "code_block");
        expect(codeBlocks).toHaveLength(1);
        expect(codeBlocks[0].content).toContain("line1");
        expect(codeBlocks[0].content).toContain("line2");
        expect(codeBlocks[0].content).toContain("line3");
        expect(getAllContent(state)).toBe(content);
      }
    });

    test("single-token processing of full content with code block", () => {
      const content = "Text before.\n```python\ncode\n\nmore code\n```\nText after.\n\n";
      let state = { ...initialParserState };
      state = parseMarkdownBlocks(state, content);
      state = finalizeState(state);
      const codeBlocks = state.completedBlocks.filter((b) => b.type === "code_block");
      expect(codeBlocks).toHaveLength(1);
      expect(getAllContent(state)).toBe(content);
    });
  });

  describe("Complex Documents", () => {
    test("full document with all block types", () => {
      const doc = `# Main Title

This is an introduction with inline math \\(E = mc^2\\).

## Code Example

\`\`\`python
def hello():
    print("Hello")
\`\`\`

## Math Section

Display math:

\\[
\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
\\]

## List of Items

- First item
- Second item
- Third item

Conclusion paragraph.

`;

      const state = streamAndFinalize(doc, "char");
      expect(getAllContent(state)).toBe(doc);

      // Verify we have the expected block types
      const types = state.completedBlocks.map((b) => b.type);
      expect(types).toContain("header");
      expect(types).toContain("paragraph");
      expect(types).toContain("code_block");
      expect(types).toContain("math_display");
      expect(types).toContain("list_item");
    });

    test("rapid switching between block types", () => {
      const doc = "# H\n- L\n```\nc\n```\n\\[m\\]\nP\n\n";
      const state = streamAndFinalize(doc, 1);
      expect(getAllContent(state)).toBe(doc);
    });
  });
});
