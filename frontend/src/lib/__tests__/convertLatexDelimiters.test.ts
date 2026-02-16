import { describe, test, expect } from "vitest";
import { convertLatexDelimiters } from "../utils";

describe("convertLatexDelimiters", () => {
  describe("Display math \\[...\\]", () => {
    test("single-line display math", () => {
      expect(convertLatexDelimiters("\\[ x = y \\]")).toBe("$$ x = y $$");
    });

    test("display math without spaces", () => {
      expect(convertLatexDelimiters("\\[x=y\\]")).toBe("$$x=y$$");
    });

    test("multiline aligned environment - preserves newlines", () => {
      const input = "\\[\n\\begin{aligned}\n& x = 1 \\\\\n& y = 2\n\\end{aligned}\n\\]";
      const output = convertLatexDelimiters(input);
      // Must preserve newlines for remark-math to parse correctly
      expect(output).toContain("$$\n");
      expect(output).toContain("\n$$");
      expect(output).toContain("\\begin{aligned}");
    });

    test("cases environment with ampersands", () => {
      const input =
        "\\[\nf(x) = \\begin{cases}\n1 & x > 0 \\\\\n0 & x \\leq 0\n\\end{cases}\n\\]";
      const output = convertLatexDelimiters(input);
      expect(output).toContain("\\begin{cases}");
      expect(output).toContain("& x > 0");
    });

    test("matrix environment", () => {
      const input = "\\[\n\\begin{pmatrix}\na & b \\\\\nc & d\n\\end{pmatrix}\n\\]";
      const output = convertLatexDelimiters(input);
      expect(output).toContain("\\begin{pmatrix}");
    });

    test("multiple display math blocks", () => {
      const input = "First: \\[ a = b \\] Second: \\[ c = d \\]";
      const output = convertLatexDelimiters(input);
      expect(output).toBe("First: $$ a = b $$ Second: $$ c = d $$");
    });
  });

  describe("Inline math \\(...\\)", () => {
    test("simple inline math", () => {
      expect(convertLatexDelimiters("\\( x = y \\)")).toBe("$ x = y $");
    });

    test("inline math without spaces", () => {
      expect(convertLatexDelimiters("\\(x=y\\)")).toBe("$x=y$");
    });

    test("inline math in sentence", () => {
      const input = "The equation \\( E = mc^2 \\) is famous.";
      expect(convertLatexDelimiters(input)).toBe("The equation $ E = mc^2 $ is famous.");
    });

    test("multiple inline math", () => {
      const input = "Given \\( x \\) and \\( y \\), find \\( z \\).";
      expect(convertLatexDelimiters(input)).toBe("Given $ x $ and $ y $, find $ z $.");
    });
  });

  describe("Mixed content", () => {
    test("display and inline math together", () => {
      const input = "Consider \\( x \\in \\mathbb{R} \\). Then:\n\\[ f(x) = x^2 \\]";
      const output = convertLatexDelimiters(input);
      expect(output).toContain("$ x \\in \\mathbb{R} $");
      expect(output).toContain("$$ f(x) = x^2 $$");
    });

    test("real-world KKT conditions example", () => {
      const input = `\\[
\\begin{aligned}
& \\underset{x \\in \\mathbb{R}^n}{\\text{minimize}} & & f(x) \\\\
& \\text{subject to} & & g_i(x) \\leq 0
\\end{aligned}
\\]`;
      const output = convertLatexDelimiters(input);
      // Critical: newlines must be preserved
      expect(output.startsWith("$$\n")).toBe(true);
      expect(output.endsWith("\n$$")).toBe(true);
      expect(output).toContain("\\begin{aligned}");
      expect(output).toContain("& \\underset");
    });
  });

  describe("Triple dollar sign sanitization", () => {
    test("$$$ becomes $$ $", () => {
      expect(convertLatexDelimiters("$$$")).toBe("$$ $");
    });

    test("$$$$ becomes $$ $$", () => {
      expect(convertLatexDelimiters("$$$$")).toBe("$$ $$");
    });
  });

  describe("Indented display math (list items)", () => {
    test("display math indented inside list item - strips indentation from $$", () => {
      const input = "   - For a generic layer:\n     \\[\n     \\frac{a}{b}\n     \\]";
      const output = convertLatexDelimiters(input);
      // $$ must NOT be indented (would trigger CommonMark indented code block)
      expect(output).toContain("\n$$\n");
      expect(output).not.toMatch(/^[ \t]+\$\$/m);
    });

    test("preserves non-delimiter indented content between $$", () => {
      const input = "     \\[\n     \\frac{a}{b}\n     \\]";
      const output = convertLatexDelimiters(input);
      // The math content between $$ markers can stay indented (KaTeX ignores whitespace)
      expect(output).toContain("\\frac{a}{b}");
      // But $$ delimiters themselves must be at column 0
      expect(output).toMatch(/^\$\$/m);
    });

    test("multiple indented display math blocks in list", () => {
      const input = "- Item 1:\n  \\[\n  a = b\n  \\]\n- Item 2:\n  \\[\n  c = d\n  \\]";
      const output = convertLatexDelimiters(input);
      // Both $$ pairs should be de-indented
      const dollarLines = output.split("\n").filter((l) => l.trim() === "$$");
      expect(dollarLines).toHaveLength(4);
      dollarLines.forEach((l) => expect(l).toBe("$$"));
    });
  });

  describe("Edge cases", () => {
    test("empty input", () => {
      expect(convertLatexDelimiters("")).toBe("");
    });

    test("null input", () => {
      expect(convertLatexDelimiters(null)).toBe("");
    });

    test("undefined input", () => {
      expect(convertLatexDelimiters(undefined)).toBe("");
    });

    test("no math content", () => {
      expect(convertLatexDelimiters("Just plain text")).toBe("Just plain text");
    });

    test("already $$ delimited (passthrough)", () => {
      expect(convertLatexDelimiters("$$ x = y $$")).toBe("$$ x = y $$");
    });
  });
});
