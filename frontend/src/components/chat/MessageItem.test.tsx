import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageItem } from "./MessageItem";

describe("MessageItem", () => {
  describe("markdown rendering", () => {
    it("renders basic markdown formatting", () => {
      const message = {
        role: "assistant" as const,
        content: "This is **bold** and *italic* text.",
      };
      render(<MessageItem message={message} />);

      // Check that bold/italic are rendered (strong/em elements)
      expect(screen.getByText("bold").tagName).toBe("STRONG");
      expect(screen.getByText("italic").tagName).toBe("EM");
    });

    it("renders code blocks with language class for syntax highlighting", () => {
      const message = {
        role: "assistant" as const,
        content: "```python\ndef hello():\n    print('Hello')\n```",
      };
      const { container } = render(<MessageItem message={message} />);

      // rehype-highlight adds language-* classes
      const codeBlock = container.querySelector("code");
      expect(codeBlock).toBeTruthy();
      // After rehype-highlight processes it, we expect hljs classes
      expect(codeBlock?.className).toMatch(/language-python|hljs/);
    });

    it("renders inline code", () => {
      const message = {
        role: "assistant" as const,
        content: "Use `const x = 1` for variables.",
      };
      const { container } = render(<MessageItem message={message} />);

      const inlineCode = container.querySelector("code");
      expect(inlineCode).toBeTruthy();
      expect(inlineCode?.textContent).toBe("const x = 1");
    });

    it("renders links", () => {
      const message = {
        role: "assistant" as const,
        content: "Check [this link](https://example.com).",
      };
      render(<MessageItem message={message} />);

      const link = screen.getByRole("link", { name: "this link" });
      expect(link).toHaveAttribute("href", "https://example.com");
    });

    it("renders lists", () => {
      const message = {
        role: "assistant" as const,
        content: "- Item 1\n- Item 2\n- Item 3",
      };
      const { container } = render(<MessageItem message={message} />);

      const listItems = container.querySelectorAll("li");
      expect(listItems).toHaveLength(3);
    });

    it("renders headings", () => {
      const message = {
        role: "assistant" as const,
        content: "## Section Title\n\nSome content here.",
      };
      const { container } = render(<MessageItem message={message} />);

      const heading = container.querySelector("h2");
      expect(heading).toBeTruthy();
      expect(heading?.textContent).toBe("Section Title");
    });
  });

  describe("LaTeX/math rendering", () => {
    it("renders inline math with $ delimiters", () => {
      const message = {
        role: "assistant" as const,
        content: "The equation $E = mc^2$ is famous.",
      };
      const { container } = render(<MessageItem message={message} />);

      // KaTeX renders math in .katex or .math spans
      const mathElement = container.querySelector(".katex, .math-inline, [class*='katex']");
      expect(mathElement).toBeTruthy();
    });

    it("renders block math with $$ delimiters", () => {
      const message = {
        role: "assistant" as const,
        content: "The quadratic formula:\n\n$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\nIs useful.",
      };
      const { container } = render(<MessageItem message={message} />);

      // KaTeX renders display math differently
      const mathElement = container.querySelector(".katex-display, .math-display, [class*='katex']");
      expect(mathElement).toBeTruthy();
    });
  });

  describe("user vs assistant styling", () => {
    it("renders user messages with user icon", () => {
      const message = { role: "user" as const, content: "Hello" };
      const { container } = render(<MessageItem message={message} />);

      // User messages should be reversed (flex-row-reverse)
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper.className).toContain("flex-row-reverse");
    });

    it("renders assistant messages with bot icon", () => {
      const message = { role: "assistant" as const, content: "Hello" };
      const { container } = render(<MessageItem message={message} />);

      // Assistant messages should be normal (flex-row)
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper.className).toContain("flex-row");
      expect(wrapper.className).not.toContain("flex-row-reverse");
    });
  });
});
