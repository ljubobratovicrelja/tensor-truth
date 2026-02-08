import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Converts LaTeX math delimiters from standard LaTeX format to markdown format.
 *
 * Converts:
 * - \(...\) to $...$ (inline math)
 * - \[...\] to $$...$$ (display math)
 *
 * Also sanitizes problematic delimiter patterns:
 * - $$$ (three dollar signs) -> $$ $ (prevents remark-math parsing confusion)
 *
 * LLMs often use standard LaTeX notation \(...\) and \[...\] when generating
 * mathematical content, but remark-math/rehype-katex expect dollar sign delimiters.
 *
 * @param text - String containing LaTeX expressions with standard delimiters
 * @returns String with markdown-compatible LaTeX delimiters
 */
export function convertLatexDelimiters(text: string | null | undefined): string {
  if (!text) {
    return "";
  }

  // Convert display math \[...\] to $$...$$
  // Preserve all whitespace/newlines for proper remark-math parsing of environments like \begin{aligned}
  let converted = text.replace(/\\\[([\s\S]*?)\\\]/g, (_match, p1) => `$$${p1}$$`);

  // Convert inline math \(...\) to $...$
  // Preserve whitespace to maintain original formatting
  converted = converted.replace(/\\\(([\s\S]*?)\\\)/g, (_match, p1) => `$${p1}$`);

  // Sanitize $$$ (triple dollar signs) which confuse remark-math parser
  // This typically happens when display math ($$) is immediately followed by inline math ($)
  // Convert $$$ to $$ $ (add space to separate the delimiters)
  converted = converted.replace(/\${3,}/g, (match) => {
    // For $$$, insert space: "$$ $"
    // For $$$$, insert space: "$$ $$"
    // etc.
    const pairs = Math.floor(match.length / 2);
    const remainder = match.length % 2;
    const pairsStr = Array(pairs).fill("$$").join(" ");
    return pairsStr + (remainder ? " $" : "");
  });

  return converted;
}

/**
 * Preprocesses markdown tables that contain fenced code blocks with `<br>` tags.
 *
 * GFM tables cannot contain fenced code blocks (``` becomes inline code delimiters).
 * LLMs sometimes generate table cells like: ```python<br>code<br>more```
 * This converts them to proper inline code with %%BR%% placeholders that can be
 * rendered as actual line breaks by a custom React component.
 *
 * Only processes lines that look like table rows (start and end with |).
 */
export function preprocessTableCodeBlocks(markdown: string): string {
  const lines = markdown.split("\n");
  return lines
    .map((line) => {
      if (!line.match(/^\|.*\|$/)) return line;
      return line.replace(/```\w*<br\s*\/?>(.*?)```/gi, (_match, code) => {
        const cleanCode = code.replace(/<br\s*\/?>/gi, "%%BR%%");
        return "`" + cleanCode + "`";
      });
    })
    .join("\n");
}
