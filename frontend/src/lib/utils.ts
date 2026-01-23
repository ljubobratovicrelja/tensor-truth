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
  // Use 's' flag for dotall mode to match across newlines
  let converted = text.replace(/\\\[\s*(.*?)\s*\\\]/gs, (_match, p1) => `$$${p1}$$`);

  // Convert inline math \(...\) to $...$
  converted = converted.replace(/\\\(\s*(.*?)\s*\\\)/gs, (_match, p1) => `$${p1}$`);

  // Sanitize $$$ (triple dollar signs) which confuse remark-math parser
  // This typically happens when display math ($$) is immediately followed by inline math ($)
  // Convert $$$ to $$ $ (add space to separate the delimiters)
  converted = converted.replace(/\${3,}/g, (match) => {
    // For $$$, insert space: "$$ $"
    // For $$$$, insert space: "$$ $$"
    // etc.
    const pairs = Math.floor(match.length / 2);
    const remainder = match.length % 2;
    return "$$".repeat(pairs) + (remainder ? " $" : "");
  });

  return converted;
}
