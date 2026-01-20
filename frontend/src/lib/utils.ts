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

  return converted;
}
