import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { PluggableList } from "unified";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { cn, convertLatexDelimiters } from "@/lib/utils";

interface MemoizedMarkdownProps {
  content: string;
  className?: string;
}

// Module-level stable plugin arrays (prevents ReactMarkdown re-parsing on every render)
const remarkPlugins: PluggableList = [remarkGfm, remarkMath];
const rehypePlugins: PluggableList = [
  [rehypeHighlight, { detect: true }],
  [rehypeKatex, { throwOnError: false, strict: false }],
  rehypeSlug,
  [
    rehypeAutolinkHeadings,
    { behavior: "wrap", properties: { className: ["header-anchor"] } },
  ],
];

function MarkdownRenderer({ content, className }: MemoizedMarkdownProps) {
  const convertedContent = useMemo(() => convertLatexDelimiters(content), [content]);

  return (
    <div className={cn("chat-markdown max-w-none", className)}>
      <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
        {convertedContent}
      </ReactMarkdown>
    </div>
  );
}

export const MemoizedMarkdown = memo(
  MarkdownRenderer,
  (prev, next) => prev.content === next.content && prev.className === next.className
);
