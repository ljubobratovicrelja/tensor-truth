import { memo, useMemo } from "react";
import React from "react";
import ReactMarkdown from "react-markdown";
import type { PluggableList } from "unified";
import type { Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { cn, convertLatexDelimiters, preprocessTableCodeBlocks } from "@/lib/utils";

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

// Custom code component that renders %%BR%% placeholders as actual line breaks
const markdownComponents: Components = {
  code({ children, className, ...props }) {
    // Only process inline code (no className means not inside a <pre> from highlight)
    if (className) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }

    const text = String(children);
    if (!text.includes("%%BR%%")) {
      return <code {...props}>{children}</code>;
    }

    const parts = text.split("%%BR%%");
    return (
      <code {...props}>
        {parts.map((part, i) =>
          i < parts.length - 1
            ? React.createElement(
                React.Fragment,
                { key: i },
                part,
                React.createElement("br")
              )
            : React.createElement(React.Fragment, { key: i }, part)
        )}
      </code>
    );
  },
};

function MarkdownRenderer({ content, className }: MemoizedMarkdownProps) {
  const convertedContent = useMemo(
    () => preprocessTableCodeBlocks(convertLatexDelimiters(content)),
    [content]
  );

  return (
    <div className={cn("chat-markdown max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={markdownComponents}
      >
        {convertedContent}
      </ReactMarkdown>
    </div>
  );
}

export const MemoizedMarkdown = memo(
  MarkdownRenderer,
  (prev, next) => prev.content === next.content && prev.className === next.className
);
