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

// Display names for common language identifiers
const LANGUAGE_LABELS: Record<string, string> = {
  js: "JavaScript",
  jsx: "JavaScript (JSX)",
  ts: "TypeScript",
  tsx: "TypeScript (TSX)",
  py: "Python",
  python: "Python",
  rb: "Ruby",
  rs: "Rust",
  go: "Go",
  sh: "Shell",
  bash: "Bash",
  zsh: "Zsh",
  fish: "Fish",
  ps1: "PowerShell",
  powershell: "PowerShell",
  yml: "YAML",
  yaml: "YAML",
  md: "Markdown",
  json: "JSON",
  html: "HTML",
  css: "CSS",
  scss: "SCSS",
  sql: "SQL",
  graphql: "GraphQL",
  dockerfile: "Dockerfile",
  tf: "Terraform",
  hcl: "HCL",
  cpp: "C++",
  "c++": "C++",
  cs: "C#",
  csharp: "C#",
  kt: "Kotlin",
  swift: "Swift",
  java: "Java",
  scala: "Scala",
  php: "PHP",
  lua: "Lua",
  r: "R",
  dart: "Dart",
  toml: "TOML",
  ini: "INI",
  xml: "XML",
  makefile: "Makefile",
  cmake: "CMake",
  zig: "Zig",
  elixir: "Elixir",
  erlang: "Erlang",
  clojure: "Clojure",
  haskell: "Haskell",
  ocaml: "OCaml",
  vim: "Vim",
  plaintext: "Text",
  text: "Text",
  txt: "Text",
};

function getLangLabel(className?: string): string | null {
  if (!className) return null;
  const match = className.match(/language-(\S+)/);
  if (!match) return null;
  const lang = match[1].toLowerCase();
  return LANGUAGE_LABELS[lang] ?? lang.charAt(0).toUpperCase() + lang.slice(1);
}

// Custom components for markdown rendering
const markdownComponents: Components = {
  pre({ children, ...props }) {
    // Extract language from the child <code> element's className
    let lang: string | null = null;
    React.Children.forEach(children, (child) => {
      if (React.isValidElement(child)) {
        const childProps = child.props as { className?: string };
        if (childProps.className) {
          lang = getLangLabel(childProps.className);
        }
      }
    });

    return (
      <pre {...props}>
        {lang && <div className="code-lang-label">{lang}</div>}
        {children}
      </pre>
    );
  },
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
