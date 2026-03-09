import { memo, useEffect, useMemo, useRef, useState } from "react";
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

// Delay before showing error placeholder, so localized URLs can replace
// external ones during streaming without a flash.
const IMAGE_ERROR_DELAY_MS = 3000;

function MarkdownImageInner({
  src,
  alt,
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement>) {
  // "hidden" = image failed but we're waiting before showing placeholder
  // "failed" = grace period expired, show placeholder
  const [state, setState] = useState<"ok" | "hidden" | "failed">("ok");
  const timerRef = useRef<ReturnType<typeof setTimeout>>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (state === "failed") {
    return (
      <span
        className="border-border bg-muted text-muted-foreground inline-flex items-center gap-1.5 rounded border px-3 py-2 text-xs"
        title={src}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
          <circle cx="9" cy="9" r="2" />
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
        </svg>
        {alt || "Image"} — failed to load
      </span>
    );
  }

  return (
    <img
      src={src}
      alt={alt || ""}
      loading="lazy"
      style={state === "hidden" ? { display: "none" } : undefined}
      onError={() => {
        setState("hidden");
        timerRef.current = setTimeout(() => setState("failed"), IMAGE_ERROR_DELAY_MS);
      }}
      {...props}
    />
  );
}

// Wrapper that resets error state when src changes by remounting via key
function MarkdownImage(props: React.ImgHTMLAttributes<HTMLImageElement>) {
  return <MarkdownImageInner key={props.src} {...props} />;
}

// Custom components for markdown rendering
const markdownComponents: Components = {
  img({ src, alt, ...props }) {
    return <MarkdownImage src={src} alt={alt} {...props} />;
  },
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
