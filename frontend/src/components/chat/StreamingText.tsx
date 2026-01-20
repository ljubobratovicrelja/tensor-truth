import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { cn } from "@/lib/utils";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

// Characters per animation cycle - adjust for smoother/chunkier animation
const CHARS_PER_ANIMATION = 20;

/**
 * Component that renders streaming text with a smooth fade-in animation.
 * Animation replays every ~20 characters to create a flowing appearance.
 */
export function StreamingText({ content, isStreaming, className }: StreamingTextProps) {
  // Compute animation key based on content length chunks
  // This causes the animation to replay every CHARS_PER_ANIMATION characters
  const animationKey = useMemo(() => {
    if (!isStreaming) return "static";
    return `streaming-${Math.floor(content.length / CHARS_PER_ANIMATION)}`;
  }, [content.length, isStreaming]);

  return (
    <div
      key={animationKey}
      className={cn(
        "chat-markdown max-w-none",
        isStreaming && "streaming-text",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[
          rehypeHighlight,
          rehypeKatex,
          rehypeSlug,
          [
            rehypeAutolinkHeadings,
            {
              behavior: "wrap",
              properties: {
                className: ["header-anchor"],
              },
            },
          ],
        ]}
      >
        {content}
      </ReactMarkdown>
      {isStreaming && <span className="streaming-cursor" />}
    </div>
  );
}
