/**
 * Markdown splitter for progressive streaming.
 *
 * Parses markdown into an MDAST using remark, then splits into stable
 * (all children except last) and unstable (last child) blocks.
 * Stable blocks are cached and never re-rendered; the unstable block
 * is re-rendered each frame to show partial content immediately.
 */

import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { toMarkdown } from "mdast-util-to-markdown";
import { gfmToMarkdown } from "mdast-util-gfm";
import { mathToMarkdown } from "mdast-util-math";
import type { Root, RootContent } from "mdast";

export interface SplitResult {
  stableBlocks: string[];
  unstableBlock: string;
  isIncompleteCodeFence: boolean;
}

export const EMPTY_RESULT: SplitResult = {
  stableBlocks: [],
  unstableBlock: "",
  isIncompleteCodeFence: false,
};

// Create parser once at module level
const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);

// Serialization extensions (reused across calls)
const markdownExtensions = [gfmToMarkdown(), mathToMarkdown()];

function serializeNode(node: RootContent): string {
  // Wrap in a root node for serialization
  const root: Root = { type: "root", children: [node] };
  return toMarkdown(root, { extensions: markdownExtensions });
}

/**
 * Count triple-backtick fences at line starts.
 * Odd count means there's an unclosed code fence.
 */
function hasIncompleteCodeFence(content: string): boolean {
  let count = 0;
  const lines = content.split("\n");
  for (const line of lines) {
    if (/^`{3,}/.test(line.trimStart())) {
      count++;
    }
  }
  return count % 2 === 1;
}

/**
 * Split markdown content into stable and unstable blocks using remark AST.
 *
 * All AST children except the last are stable (won't change as more tokens arrive).
 * The last child is unstable (may grow with new tokens).
 */
export function splitMarkdown(content: string): SplitResult {
  if (!content) return EMPTY_RESULT;

  const tree = parser.parse(content);
  const children = tree.children;

  if (children.length === 0) {
    return {
      stableBlocks: [],
      unstableBlock: content,
      isIncompleteCodeFence: hasIncompleteCodeFence(content),
    };
  }

  if (children.length === 1) {
    return {
      stableBlocks: [],
      unstableBlock: serializeNode(children[0]),
      isIncompleteCodeFence: hasIncompleteCodeFence(content),
    };
  }

  const stableBlocks = children.slice(0, -1).map(serializeNode);
  const unstableBlock = serializeNode(children[children.length - 1]);

  return {
    stableBlocks,
    unstableBlock,
    isIncompleteCodeFence: hasIncompleteCodeFence(content),
  };
}
