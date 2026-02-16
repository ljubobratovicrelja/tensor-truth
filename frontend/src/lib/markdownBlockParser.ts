/**
 * Markdown block parser for progressive streaming.
 *
 * Parses incoming tokens into logical markdown blocks that can be rendered
 * progressively during streaming. Each block is rendered with markdown
 * once complete, while pending content is shown as plain text.
 */

export type BlockType =
  | "header"
  | "code_block"
  | "math_display"
  | "list_item"
  | "paragraph"
  | "pending";

export interface MarkdownBlock {
  type: BlockType;
  content: string;
  isComplete: boolean;
}

export interface ParserState {
  completedBlocks: MarkdownBlock[];
  pendingBuffer: string;
  currentBlockType: BlockType | null;
  /** Track if we're inside a code block (waiting for closing ```) */
  inCodeBlock: boolean;
  /** Track if we're inside a display math block (waiting for closing \]) */
  inMathBlock: boolean;
  /** Track if we're building a list group (top-level item + sub-items) */
  inListBlock: boolean;
  /** Indentation of the first list marker in the current group */
  listBlockBaseIndent: number;
}

export const initialParserState: ParserState = {
  completedBlocks: [],
  pendingBuffer: "",
  currentBlockType: null,
  inCodeBlock: false,
  inMathBlock: false,
  inListBlock: false,
  listBlockBaseIndent: 0,
};

/**
 * Main parser function - processes new tokens and returns updated state.
 */
export function parseMarkdownBlocks(state: ParserState, newToken: string): ParserState {
  let buffer = state.pendingBuffer + newToken;
  const completedBlocks = [...state.completedBlocks];
  let inCodeBlock = state.inCodeBlock;
  let inMathBlock = state.inMathBlock;
  let inListBlock = state.inListBlock;
  let listBlockBaseIndent = state.listBlockBaseIndent;

  // Keep trying to extract complete blocks
  let extractionOccurred = true;
  while (extractionOccurred && buffer.length > 0) {
    extractionOccurred = false;

    // If inside a code block, only look for closing ```
    if (inCodeBlock) {
      const result = tryCompleteCodeBlock(buffer);
      if (result.block) {
        completedBlocks.push(result.block);
        buffer = result.remaining;
        inCodeBlock = !result.block.isComplete;
        extractionOccurred = true;
        continue;
      }
      break; // Stay in code block mode, wait for more content
    }

    // If inside a list block, group sub-items and continuations
    if (inListBlock) {
      const newlineIdx = buffer.indexOf("\n");
      if (newlineIdx === -1) break; // wait for more tokens

      const line = buffer.slice(0, newlineIdx + 1);
      const indent = line.search(/\S/);
      const isBlankLine = indent === -1;

      if (isBlankLine) {
        // Extend last completed block with blank line
        const lastIdx = completedBlocks.length - 1;
        completedBlocks[lastIdx] = {
          ...completedBlocks[lastIdx],
          content: completedBlocks[lastIdx].content + line,
        };
        buffer = buffer.slice(newlineIdx + 1);
        // Two consecutive blank lines end the list (content ends with \n\n\n)
        if (completedBlocks[lastIdx].content.endsWith("\n\n\n")) {
          inListBlock = false;
        }
        extractionOccurred = true;
        continue;
      }

      // Non-blank line with indent > baseIndent → sub-item or continuation
      if (indent > listBlockBaseIndent) {
        const lastIdx = completedBlocks.length - 1;
        completedBlocks[lastIdx] = {
          ...completedBlocks[lastIdx],
          content: completedBlocks[lastIdx].content + line,
        };
        buffer = buffer.slice(newlineIdx + 1);
        extractionOccurred = true;
        continue;
      }

      // indent <= baseIndent → finalize list group, let main loop re-process
      inListBlock = false;
      extractionOccurred = true;
      continue;
    }

    // If inside a math display block, only look for closing \]
    if (inMathBlock) {
      const result = tryCompleteMathBlock(buffer);
      if (result.block) {
        completedBlocks.push(result.block);
        buffer = result.remaining;
        inMathBlock = !result.block.isComplete;
        extractionOccurred = true;
        continue;
      }
      break; // Stay in math block mode, wait for more content
    }

    // Check for block starts in priority order
    // 1. Code block start
    if (buffer.startsWith("```")) {
      const result = tryCompleteCodeBlock(buffer);
      if (result.block && result.block.isComplete) {
        completedBlocks.push(result.block);
        buffer = result.remaining;
        extractionOccurred = true;
        continue;
      }
      // Not complete, enter code block mode
      inCodeBlock = true;
      break;
    }

    // 2. Display math block start \[ (possibly indented inside list items)
    if (buffer.trimStart().startsWith("\\[")) {
      const result = tryCompleteMathBlock(buffer);
      if (result.block && result.block.isComplete) {
        completedBlocks.push(result.block);
        buffer = result.remaining;
        extractionOccurred = true;
        continue;
      }
      // Not complete, enter math block mode
      inMathBlock = true;
      break;
    }

    // 3. Header (at line start)
    if (isAtLineStart(buffer, completedBlocks) && buffer.match(/^#{1,6}\s/)) {
      const result = tryCompleteHeader(buffer);
      if (result.block && result.block.isComplete) {
        completedBlocks.push(result.block);
        buffer = result.remaining;
        extractionOccurred = true;
        continue;
      }
      break; // Header not complete yet
    }

    // 4. List item (at line start)
    if (
      isAtLineStart(buffer, completedBlocks) &&
      (buffer.match(/^[\s]*[-*+]\s/) || buffer.match(/^[\s]*\d+\.\s/))
    ) {
      const result = tryCompleteListItem(buffer);
      if (result.block && result.block.isComplete) {
        listBlockBaseIndent = buffer.search(/\S/);
        completedBlocks.push(result.block);
        buffer = result.remaining;
        inListBlock = true;
        extractionOccurred = true;
        continue;
      }
      break; // List item not complete yet
    }

    // 5. Default: treat as paragraph
    const result = tryCompleteParagraph(buffer);
    if (result.block && result.block.isComplete) {
      completedBlocks.push(result.block);
      buffer = result.remaining;
      extractionOccurred = true;
      continue;
    }

    break; // Nothing could be extracted
  }

  return {
    completedBlocks,
    pendingBuffer: buffer,
    currentBlockType: detectBlockType(buffer, inCodeBlock, inMathBlock),
    inCodeBlock,
    inMathBlock,
    inListBlock,
    listBlockBaseIndent,
  };
}

/**
 * Helper to check if we're at the start of a line (for header/list detection).
 */
function isAtLineStart(_buffer: string, completedBlocks: MarkdownBlock[]): boolean {
  if (completedBlocks.length === 0) {
    return true;
  }
  const lastBlock = completedBlocks[completedBlocks.length - 1];
  return lastBlock.content.endsWith("\n");
}

/**
 * Detect what type of block the buffer might become.
 */
function detectBlockType(
  buffer: string,
  inCodeBlock: boolean,
  inMathBlock: boolean
): BlockType | null {
  if (!buffer) return null;
  if (inCodeBlock) return "code_block";
  if (inMathBlock) return "math_display";
  if (buffer.startsWith("```")) return "code_block";
  if (buffer.trimStart().startsWith("\\[")) return "math_display";
  if (buffer.match(/^#{1,6}\s/)) return "header";
  if (buffer.match(/^[\s]*[-*+]\s/) || buffer.match(/^[\s]*\d+\.\s/)) {
    return "list_item";
  }
  return "paragraph";
}

/**
 * Try to extract a complete header block.
 * Headers end at newline.
 */
function tryCompleteHeader(buffer: string): {
  block: MarkdownBlock | null;
  remaining: string;
} {
  const newlineIndex = buffer.indexOf("\n");
  if (newlineIndex === -1) {
    return { block: null, remaining: buffer };
  }

  // Include the newline in the content
  const content = buffer.slice(0, newlineIndex + 1);
  const remaining = buffer.slice(newlineIndex + 1);

  return {
    block: { type: "header", content, isComplete: true },
    remaining,
  };
}

/**
 * Try to extract a complete code block.
 * Code blocks start with ``` and end with ```.
 */
function tryCompleteCodeBlock(buffer: string): {
  block: MarkdownBlock | null;
  remaining: string;
} {
  // Must start with ```
  if (!buffer.startsWith("```")) {
    return { block: null, remaining: buffer };
  }

  // Find the closing ``` (must be on its own line or at buffer end)
  // Search after the opening ``` plus any language identifier
  const firstNewline = buffer.indexOf("\n");
  if (firstNewline === -1) {
    return { block: null, remaining: buffer };
  }

  // Look for closing ``` starting from after the first newline
  // The closing ``` must be at the start of a line
  let searchStart = firstNewline + 1;
  while (searchStart < buffer.length) {
    const closingIndex = buffer.indexOf("```", searchStart);
    if (closingIndex === -1) {
      return { block: null, remaining: buffer };
    }

    // Check if the ``` is at the start of a line
    const charBefore = buffer[closingIndex - 1];
    if (charBefore === "\n" || closingIndex === searchStart) {
      // Found valid closing - find the end (newline after ``` or end of string)
      let endIndex = closingIndex + 3;
      if (buffer[endIndex] === "\n") {
        endIndex++;
      }

      const content = buffer.slice(0, endIndex);
      const remaining = buffer.slice(endIndex);

      return {
        block: { type: "code_block", content, isComplete: true },
        remaining,
      };
    }

    // Not at line start, keep searching
    searchStart = closingIndex + 3;
  }

  return { block: null, remaining: buffer };
}

/**
 * Try to extract a complete display math block.
 * Display math starts with \[ and ends with \].
 */
function tryCompleteMathBlock(buffer: string): {
  block: MarkdownBlock | null;
  remaining: string;
} {
  // Must start with \[ (possibly indented inside list items)
  if (!buffer.trimStart().startsWith("\\[")) {
    return { block: null, remaining: buffer };
  }

  // Find the closing \]
  const closingIndex = buffer.indexOf("\\]");
  if (closingIndex === -1) {
    return { block: null, remaining: buffer };
  }

  // Include the closing \] and any following newline
  let endIndex = closingIndex + 2;
  if (buffer[endIndex] === "\n") {
    endIndex++;
  }

  const content = buffer.slice(0, endIndex);
  const remaining = buffer.slice(endIndex);

  return {
    block: { type: "math_display", content, isComplete: true },
    remaining,
  };
}

/**
 * Try to extract a complete list item.
 * List items end at newline.
 */
function tryCompleteListItem(buffer: string): {
  block: MarkdownBlock | null;
  remaining: string;
} {
  const newlineIndex = buffer.indexOf("\n");
  if (newlineIndex === -1) {
    return { block: null, remaining: buffer };
  }

  // Include the newline in the content
  const content = buffer.slice(0, newlineIndex + 1);
  const remaining = buffer.slice(newlineIndex + 1);

  return {
    block: { type: "list_item", content, isComplete: true },
    remaining,
  };
}

/**
 * Try to extract a complete paragraph.
 * Paragraphs end at double newline or when another block type starts.
 */
function tryCompleteParagraph(buffer: string): {
  block: MarkdownBlock | null;
  remaining: string;
} {
  // Single pass: scan each \n and check both conditions, take first match.
  // Block starts take priority over \n\n ONLY when afterNewline doesn't
  // itself start with \n (which would be a double-newline paragraph break).
  let searchPos = 0;
  while (searchPos < buffer.length) {
    const newlinePos = buffer.indexOf("\n", searchPos);
    if (newlinePos === -1) break;

    const afterNewline = buffer.slice(newlinePos + 1);

    // 1. Block start after newline (but NOT if it's actually a double newline)
    if (
      !afterNewline.startsWith("\n") &&
      (afterNewline.startsWith("```") ||
        afterNewline.trimStart().startsWith("\\[") ||
        afterNewline.match(/^#{1,6}\s/) ||
        afterNewline.match(/^[\s]*[-*+]\s/) ||
        afterNewline.match(/^[\s]*\d+\.\s/))
    ) {
      const content = buffer.slice(0, newlinePos + 1);
      const remaining = buffer.slice(newlinePos + 1);
      return {
        block: { type: "paragraph", content, isComplete: true },
        remaining,
      };
    }

    // 2. Double newline (paragraph break)
    if (afterNewline.startsWith("\n")) {
      const content = buffer.slice(0, newlinePos + 2);
      const remaining = buffer.slice(newlinePos + 2);
      return {
        block: { type: "paragraph", content, isComplete: true },
        remaining,
      };
    }

    searchPos = newlinePos + 1;
  }

  return { block: null, remaining: buffer };
}

/**
 * Concatenate all content from completed blocks and pending buffer.
 * Used for integrity checks - should equal original input.
 */
export function getAllContent(state: ParserState): string {
  return state.completedBlocks.map((b) => b.content).join("") + state.pendingBuffer;
}

/**
 * Finalize the state when streaming ends.
 * Converts any pending buffer to a final paragraph block.
 * If pending buffer is just whitespace, merges it with the previous block.
 */
export function finalizeState(state: ParserState): ParserState {
  if (!state.pendingBuffer) {
    return state;
  }

  // If in list block, merge pending buffer into the last list block
  if (state.inListBlock && state.completedBlocks.length > 0) {
    const blocks = [...state.completedBlocks];
    const lastBlock = blocks[blocks.length - 1];
    blocks[blocks.length - 1] = {
      ...lastBlock,
      content: lastBlock.content + state.pendingBuffer,
    };
    return {
      completedBlocks: blocks,
      pendingBuffer: "",
      currentBlockType: null,
      inCodeBlock: false,
      inMathBlock: false,
      inListBlock: false,
      listBlockBaseIndent: 0,
    };
  }

  // If pending buffer is just whitespace and we have previous blocks,
  // merge it with the last block (handles trailing newlines from streaming)
  if (
    /^\s*$/.test(state.pendingBuffer) &&
    state.completedBlocks.length > 0 &&
    !state.inCodeBlock &&
    !state.inMathBlock
  ) {
    const blocks = [...state.completedBlocks];
    const lastBlock = blocks[blocks.length - 1];
    blocks[blocks.length - 1] = {
      ...lastBlock,
      content: lastBlock.content + state.pendingBuffer,
    };
    return {
      completedBlocks: blocks,
      pendingBuffer: "",
      currentBlockType: null,
      inCodeBlock: false,
      inMathBlock: false,
      inListBlock: false,
      listBlockBaseIndent: 0,
    };
  }

  // Convert pending buffer to a final block
  const finalBlock: MarkdownBlock = {
    type: state.inCodeBlock
      ? "code_block"
      : state.inMathBlock
        ? "math_display"
        : "paragraph",
    content: state.pendingBuffer,
    isComplete: true,
  };

  return {
    completedBlocks: [...state.completedBlocks, finalBlock],
    pendingBuffer: "",
    currentBlockType: null,
    inCodeBlock: false,
    inMathBlock: false,
    inListBlock: false,
    listBlockBaseIndent: 0,
  };
}
