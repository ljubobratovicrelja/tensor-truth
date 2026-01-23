import { useState, useCallback, useEffect, useRef } from "react";
import { AnimatedBlock, type AnimationType } from "./AnimatedBlock";
import type { ParserState, BlockType } from "@/lib/markdownBlockParser";

/**
 * Determine animation type based on block type.
 *
 * Options:
 * - "fade": All blocks fade in (clean, no partial content shown)
 * - "typewriter": Character-by-character reveal (more dynamic)
 *
 * Currently: All blocks use fade animation.
 * To switch back to typewriter for non-math blocks, check git history.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function getAnimationType(_blockType: BlockType): AnimationType {
  // All blocks use fade animation
  return "fade";

  // Previous behavior (typewriter for most, fade for math):
  // if (blockType === "math_display") {
  //   return "fade";
  // }
  // return "typewriter";
}

interface StreamingBlockRendererProps {
  parserState: ParserState;
  isStreaming: boolean;
  /** Called when all block animations have completed */
  onAllAnimationsComplete?: () => void;
}

/**
 * Renders blocks from the parser state during streaming.
 *
 * - Completed blocks are rendered with markdown (AnimatedBlock)
 * - Animations are queued - only one block animates at a time
 * - Next block's animation starts only when previous completes
 * - Pending buffer is hidden (to avoid showing raw markdown syntax)
 * - Blinking cursor shows while streaming
 */
export function StreamingBlockRenderer({
  parserState,
  isStreaming,
  onAllAnimationsComplete,
}: StreamingBlockRendererProps) {
  const { completedBlocks } = parserState;

  // Track which block is currently animating (by index)
  // Blocks before this index: fully rendered (no animation)
  // Block at this index: currently animating
  // Blocks after this index: waiting to animate
  const [animatingIndex, setAnimatingIndex] = useState(0);
  const prevBlockCountRef = useRef(0);
  const allAnimationsCompleteNotified = useRef(false);

  // When a block finishes animating, move to the next one
  const handleAnimationComplete = useCallback(() => {
    setAnimatingIndex((prev) => prev + 1);
  }, []);

  // Reset animation index when starting a new stream
  useEffect(() => {
    if (completedBlocks.length === 0 && prevBlockCountRef.current > 0) {
      queueMicrotask(() => setAnimatingIndex(0));
      allAnimationsCompleteNotified.current = false;
    }
    prevBlockCountRef.current = completedBlocks.length;
  }, [completedBlocks.length]);

  // Notify when all animations are complete (streaming done + all blocks animated)
  useEffect(() => {
    if (
      !isStreaming &&
      completedBlocks.length > 0 &&
      animatingIndex >= completedBlocks.length &&
      !allAnimationsCompleteNotified.current &&
      onAllAnimationsComplete
    ) {
      allAnimationsCompleteNotified.current = true;
      onAllAnimationsComplete();
    }
  }, [isStreaming, animatingIndex, completedBlocks.length, onAllAnimationsComplete]);

  return (
    <>
      {/* Render blocks up to and including the currently animating one */}
      {completedBlocks.map((block, index) => {
        // Don't render blocks that haven't had their turn yet
        if (index > animatingIndex) return null;

        return (
          <AnimatedBlock
            key={`block-${index}-${block.type}`}
            block={block}
            // Only animate if this is the current block in the queue
            animate={index === animatingIndex}
            // Math blocks fade in, others use typewriter
            animationType={getAnimationType(block.type)}
            onAnimationComplete={handleAnimationComplete}
          />
        );
      })}

    </>
  );
}
