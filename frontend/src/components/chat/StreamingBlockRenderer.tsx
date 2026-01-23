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

// Maximum number of blocks that can animate simultaneously
const MAX_CONCURRENT_ANIMATIONS = 2;

/**
 * Renders blocks from the parser state during streaming.
 *
 * - Completed blocks are rendered with markdown (AnimatedBlock)
 * - Up to 2 blocks can animate concurrently for smoother appearance
 * - Pending buffer is hidden (to avoid showing raw markdown syntax)
 */
export function StreamingBlockRenderer({
  parserState,
  isStreaming,
  onAllAnimationsComplete,
}: StreamingBlockRendererProps) {
  const { completedBlocks } = parserState;

  // Track which blocks have completed their animation (using state for reactivity)
  const [completedAnimations, setCompletedAnimations] = useState<Set<number>>(
    () => new Set()
  );

  // For detecting new stream and completion notification
  const prevBlockCountRef = useRef(0);
  const allAnimationsCompleteNotified = useRef(false);

  // Derive highest started index: show up to MAX_CONCURRENT_ANIMATIONS more than completed
  // This allows 2 blocks to animate concurrently while ensuring we don't render blocks
  // that haven't started animating yet
  const highestStartedIndex = Math.min(
    completedBlocks.length - 1,
    completedAnimations.size + MAX_CONCURRENT_ANIMATIONS - 1
  );

  // Called when a specific block finishes animating
  const handleBlockAnimationComplete = useCallback((blockIndex: number) => {
    setCompletedAnimations((prev) => {
      if (prev.has(blockIndex)) return prev;
      const next = new Set(prev);
      next.add(blockIndex);
      return next;
    });
  }, []);

  // Reset when starting a new stream
  useEffect(() => {
    if (completedBlocks.length === 0 && prevBlockCountRef.current > 0) {
      queueMicrotask(() => {
        setCompletedAnimations(new Set());
      });
      allAnimationsCompleteNotified.current = false;
    }
    prevBlockCountRef.current = completedBlocks.length;
  }, [completedBlocks.length]);

  // Notify when all animations complete
  useEffect(() => {
    const allBlocksFinished = completedAnimations.size >= completedBlocks.length;

    if (
      !isStreaming &&
      completedBlocks.length > 0 &&
      allBlocksFinished &&
      !allAnimationsCompleteNotified.current &&
      onAllAnimationsComplete
    ) {
      allAnimationsCompleteNotified.current = true;
      onAllAnimationsComplete();
    }
  }, [
    isStreaming,
    completedBlocks.length,
    completedAnimations.size,
    onAllAnimationsComplete,
  ]);

  return (
    <>
      {completedBlocks.map((block, index) => {
        // Don't render blocks that haven't started yet
        if (index > highestStartedIndex) return null;

        // Animate if this block hasn't completed its animation yet
        const isAnimating = !completedAnimations.has(index);

        return (
          <AnimatedBlock
            key={`block-${index}-${block.type}`}
            block={block}
            animate={isAnimating}
            animationType={getAnimationType(block.type)}
            onAnimationComplete={() => handleBlockAnimationComplete(index)}
          />
        );
      })}
    </>
  );
}
