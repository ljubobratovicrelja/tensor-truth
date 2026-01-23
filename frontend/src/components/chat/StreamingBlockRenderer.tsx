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
  /** When false, all blocks render immediately without animation (for historical messages) */
  animate?: boolean;
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
  animate = true,
}: StreamingBlockRendererProps) {
  const { completedBlocks } = parserState;

  // Track which blocks have completed their animation (using state for reactivity)
  const [completedAnimations, setCompletedAnimations] = useState<Set<number>>(
    () => new Set()
  );

  // For detecting new stream reset
  const prevBlockCountRef = useRef(0);

  // Derive highest started index: show up to MAX_CONCURRENT_ANIMATIONS more than completed
  // When animations are disabled, show all blocks immediately
  const highestStartedIndex = animate
    ? Math.min(
        completedBlocks.length - 1,
        completedAnimations.size + MAX_CONCURRENT_ANIMATIONS - 1
      )
    : completedBlocks.length - 1;

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
    }
    prevBlockCountRef.current = completedBlocks.length;
  }, [completedBlocks.length]);

  return (
    <>
      {completedBlocks.map((block, index) => {
        // Don't render blocks that haven't started yet
        if (index > highestStartedIndex) return null;

        // Animate if animations enabled and this block hasn't completed yet
        const isAnimating = animate && !completedAnimations.has(index);

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
