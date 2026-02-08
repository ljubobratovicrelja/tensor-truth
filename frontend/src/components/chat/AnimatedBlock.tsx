import { memo, useState, useEffect, useRef } from "react";
import { MemoizedMarkdown } from "./MemoizedMarkdown";
import type { MarkdownBlock } from "@/lib/markdownBlockParser";

export type AnimationType = "typewriter" | "fade";

interface AnimatedBlockProps {
  block: MarkdownBlock;
  /** Whether to animate this block (typically true for the most recently completed block) */
  animate?: boolean;
  /** Animation type: "typewriter" for character-by-character, "fade" for fade-in */
  animationType?: AnimationType;
  /** Animation speed in characters per batch (higher = faster) - typewriter only */
  charsPerBatch?: number;
  /** Interval between batches in ms (lower = faster) - typewriter only */
  batchInterval?: number;
  /** Duration of fade animation in ms - fade only */
  fadeDuration?: number;
  /** Callback when animation completes */
  onAnimationComplete?: () => void;
}

/**
 * Renders a single markdown block with optional animation.
 *
 * Supports two animation types:
 * - "typewriter": Character-by-character reveal (good for text)
 * - "fade": Fade-in effect (good for math blocks that shouldn't show partial content)
 *
 * LaTeX delimiter conversion (\[..\] -> $$..$$ and \(..\) -> $..$) is handled
 * by MemoizedMarkdown — this component passes raw content through.
 */
function AnimatedBlockComponent({
  block,
  animate = false,
  animationType = "typewriter",
  charsPerBatch = 6,
  batchInterval = 16,
  fadeDuration = 400,
  onAnimationComplete,
}: AnimatedBlockProps) {
  // Use fade animation for this block?
  const useFade = animationType === "fade";

  if (useFade) {
    return (
      <FadeAnimatedBlock
        content={block.content}
        animate={animate}
        fadeDuration={fadeDuration}
        onAnimationComplete={onAnimationComplete}
      />
    );
  }

  return (
    <TypewriterAnimatedBlock
      block={block}
      animate={animate}
      charsPerBatch={charsPerBatch}
      batchInterval={batchInterval}
      onAnimationComplete={onAnimationComplete}
    />
  );
}

// ============================================================================
// FADE ANIMATION
// ============================================================================

interface FadeAnimatedBlockProps {
  content: string;
  animate: boolean;
  fadeDuration: number;
  onAnimationComplete?: () => void;
}

function FadeAnimatedBlock({
  content,
  animate,
  fadeDuration,
  onAnimationComplete,
}: FadeAnimatedBlockProps) {
  const animationCompleteNotified = useRef(false);

  // Notify completion after fade duration
  useEffect(() => {
    if (!animate) {
      // Not animating, notify immediately
      if (!animationCompleteNotified.current && onAnimationComplete) {
        animationCompleteNotified.current = true;
        onAnimationComplete();
      }
      return;
    }

    // Notify completion after fade animation finishes
    const timer = setTimeout(() => {
      if (!animationCompleteNotified.current && onAnimationComplete) {
        animationCompleteNotified.current = true;
        onAnimationComplete();
      }
    }, fadeDuration);

    return () => clearTimeout(timer);
  }, [animate, fadeDuration, onAnimationComplete]);

  return (
    <div
      className={animate ? "animate-block-fade-in" : undefined}
      style={animate ? { animationDuration: `${fadeDuration}ms` } : undefined}
    >
      <MemoizedMarkdown content={content} />
    </div>
  );
}

// ============================================================================
// TYPEWRITER ANIMATION
// ============================================================================

interface TypewriterAnimatedBlockProps {
  block: MarkdownBlock;
  animate: boolean;
  charsPerBatch: number;
  batchInterval: number;
  onAnimationComplete?: () => void;
}

function TypewriterAnimatedBlock({
  block,
  animate,
  charsPerBatch,
  batchInterval,
  onAnimationComplete,
}: TypewriterAnimatedBlockProps) {
  const [displayLength, setDisplayLength] = useState(animate ? 0 : block.content.length);
  const animationStarted = useRef(false);
  const animationCompleteNotified = useRef(false);

  // Typewriter effect
  useEffect(() => {
    // If not animating, skip
    if (!animate) return;

    // Check if animation just completed
    if (displayLength >= block.content.length) {
      if (!animationCompleteNotified.current && onAnimationComplete) {
        animationCompleteNotified.current = true;
        onAnimationComplete();
      }
      return;
    }

    // Mark that animation has started
    animationStarted.current = true;

    // Animate in batches for performance
    const timer = setTimeout(() => {
      setDisplayLength((prev) => Math.min(prev + charsPerBatch, block.content.length));
    }, batchInterval);

    return () => clearTimeout(timer);
  }, [
    displayLength,
    block.content.length,
    animate,
    charsPerBatch,
    batchInterval,
    onAnimationComplete,
  ]);

  // Update display length if block content changes (e.g., trailing whitespace merged)
  useEffect(() => {
    if (!animate && displayLength !== block.content.length) {
      queueMicrotask(() => setDisplayLength(block.content.length));
    }
  }, [block.content.length, animate, displayLength]);

  // During animation: render partial content (MemoizedMarkdown handles LaTeX conversion)
  if (animate && displayLength < block.content.length) {
    const partialContent = block.content.slice(0, displayLength);
    return <MemoizedMarkdown content={partialContent} />;
  }

  // Animation complete: render full content
  return <MemoizedMarkdown content={block.content} />;
}

/**
 * Memoized AnimatedBlock to prevent re-renders of completed blocks.
 */
export const AnimatedBlock = memo(AnimatedBlockComponent, (prev, next) => {
  // Always re-render if animation status changes
  if (prev.animate !== next.animate) return false;
  // Always re-render if block content changes
  if (prev.block.content !== next.block.content) return false;
  // Always re-render if block type changes
  if (prev.block.type !== next.block.type) return false;
  // Always re-render if animation type changes
  if (prev.animationType !== next.animationType) return false;
  // Otherwise, skip re-render
  return true;
});
