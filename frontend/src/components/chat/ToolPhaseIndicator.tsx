import { useEffect, useRef, useMemo, useState } from "react";
import {
  Search,
  Brain,
  Sparkles,
  Loader2,
  Scale,
  Globe,
  Download,
  Database,
  Cpu,
  Wrench,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { StreamToolPhase } from "@/api/types";
import { useChatStore } from "@/stores";
import { cn } from "@/lib/utils";

interface ToolPhaseIndicatorProps {
  phase: StreamToolPhase;
}

interface PhaseConfig {
  icon: LucideIcon;
  animation: string;
}

// Extensible record mapping phase names to icon + animation.
// Unknown phases fall back to a generic spinner.
const PHASE_ICON_MAP: Record<string, PhaseConfig> = {
  loading_models: { icon: Loader2, animation: "animate-spin" },
  loading_model: { icon: Loader2, animation: "animate-spin" },
  retrieving: { icon: Search, animation: "animate-bounce" },
  searching: { icon: Search, animation: "animate-bounce" },
  reranking: { icon: Scale, animation: "animate-pulse" },
  ranking: { icon: Scale, animation: "animate-pulse" },
  thinking: { icon: Brain, animation: "animate-pulse" },
  generating: { icon: Sparkles, animation: "animate-pulse" },
  fetching: { icon: Download, animation: "animate-pulse" },
  fetched: { icon: Download, animation: "" },
  analyzing: { icon: Brain, animation: "animate-pulse" },
  web_search: { icon: Globe, animation: "animate-bounce" },
  indexing: { icon: Database, animation: "animate-pulse" },
  processing: { icon: Cpu, animation: "animate-pulse" },
  tool_call: { icon: Wrench, animation: "animate-pulse" },
};

const DEFAULT_CONFIG: PhaseConfig = {
  icon: Loader2,
  animation: "animate-spin",
};

// Funny messages for the "generating" phase, preserving the legacy personality
const GENERATING_MESSAGES = [
  "Crafting the perfect response...",
  "Weaving words together...",
  "Assembling syllables...",
  "Channeling Shakespeare...",
  "Typing with purpose...",
  "Constructing sentences...",
  "Summoning the muse...",
];

function pickRandom<T>(array: readonly T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

export function ToolPhaseIndicator({ phase }: ToolPhaseIndicatorProps) {
  const streamingReasoning = useChatStore((state) => state.streamingReasoning);
  const streamingThinking = useChatStore((state) => state.streamingThinking);
  const streamingContent = useChatStore((state) => state.streamingContent);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Determine intermediate content: show reasoning or thinking, but only
  // before response tokens start flowing (once content arrives, thinking
  // is displayed inside MessageItem's ThinkingBox instead).
  const intermediateContent = !streamingContent
    ? streamingReasoning || streamingThinking
    : "";
  const hasContent = !!intermediateContent;

  // When showing thinking (not reasoning), override title to "Reasoning"
  const isShowingThinking =
    !streamingContent && !streamingReasoning && !!streamingThinking;

  // Determine icon and animation from phase config
  const phaseConfig = PHASE_ICON_MAP[phase.phase] ?? DEFAULT_CONFIG;
  const TitleIcon = isShowingThinking ? Brain : phaseConfig.icon;
  const titleAnimation = isShowingThinking ? "animate-pulse" : phaseConfig.animation;

  // Determine title text
  const displayMessage = useMemo(() => {
    if (phase.phase === "generating") {
      return pickRandom(GENERATING_MESSAGES);
    }
    return phase.message;
  }, [phase.phase, phase.message]);
  const title = isShowingThinking ? "Reasoning" : displayMessage;

  // Auto-expand when content arrives, auto-collapse when cleared
  const [expanded, setExpanded] = useState(hasContent);
  useEffect(() => {
    setExpanded(hasContent);
  }, [hasContent]);

  // Auto-scroll content to bottom as it streams
  useEffect(() => {
    if (scrollRef.current && expanded) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [intermediateContent, expanded]);

  // No content — header-only box (no chevron, not expandable)
  if (!hasContent) {
    return (
      <div className="border-muted bg-muted/30 mb-3 rounded-lg border">
        <div className="flex items-center gap-2 px-3 py-2">
          <TitleIcon className={`text-muted-foreground h-4 w-4 ${titleAnimation}`} />
          <span className="text-muted-foreground text-sm font-medium">{title}</span>
        </div>
      </div>
    );
  }

  // Has content — collapsible box with streaming text
  return (
    <div className="border-muted bg-muted/30 mb-3 rounded-lg border">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="border-muted hover:bg-muted/50 flex w-full items-center justify-between gap-2 border-b px-3 py-2 transition-colors"
      >
        <div className="flex items-center gap-2">
          <TitleIcon className={`text-muted-foreground h-4 w-4 ${titleAnimation}`} />
          <span className="text-muted-foreground text-sm font-medium">{title}</span>
        </div>
        {expanded ? (
          <ChevronUp className="text-muted-foreground h-4 w-4" />
        ) : (
          <ChevronDown className="text-muted-foreground h-4 w-4" />
        )}
      </button>
      <div
        ref={scrollRef}
        className={cn(
          "overflow-y-auto px-3 py-2 text-sm transition-all",
          expanded ? "max-h-48" : "max-h-0 overflow-hidden py-0"
        )}
      >
        <pre className="text-muted-foreground font-mono whitespace-pre-wrap">
          {intermediateContent}
        </pre>
      </div>
    </div>
  );
}
