import { useMemo } from "react";
import { Search, Brain, Sparkles, Loader2, Scale } from "lucide-react";
import { useChatStore } from "@/stores";
import { ToolPhaseIndicator } from "./ToolPhaseIndicator";
import type { PipelineStatus } from "@/stores/chatStore";

interface StreamingIndicatorProps {
  status?: PipelineStatus;
}

const STATUS_CONFIG = {
  loading_models: {
    icon: Loader2,
    labels: [
      "Waking up the AI...",
      "Booting up neural networks...",
      "Spinning up the hamsters...",
      "Initializing brain cells...",
      "Loading digital consciousness...",
      "Preparing the think tank...",
    ],
  },
  retrieving: {
    icon: Search,
    labels: [
      "Hunting for clues...",
      "Digging through the archives...",
      "Scouring the knowledge base...",
      "Following the paper trail...",
      "Raiding the library...",
      "Mining for nuggets of wisdom...",
      "Searching high and low...",
    ],
  },
  reranking: {
    icon: Scale,
    labels: [
      "Separating wheat from chaff...",
      "Ranking the suspects...",
      "Weighing the evidence...",
      "Judging the contenders...",
      "Sorting the good from the great...",
      "Playing favorites with facts...",
      "Assembling the dream team...",
    ],
  },
  thinking: {
    icon: Brain,
    labels: [
      "Deep in thought...",
      "Pondering the mysteries...",
      "Engaging big brain mode...",
      "Connecting the dots...",
      "Having an internal debate...",
      "Philosophizing intensely...",
      "Computing the cosmos...",
    ],
  },
  generating: {
    icon: Sparkles,
    labels: [
      "Crafting the perfect response...",
      "Weaving words together...",
      "Assembling syllables...",
      "Channeling Shakespeare...",
      "Typing with purpose...",
      "Constructing sentences...",
      "Summoning the muse...",
    ],
  },
} as const;

function pickRandom<T>(array: readonly T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

export function StreamingIndicator({ status }: StreamingIndicatorProps) {
  const toolPhase = useChatStore((state) => state.toolPhase);

  // Pick a random label when status changes, but keep it stable while status is the same
  const label = useMemo(() => {
    if (!status) return "Processing...";
    const config = STATUS_CONFIG[status];
    return config ? pickRandom(config.labels) : "Processing...";
  }, [status]);

  // When toolPhase is present, delegate to ToolPhaseIndicator
  if (toolPhase) {
    return <ToolPhaseIndicator phase={toolPhase} />;
  }

  // Legacy status rendering (backward compat with old pipeline status messages)
  const config = status ? STATUS_CONFIG[status] : null;
  const Icon = config?.icon;

  // Different animation per icon type
  const getIconAnimation = () => {
    if (!status) return "animate-pulse";
    switch (status) {
      case "loading_models":
        return "animate-spin";
      case "retrieving":
        return "animate-bounce";
      case "reranking":
        return "animate-pulse";
      case "thinking":
        return "animate-pulse";
      case "generating":
        return "animate-pulse";
      default:
        return "animate-pulse";
    }
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2">
      {Icon && <Icon className={`text-muted-foreground h-4 w-4 ${getIconAnimation()}`} />}
      <span className="text-muted-foreground text-sm">{label}</span>
    </div>
  );
}
