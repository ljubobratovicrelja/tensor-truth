import { useMemo } from "react";
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
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { StreamToolPhase } from "@/api/types";

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
  const config = PHASE_ICON_MAP[phase.phase] ?? DEFAULT_CONFIG;
  const Icon = config.icon;

  // For the "generating" phase, overlay a random fun message (preserving legacy behavior).
  // For all other phases, use the server-provided message.
  const displayMessage = useMemo(() => {
    if (phase.phase === "generating") {
      return pickRandom(GENERATING_MESSAGES);
    }
    return phase.message;
    // Re-pick only when the phase itself changes
  }, [phase.phase, phase.message]);

  return (
    <div className="flex items-center gap-2 px-4 py-2">
      <Icon className={`text-muted-foreground h-4 w-4 ${config.animation}`} />
      <span className="text-muted-foreground text-sm">{displayMessage}</span>
    </div>
  );
}
