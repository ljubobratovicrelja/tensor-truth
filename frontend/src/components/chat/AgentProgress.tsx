import { Search, Download, Brain, CheckCircle } from "lucide-react";
import type { StreamAgentProgress, AgentPhase } from "@/api/types";

interface AgentProgressProps {
  progress: StreamAgentProgress | null;
}

const PHASE_CONFIG: Record<
  AgentPhase,
  {
    icon: typeof Search;
    animation: string;
  }
> = {
  searching: {
    icon: Search,
    animation: "animate-bounce",
  },
  fetching: {
    icon: Download,
    animation: "animate-pulse",
  },
  summarizing: {
    icon: Brain,
    animation: "animate-pulse",
  },
  complete: {
    icon: CheckCircle,
    animation: "",
  },
};

export function AgentProgress({ progress }: AgentProgressProps) {
  if (!progress) return null;

  const phase = progress.phase as AgentPhase;
  const config = PHASE_CONFIG[phase];
  const Icon = config?.icon || Search;
  const animation = config?.animation ?? "animate-pulse";

  // Build status text based on phase
  let statusText = progress.message;

  // Add progress counter for fetching phase
  if (
    phase === "fetching" &&
    progress.pages_target &&
    progress.pages_fetched !== undefined
  ) {
    statusText = `${progress.message} (${progress.pages_fetched}/${progress.pages_target})`;
  }

  // Add hit count for searching phase
  if (phase === "searching" && progress.search_hits !== undefined) {
    statusText = `${progress.message} - ${progress.search_hits} results`;
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2">
      <Icon className={`text-muted-foreground h-4 w-4 ${animation}`} />
      <span className="text-muted-foreground text-sm">{statusText}</span>
    </div>
  );
}
