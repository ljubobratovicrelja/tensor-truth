import { Search, Brain, Sparkles, Loader2 } from "lucide-react";
import type { PipelineStatus } from "@/stores/chatStore";

interface StreamingIndicatorProps {
  status?: PipelineStatus;
}

const STATUS_CONFIG = {
  loading_models: {
    icon: Loader2,
    label: "Loading models...",
  },
  retrieving: {
    icon: Search,
    label: "Searching documents...",
  },
  thinking: {
    icon: Brain,
    label: "Reasoning...",
  },
  generating: {
    icon: Sparkles,
    label: "Generating response...",
  },
} as const;

export function StreamingIndicator({ status }: StreamingIndicatorProps) {
  const config = status ? STATUS_CONFIG[status] : null;
  const Icon = config?.icon;

  return (
    <div className="flex items-center gap-2 px-4 py-2">
      <div className="flex gap-1">
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full [animation-delay:-0.3s]" />
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full [animation-delay:-0.15s]" />
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full" />
      </div>
      {Icon && <Icon className="text-muted-foreground h-4 w-4" />}
      <span className="text-muted-foreground text-sm">
        {config?.label ?? "Processing..."}
      </span>
    </div>
  );
}
