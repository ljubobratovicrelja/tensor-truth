import { useState } from "react";
import { Loader2, Power } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface ModelStatusDotProps {
  status: "loaded" | "loading" | "unloaded" | null | undefined;
  isActionInFlight?: boolean;
  onLoad?: () => void;
  onUnload?: () => void;
}

export function ModelStatusDot({
  status,
  isActionInFlight,
  onLoad,
  onUnload,
}: ModelStatusDotProps) {
  const [hovered, setHovered] = useState(false);

  // Don't render for openai_compatible (null/undefined status)
  if (status == null) return null;

  const isLoading = status === "loading" || isActionInFlight;

  if (isLoading) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex items-center justify-center p-0.5">
            <Loader2 className="text-muted-foreground h-3 w-3 animate-spin" />
          </span>
        </TooltipTrigger>
        <TooltipContent side="right">Loading model...</TooltipContent>
      </Tooltip>
    );
  }

  if (status === "loaded") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className="inline-flex cursor-pointer items-center justify-center p-0.5"
            onPointerDown={(e) => {
              e.stopPropagation();
              e.preventDefault();
              onUnload?.();
            }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
          >
            {hovered ? (
              <Power className="text-muted-foreground h-3 w-3" />
            ) : (
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-500" />
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent side="right">
          {hovered ? "Click to unload" : "Model loaded"}
        </TooltipContent>
      </Tooltip>
    );
  }

  // unloaded
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-flex cursor-pointer items-center justify-center p-0.5"
          onPointerDown={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onLoad?.();
          }}
        >
          <span className="bg-muted-foreground/40 inline-block h-2.5 w-2.5 rounded-full" />
        </span>
      </TooltipTrigger>
      <TooltipContent side="right">Click to load model into memory</TooltipContent>
    </Tooltip>
  );
}
