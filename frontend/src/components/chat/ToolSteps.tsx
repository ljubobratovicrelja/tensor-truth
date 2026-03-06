import { useState, useCallback } from "react";
import { ChevronDown, ChevronUp, Wrench, Check, X, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ToolStep } from "@/api/types";

export type ToolStepWithStatus = ToolStep & {
  status: "calling" | "completed" | "failed";
};

interface ToolStepCardProps {
  step: ToolStepWithStatus;
  index: number;
}

function ToolStepCard({ step, index }: ToolStepCardProps) {
  const [expanded, setExpanded] = useState(false);

  const paramsSummary = Object.entries(step.params)
    .map(([k, v]) => {
      const val = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}: ${val.length > 40 ? val.slice(0, 40) + "..." : val}`;
    })
    .join(", ");

  return (
    <div className="border-muted-foreground/10 border-b last:border-b-0">
      <button
        onClick={() => step.output && setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs",
          step.output && "hover:bg-muted/50 cursor-pointer"
        )}
      >
        <span className="text-muted-foreground shrink-0 font-mono text-[10px]">
          {index + 1}
        </span>
        <Wrench className="text-muted-foreground h-3 w-3 shrink-0" />
        <span className="min-w-0 flex-1 truncate">
          <span className="font-medium">{step.tool}</span>
          {paramsSummary && (
            <span className="text-muted-foreground ml-1.5">{paramsSummary}</span>
          )}
        </span>
        {step.status === "calling" && (
          <Badge variant="secondary" className="shrink-0 gap-1 text-[10px]">
            <Loader2 className="h-2.5 w-2.5 animate-spin" />
            Calling
          </Badge>
        )}
        {step.status === "completed" && (
          <Badge
            variant="secondary"
            className="shrink-0 gap-1 border-green-500/20 bg-green-500/10 text-[10px] text-green-700 dark:text-green-400"
          >
            <Check className="h-2.5 w-2.5" />
            Done
          </Badge>
        )}
        {step.status === "failed" && (
          <Badge variant="destructive" className="shrink-0 gap-1 text-[10px]">
            <X className="h-2.5 w-2.5" />
            Failed
          </Badge>
        )}
        {step.output &&
          (expanded ? (
            <ChevronUp className="text-muted-foreground h-3 w-3 shrink-0" />
          ) : (
            <ChevronDown className="text-muted-foreground h-3 w-3 shrink-0" />
          ))}
      </button>
      {expanded && step.output && (
        <div className="bg-background/50 max-h-48 overflow-auto px-2 py-1.5">
          <pre className="text-muted-foreground text-[11px] leading-relaxed break-words whitespace-pre-wrap">
            {step.output}
          </pre>
        </div>
      )}
    </div>
  );
}

interface ToolStepsProps {
  steps: ToolStepWithStatus[];
  defaultOpen?: boolean;
}

export function ToolSteps({ steps, defaultOpen = false }: ToolStepsProps) {
  // userToggle: null = follow defaultOpen, true/false = user override
  const [userToggle, setUserToggle] = useState<boolean | null>(null);
  const collapsed = userToggle !== null ? !userToggle : !defaultOpen;
  const toggle = useCallback(
    () => setUserToggle((prev) => !(prev ?? defaultOpen)),
    [defaultOpen]
  );

  if (steps.length === 0) return null;

  const completed = steps.filter((s) => s.status === "completed").length;
  const failed = steps.filter((s) => s.status === "failed").length;
  const calling = steps.filter((s) => s.status === "calling").length;

  return (
    <div className="mt-2 border-t pt-2">
      <button
        onClick={toggle}
        className="text-muted-foreground hover:text-foreground mb-1.5 flex w-full items-center justify-between gap-2 text-xs transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium tracking-wide uppercase">
            Steps ({steps.length})
          </span>
          <span className="text-muted-foreground/70 font-normal tracking-normal normal-case">
            {completed > 0 && <>{completed} completed</>}
            {failed > 0 && <> | {failed} failed</>}
            {calling > 0 && <> | {calling} running</>}
          </span>
        </div>
        {collapsed ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <ChevronUp className="h-3.5 w-3.5 shrink-0" />
        )}
      </button>
      <div
        className={cn(
          "overflow-hidden transition-all duration-200",
          collapsed ? "max-h-0" : "max-h-[2000px]"
        )}
      >
        {steps.map((step, index) => (
          <ToolStepCard key={index} step={step} index={index} />
        ))}
      </div>
    </div>
  );
}
