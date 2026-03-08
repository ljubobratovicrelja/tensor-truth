import { useState, useCallback } from "react";
import { ChevronDown, ChevronUp, Wrench, Check, X, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolStep } from "@/api/types";

export type ToolStepWithStatus = ToolStep & {
  status: "calling" | "completed" | "failed";
};

interface ToolStepCardProps {
  step: ToolStepWithStatus;
  isLast: boolean;
}

function StatusDot({ status }: { status: ToolStepWithStatus["status"] }) {
  if (status === "calling") {
    return (
      <span className="relative flex h-4 w-4 items-center justify-center">
        <span className="bg-primary/40 absolute h-4 w-4 animate-ping rounded-full" />
        <Loader2 className="text-primary relative h-3 w-3 animate-spin" />
      </span>
    );
  }
  if (status === "completed") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-green-500/20">
        <Check className="h-2.5 w-2.5 text-green-500" />
      </span>
    );
  }
  return (
    <span className="flex h-4 w-4 items-center justify-center rounded-full bg-red-500/20">
      <X className="h-2.5 w-2.5 text-red-500" />
    </span>
  );
}

function ToolStepCard({ step, isLast }: ToolStepCardProps) {
  const [expanded, setExpanded] = useState(false);

  const paramsSummary = Object.entries(step.params)
    .map(([k, v]) => {
      const val = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}: ${val.length > 40 ? val.slice(0, 40) + "..." : val}`;
    })
    .join(", ");

  return (
    <div className="relative flex gap-3">
      {/* Timeline rail — dot + connecting line */}
      <div className="flex flex-col items-center">
        <StatusDot status={step.status} />
        {!isLast && (
          <div
            className={cn(
              "w-px flex-1",
              step.status === "calling"
                ? "bg-primary/30"
                : step.status === "completed"
                  ? "bg-green-500/20"
                  : "bg-red-500/20"
            )}
          />
        )}
      </div>

      {/* Step content */}
      <div className={cn("min-w-0 flex-1", !isLast ? "pb-3" : "pb-0.5")}>
        <button
          onClick={() => step.output && setExpanded(!expanded)}
          className={cn(
            "flex w-full items-center gap-1.5 text-left text-xs",
            step.output && "hover:text-foreground cursor-pointer"
          )}
        >
          <Wrench className="text-muted-foreground h-3 w-3 shrink-0" />
          <span className="min-w-0 flex-1 truncate">
            <span className="font-medium">{step.tool}</span>
            {paramsSummary && (
              <span className="text-muted-foreground ml-1.5">{paramsSummary}</span>
            )}
          </span>
          {step.output && (
            <span className="text-muted-foreground shrink-0">
              {expanded ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </span>
          )}
        </button>
        {expanded && step.output && (
          <div className="bg-muted/40 mt-1.5 max-h-48 overflow-auto rounded-lg px-3 py-2">
            <pre className="text-muted-foreground text-[11px] leading-relaxed break-words whitespace-pre-wrap">
              {step.output}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

interface ToolStepsProps {
  steps: ToolStepWithStatus[];
  defaultOpen?: boolean;
}

export function ToolSteps({ steps, defaultOpen = false }: ToolStepsProps) {
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
        className="text-muted-foreground hover:text-foreground mb-2 flex w-full items-center justify-between gap-2 text-xs transition-colors"
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
        <div className="pt-0.5 pl-1">
          {steps.map((step, index) => (
            <ToolStepCard key={index} step={step} isLast={index === steps.length - 1} />
          ))}
        </div>
      </div>
    </div>
  );
}
