import { useMemo } from "react";
import { ChevronUp, Layers, Loader2, Minimize2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useSessionContext } from "@/hooks/useSessionContext";
import type { ContextCategory } from "@/api/sessions";

function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (value >= 10_000) return `${Math.round(value / 1000)}K`;
  if (value >= 1_000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  return `${value}`;
}

/** Colour the meter by how much of the window is used. */
function usageTone(percent: number): string {
  if (percent >= 90) return "bg-red-500";
  if (percent >= 70) return "bg-amber-500";
  if (percent >= 40) return "bg-yellow-500";
  return "bg-emerald-500";
}

const CATEGORY_COLORS: Record<string, string> = {
  system: "bg-sky-500",
  tools: "bg-violet-500",
  modules: "bg-teal-500",
  documents: "bg-orange-500",
  history: "bg-emerald-500",
  prompt: "bg-slate-400",
};

interface ContextMeterProps {
  sessionId?: string;
  /** Called when the user asks to compact the conversation. */
  onCompact?: () => void;
  compactPending?: boolean;
  className?: string;
}

/**
 * Status-bar context meter: total tokens against the window, with a popover
 * breaking the total down by category (system, tools, modules, documents,
 * history, current message).
 */
export function ContextMeter({
  sessionId,
  onCompact,
  compactPending = false,
  className,
}: ContextMeterProps) {
  const { data, isLoading } = useSessionContext(sessionId);
  const categories: ContextCategory[] = useMemo(
    () => (data?.categories ?? []).filter((c) => c.chars > 0),
    [data]
  );

  if (!sessionId) return null;

  const total = data?.total_tokens ?? 0;
  const window = data?.context_window ?? 0;
  const percent = data?.used_percent ?? 0;
  const shownPercent = percent > 0 && percent < 1 ? 0.1 : percent;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "text-muted-foreground hover:text-foreground hover:bg-muted/60 flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-[11px] transition-colors",
            className
          )}
          title="Context usage by category"
          data-testid="context-meter"
        >
          <Layers className="h-3 w-3 shrink-0" />
          {isLoading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <>
              <div className="bg-secondary h-1.5 w-24 shrink-0 overflow-hidden rounded-full">
                <div
                  className={cn("h-full transition-all", usageTone(percent))}
                  style={{ width: `${Math.min(100, shownPercent)}%` }}
                />
              </div>
              <span className="font-medium">
                {formatTokens(total)}
                {window > 0 ? ` / ${formatTokens(window)}` : ""}
              </span>
              {window > 0 && <span>({percent.toFixed(percent < 10 ? 1 : 0)}%)</span>}
              {!data?.measured && (
                <span className="text-muted-foreground/70 italic">estimate</span>
              )}
            </>
          )}
          <ChevronUp className="ml-auto h-3 w-3 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-3" align="start" side="top">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-sm font-semibold">Context window</p>
          <Badge variant="outline" className="text-[10px]">
            {data?.measured ? "measured" : "estimate"}
          </Badge>
        </div>

        <div className="text-muted-foreground mb-2 text-xs">
          {formatTokens(total)} tokens used
          {window > 0 && (
            <>
              {" "}
              of {formatTokens(window)} ({percent.toFixed(1)}%)
            </>
          )}
        </div>

        <div className="space-y-2">
          {categories.length === 0 && (
            <p className="text-muted-foreground text-xs">
              Nothing measured yet — send a message to see the breakdown.
            </p>
          )}
          {categories.map((category) => (
            <div key={category.id} className="space-y-1" title={category.hint}>
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className="flex items-center gap-1.5">
                  <span
                    className={cn(
                      "h-2 w-2 rounded-sm",
                      CATEGORY_COLORS[category.id] ?? "bg-slate-400"
                    )}
                  />
                  {category.label}
                </span>
                <span className="text-muted-foreground tabular-nums">
                  {formatTokens(category.tokens)}
                </span>
              </div>
              <div className="bg-secondary h-1 w-full overflow-hidden rounded-full">
                <div
                  className={cn("h-full", CATEGORY_COLORS[category.id] ?? "bg-slate-400")}
                  style={{ width: `${Math.min(100, category.share * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {data && data.messages_total > 0 && (
          <p className="text-muted-foreground mt-3 text-[11px]">
            {data.messages_included} of {data.messages_total} messages included
          </p>
        )}

        {onCompact && (
          <Button
            variant="outline"
            size="sm"
            className="mt-3 w-full gap-2"
            onClick={onCompact}
            disabled={compactPending}
          >
            {compactPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Minimize2 className="h-3.5 w-3.5" />
            )}
            Compact conversation
          </Button>
        )}
      </PopoverContent>
    </Popover>
  );
}
