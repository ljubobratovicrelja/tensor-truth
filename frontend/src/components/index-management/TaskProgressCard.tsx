import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { useTaskProgress } from "@/hooks";
import type { TaskResponse } from "@/api/types";

interface TaskProgressCardProps {
  taskId: string;
  label: string;
  onComplete?: (task: TaskResponse) => void;
  onError?: (task: TaskResponse) => void;
}

export function TaskProgressCard({
  taskId,
  label,
  onComplete,
  onError,
}: TaskProgressCardProps) {
  const { data: task } = useTaskProgress(taskId, { onComplete, onError });

  if (!task) return null;

  const isTerminal = task.status === "completed" || task.status === "error";

  return (
    <div className="space-y-2 rounded-lg border p-3">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          {task.status === "error" ? (
            <AlertCircle className="h-3.5 w-3.5 text-red-500" />
          ) : isTerminal ? (
            <CheckCircle className="h-3.5 w-3.5 text-green-500" />
          ) : (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
          )}
          <span className="font-medium">{label}</span>
        </div>
        <span className="text-muted-foreground font-mono text-xs">{task.progress}%</span>
      </div>
      <div className="bg-secondary h-1.5 overflow-hidden rounded-full">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            task.status === "error"
              ? "bg-red-500"
              : isTerminal
                ? "bg-green-500"
                : "bg-blue-500"
          }`}
          style={{ width: `${task.progress}%` }}
        />
      </div>
      {task.stage && <p className="text-muted-foreground text-xs">{task.stage}</p>}
      {task.error && <p className="text-xs text-red-500">{task.error}</p>}
    </div>
  );
}
