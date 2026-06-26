import { useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { listTasks } from "@/api/tasks";
import { TaskProgressCard } from "./TaskProgressCard";
import { useQueryClient } from "@tanstack/react-query";

export function ActiveTasks() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: QUERY_KEYS.tasks,
    queryFn: () => listTasks(),
    refetchInterval: 3000,
  });

  const relevantTasks = (data?.tasks ?? []).filter(
    (t) => t.task_type === "fetch_source" || t.task_type === "build_index"
  );

  const activeTasks = relevantTasks.filter(
    (t) => t.status === "pending" || t.status === "running"
  );
  const recentTasks = relevantTasks
    .filter((t) => t.status === "completed" || t.status === "error")
    .slice(0, 10);

  return (
    <div className="space-y-4">
      {activeTasks.length === 0 && recentTasks.length === 0 && (
        <div className="text-muted-foreground py-8 text-center text-sm">
          No active or recent index tasks.
        </div>
      )}

      {activeTasks.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Active</h3>
          {activeTasks.map((t) => (
            <TaskProgressCard
              key={t.task_id}
              taskId={t.task_id}
              label={
                t.metadata.name
                  ? `${t.task_type === "fetch_source" ? "Fetch" : "Build"}: ${(t.metadata.name as string) || (t.metadata.module_name as string)}`
                  : t.task_type
              }
              onComplete={() => {
                qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
                qc.invalidateQueries({ queryKey: QUERY_KEYS.modules });
              }}
            />
          ))}
        </div>
      )}

      {recentTasks.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Recent</h3>
          {recentTasks.map((t) => (
            <TaskProgressCard
              key={t.task_id}
              taskId={t.task_id}
              label={
                t.metadata.name
                  ? `${t.task_type === "fetch_source" ? "Fetch" : "Build"}: ${(t.metadata.name as string) || (t.metadata.module_name as string)}`
                  : t.task_type
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
