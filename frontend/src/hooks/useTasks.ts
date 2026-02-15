import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { getTask } from "@/api/tasks";
import type { TaskResponse } from "@/api/types";

interface UseTaskProgressOptions {
  onComplete?: (task: TaskResponse) => void;
  onError?: (task: TaskResponse) => void;
}

export function useTaskProgress(
  taskId: string | null | undefined,
  options?: UseTaskProgressOptions
) {
  // Ref guards to fire callbacks exactly once
  const completeFiredRef = useRef(false);
  const errorFiredRef = useRef(false);

  // Keep latest callback values in refs
  const onCompleteRef = useRef(options?.onComplete);
  const onErrorRef = useRef(options?.onError);

  // Sync refs with latest callbacks
  useEffect(() => {
    onCompleteRef.current = options?.onComplete;
  }, [options?.onComplete]);

  useEffect(() => {
    onErrorRef.current = options?.onError;
  }, [options?.onError]);

  // Reset guards when taskId changes
  useEffect(() => {
    completeFiredRef.current = false;
    errorFiredRef.current = false;
  }, [taskId]);

  const query = useQuery({
    queryKey: taskId ? QUERY_KEYS.task(taskId) : ["tasks", "none"],
    queryFn: () => getTask(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1500;
      if (data.status === "completed" || data.status === "error") return false;
      return 1500;
    },
  });

  // Fire callbacks on terminal states
  useEffect(() => {
    const data = query.data;
    if (!data) return;

    if (data.status === "completed" && !completeFiredRef.current) {
      completeFiredRef.current = true;
      onCompleteRef.current?.(data);
    }

    if (data.status === "error" && !errorFiredRef.current) {
      errorFiredRef.current = true;
      onErrorRef.current?.(data);
    }
  }, [query.data]);

  return query;
}
