import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { getRerankers, addReranker, removeReranker } from "@/api/rerankers";

export function useRerankers() {
  return useQuery({
    queryKey: QUERY_KEYS.rerankers,
    queryFn: getRerankers,
  });
}

export function useAddReranker() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (model: string) => addReranker(model),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.rerankers });
    },
  });
}

export function useRemoveReranker() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (model: string) => removeReranker(model),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.rerankers });
    },
  });
}
