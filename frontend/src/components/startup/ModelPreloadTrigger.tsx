/**
 * Fires the idempotent RAG model preload once per app mount, so the embedding
 * encoder and reranker warm in the backend background while the UI is usable.
 */

import { useEffect, useRef } from "react";
import { useTriggerModelPreload } from "@/hooks";

export function ModelPreloadTrigger() {
  const { mutate } = useTriggerModelPreload();
  const fired = useRef(false);

  useEffect(() => {
    if (fired.current) return;
    fired.current = true;
    mutate();
  }, [mutate]);

  return null;
}
