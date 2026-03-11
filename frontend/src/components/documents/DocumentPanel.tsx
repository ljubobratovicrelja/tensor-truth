import { useState } from "react";
import type { ScopeType } from "@/api/types";
import { DocumentList } from "./DocumentList";

interface DocumentPanelProps {
  scopeId: string;
  scopeType: ScopeType;
  /** External build task state (lifted for spinner on dialog trigger). */
  buildTaskId?: string | null;
  onBuildTaskIdChange?: (id: string | null) => void;
}

export function DocumentPanel({
  scopeId,
  scopeType,
  buildTaskId: externalBuildTaskId,
  onBuildTaskIdChange: externalOnChange,
}: DocumentPanelProps) {
  // Internal state used when no external control is provided (e.g. sidebar)
  const [internalBuildTaskId, setInternalBuildTaskId] = useState<string | null>(null);

  const buildTaskId = externalOnChange
    ? (externalBuildTaskId ?? null)
    : internalBuildTaskId;
  const onBuildTaskIdChange = externalOnChange ?? setInternalBuildTaskId;

  return (
    <div className="space-y-4">
      <DocumentList
        scopeId={scopeId}
        scopeType={scopeType}
        buildTaskId={buildTaskId}
        onBuildTaskIdChange={onBuildTaskIdChange}
      />
    </div>
  );
}
