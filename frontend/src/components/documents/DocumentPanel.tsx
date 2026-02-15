import type { ScopeType } from "@/api/types";
import { DocumentList } from "./DocumentList";

interface DocumentPanelProps {
  scopeId: string;
  scopeType: ScopeType;
}

export function DocumentPanel({ scopeId, scopeType }: DocumentPanelProps) {
  return (
    <div className="space-y-4">
      <DocumentList scopeId={scopeId} scopeType={scopeType} />
    </div>
  );
}
