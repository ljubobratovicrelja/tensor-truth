import { FileText, Link, Check, Loader2, AlertCircle, BookOpen } from "lucide-react";
import type { ProjectResponse } from "@/api/types";
import { cn } from "@/lib/utils";

interface ProjectKnowledgePanelProps {
  project: ProjectResponse;
}

function ModuleStatusIndicator({ status }: { status: string }) {
  switch (status) {
    case "indexed":
      return <Check className="h-3.5 w-3.5 text-green-500" />;
    case "building":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-yellow-500" />;
    case "error":
      return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
    default:
      return <div className="h-3.5 w-3.5 rounded-full bg-gray-400" />;
  }
}

function DocumentTypeIcon({ type }: { type: string }) {
  if (type === "url") {
    return <Link className="text-muted-foreground h-3.5 w-3.5 shrink-0" />;
  }
  return <FileText className="text-muted-foreground h-3.5 w-3.5 shrink-0" />;
}

export function ProjectKnowledgePanel({ project }: ProjectKnowledgePanelProps) {
  const moduleEntries = Object.entries(project.catalog_modules);
  const documents = project.documents;
  const hasContent = moduleEntries.length > 0 || documents.length > 0;

  if (!hasContent) {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Knowledge</h3>
        <div className="text-muted-foreground flex flex-col items-center gap-2 py-6 text-center">
          <BookOpen className="h-8 w-8 opacity-50" />
          <p className="text-xs">No knowledge sources added yet</p>
          <p className="text-xs opacity-70">
            Add catalog modules or upload documents via the API to give this project
            context.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">Knowledge</h3>

      {/* Catalog Modules */}
      {moduleEntries.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-muted-foreground text-xs font-medium">Catalog Modules</p>
          <ul className="space-y-1">
            {moduleEntries.map(([name, info]) => (
              <li
                key={name}
                className="flex items-center gap-2 rounded px-1.5 py-1 text-xs"
              >
                <ModuleStatusIndicator status={info.status} />
                <span className="truncate">{name}</span>
                <span
                  className={cn(
                    "ml-auto shrink-0 text-[10px]",
                    info.status === "indexed" && "text-green-600 dark:text-green-400",
                    info.status === "building" && "text-yellow-600 dark:text-yellow-400",
                    info.status === "error" && "text-red-600 dark:text-red-400"
                  )}
                >
                  {info.status}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Documents */}
      {documents.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-muted-foreground text-xs font-medium">Documents</p>
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li
                key={doc.doc_id}
                className="flex items-center gap-2 rounded px-1.5 py-1 text-xs"
              >
                <DocumentTypeIcon type={doc.type} />
                <span className="truncate">{doc.filename || doc.url || doc.doc_id}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
