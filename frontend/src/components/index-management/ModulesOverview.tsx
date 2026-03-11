import { useState, useMemo } from "react";
import { Search, Library, FileText, BookOpen } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { useModulesOverview } from "@/hooks";
import type { ModuleOverview } from "@/api/types";
import { ModuleCard } from "./ModuleCard";

const SOURCE_TYPE_CONFIG: Record<string, { label: string; icon: React.ReactNode }> = {
  libraries: { label: "Libraries", icon: <Library className="h-4 w-4" /> },
  papers: { label: "Papers", icon: <FileText className="h-4 w-4" /> },
  books: { label: "Books", icon: <BookOpen className="h-4 w-4" /> },
};

interface ColumnProps {
  type: string;
  modules: ModuleOverview[];
  embeddingModelId: string;
}

function ModuleColumn({ type, modules, embeddingModelId }: ColumnProps) {
  const config = SOURCE_TYPE_CONFIG[type] || { label: type, icon: null };

  return (
    <div className="min-w-0">
      <div className="text-muted-foreground mb-3 flex items-center gap-2 text-sm font-medium">
        {config.icon}
        <span>{config.label}</span>
        <Badge variant="secondary" className="ml-1 px-1.5 py-0 text-[10px]">
          {modules.length}
        </Badge>
      </div>
      <div className="space-y-2">
        {modules.map((m) => (
          <ModuleCard key={m.name} module={m} embeddingModelId={embeddingModelId} />
        ))}
      </div>
    </div>
  );
}

export function ModulesOverview() {
  const [search, setSearch] = useState("");
  const { data, isLoading } = useModulesOverview();

  // Determine embedding model ID from first module's index metadata, fallback to bge-m3
  const embeddingModelId = useMemo(() => {
    if (!data?.modules) return "bge-m3";
    for (const m of data.modules) {
      if (m.index_metadata?.embedding_model_id) {
        return m.index_metadata.embedding_model_id as string;
      }
    }
    return "bge-m3";
  }, [data]);

  const grouped = useMemo(() => {
    if (!data?.modules) return {};
    const filtered = search
      ? data.modules.filter(
          (m) =>
            m.name.toLowerCase().includes(search.toLowerCase()) ||
            m.display_name.toLowerCase().includes(search.toLowerCase())
        )
      : data.modules;

    const groups: Record<string, ModuleOverview[]> = {};
    for (const m of filtered) {
      const type = m.source_type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(m);
    }
    return groups;
  }, [data, search]);

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-16 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  const totalModules = data?.modules?.length ?? 0;
  const activeTypes = ["libraries", "papers", "books"].filter(
    (type) => grouped[type]?.length
  );

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="text-muted-foreground absolute top-2.5 left-2.5 h-4 w-4" />
        <Input
          placeholder={`Search ${totalModules} modules...`}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {totalModules === 0 ? (
        <div className="text-muted-foreground py-8 text-center text-sm">
          No modules configured. Add a source to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {activeTypes.map((type) => (
            <ModuleColumn
              key={type}
              type={type}
              modules={grouped[type]}
              embeddingModelId={embeddingModelId}
            />
          ))}
        </div>
      )}
    </div>
  );
}
