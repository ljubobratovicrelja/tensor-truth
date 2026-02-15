import { useState } from "react";
import { Plus, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useModules, useAddCatalogModule } from "@/hooks";
import { generateDisplayName } from "@/lib/moduleUtils";
import { toast } from "sonner";

interface CatalogModuleAddPickerProps {
  projectId: string;
  existingModules: string[];
}

export function CatalogModuleAddPicker({
  projectId,
  existingModules,
}: CatalogModuleAddPickerProps) {
  const [open, setOpen] = useState(false);
  const [addingModule, setAddingModule] = useState<string | null>(null);
  const { data: modulesData, isLoading } = useModules();
  const addModule = useAddCatalogModule();

  const existingSet = new Set(existingModules);
  const availableModules = (modulesData?.modules ?? []).filter(
    (m) => !existingSet.has(m.name)
  );

  const handleAdd = async (moduleName: string) => {
    setAddingModule(moduleName);
    try {
      await addModule.mutateAsync({ projectId, moduleName });
      toast.success(`Adding module: ${generateDisplayName(moduleName)}`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Add failed";
      toast.error(`Failed to add module: ${msg}`);
    } finally {
      setAddingModule(null);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          title="Add catalog module"
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" side="bottom" className="w-64 p-0">
        <div className="border-b px-3 py-2">
          <span className="text-sm font-medium">Add Module</span>
        </div>
        <div className="max-h-60 overflow-y-auto p-1">
          {isLoading ? (
            <div className="text-muted-foreground flex items-center justify-center py-4 text-sm">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading...
            </div>
          ) : availableModules.length === 0 ? (
            <div className="text-muted-foreground py-4 text-center text-sm">
              No additional modules available.
            </div>
          ) : (
            availableModules.map((module) => {
              const isAdding = addingModule === module.name;
              return (
                <button
                  key={module.name}
                  onClick={() => handleAdd(module.name)}
                  disabled={isAdding || addModule.isPending}
                  className="hover:bg-muted/50 flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors disabled:opacity-50"
                >
                  {isAdding ? (
                    <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                  ) : (
                    <Plus className="text-muted-foreground h-3.5 w-3.5 shrink-0" />
                  )}
                  <span className="truncate">
                    {module.display_name || generateDisplayName(module.name)}
                  </span>
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
