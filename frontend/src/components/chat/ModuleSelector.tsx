import { useState, useEffect, useMemo } from "react";
import { Database, Check, Book, FileText, Package, Folder } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { useModules } from "@/hooks";
import type { ModuleInfo } from "@/api/types";

interface ModuleSelectorProps {
  selectedModules: string[];
  onModulesChange: (modules: string[]) => void;
  disabled?: boolean;
}

export function ModuleSelector({
  selectedModules,
  onModulesChange,
  disabled = false,
}: ModuleSelectorProps) {
  const [open, setOpen] = useState(false);
  const [localSelection, setLocalSelection] = useState<string[]>(selectedModules);
  const { data: modulesData, isLoading } = useModules();

  // Sync local selection when prop changes
  useEffect(() => {
    setLocalSelection(selectedModules);
  }, [selectedModules]);

  const hasChanges =
    localSelection.length !== selectedModules.length ||
    localSelection.some((m) => !selectedModules.includes(m));

  const handleToggle = (moduleName: string) => {
    setLocalSelection((prev) =>
      prev.includes(moduleName)
        ? prev.filter((m) => m !== moduleName)
        : [...prev, moduleName]
    );
  };

  const handleApply = () => {
    onModulesChange(localSelection);
    setOpen(false);
  };

  const handleCancel = () => {
    setLocalSelection(selectedModules);
    setOpen(false);
  };

  const handleClearAll = () => {
    setLocalSelection([]);
  };

  const modules = modulesData?.modules ?? [];
  const selectedCount = localSelection.length;

  // Group modules by doc_type
  const groupedModules = useMemo(() => {
    const groups: Record<string, ModuleInfo[]> = {};
    for (const module of modulesData?.modules ?? []) {
      const type = module.doc_type || "unknown";
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(module);
    }
    return groups;
  }, [modulesData?.modules]);

  // Type display config
  const typeConfig: Record<string, { label: string; icon: typeof Book }> = {
    book: { label: "Books", icon: Book },
    paper: { label: "Papers", icon: FileText },
    library_doc: { label: "Libraries", icon: Package },
    unknown: { label: "Other", icon: Folder },
  };

  // Order for displaying groups
  const typeOrder = ["book", "paper", "library_doc", "unknown"];

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          disabled={disabled}
          className={cn(
            "h-8 w-8",
            selectedModules.length > 0
              ? "text-primary"
              : "text-muted-foreground hover:text-foreground"
          )}
          title={
            selectedModules.length > 0
              ? `${selectedModules.length} module${selectedModules.length !== 1 ? "s" : ""} selected`
              : "Select knowledge modules"
          }
        >
          <Database className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        side="top"
        className="flex max-h-[400px] w-80 flex-col p-0"
        onInteractOutside={(e) => {
          // Prevent closing if there are unsaved changes
          if (hasChanges) {
            e.preventDefault();
          }
        }}
      >
        {/* Header */}
        <div className="flex flex-shrink-0 items-center justify-between border-b px-3 py-2">
          <div className="flex items-center gap-2">
            <Database className="text-muted-foreground h-4 w-4" />
            <span className="text-sm font-medium">Knowledge Modules</span>
          </div>
          {selectedCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="text-muted-foreground hover:text-foreground h-auto px-2 py-1 text-xs"
            >
              Clear all
            </Button>
          )}
        </div>

        {/* Module list - scrollable */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="p-2">
            {isLoading ? (
              <div className="text-muted-foreground py-4 text-center text-sm">
                Loading modules...
              </div>
            ) : modules.length === 0 ? (
              <div className="text-muted-foreground py-4 text-center text-sm">
                No modules available.
                <br />
                <span className="text-xs">
                  Index documents using the CLI to create modules.
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                {typeOrder.map((type) => {
                  const group = groupedModules[type];
                  if (!group || group.length === 0) return null;

                  const config = typeConfig[type] || typeConfig.unknown;
                  const Icon = config.icon;

                  return (
                    <div key={type}>
                      {/* Group header */}
                      <div className="text-muted-foreground mb-1 flex items-center gap-1.5 px-2 text-xs font-medium">
                        <Icon className="h-3 w-3" />
                        <span>{config.label}</span>
                      </div>
                      {/* Group items */}
                      <div className="space-y-0.5">
                        {group.map((module) => {
                          const isSelected = localSelection.includes(module.name);
                          return (
                            <button
                              key={module.name}
                              onClick={() => handleToggle(module.name)}
                              className={cn(
                                "flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left transition-colors",
                                "hover:bg-muted/50",
                                isSelected && "bg-primary/10"
                              )}
                            >
                              <Checkbox
                                checked={isSelected}
                                className="pointer-events-none"
                              />
                              <span className="min-w-0 flex-1 truncate text-sm">
                                {module.display_name || module.name}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex flex-shrink-0 items-center justify-between border-t px-3 py-2">
          <span className="text-muted-foreground text-xs">
            {selectedCount === 0
              ? "No modules selected (LLM only)"
              : `${selectedCount} module${selectedCount !== 1 ? "s" : ""} selected`}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCancel}
              className="h-7 px-2 text-xs"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleApply}
              disabled={!hasChanges}
              className="h-7 gap-1 px-2 text-xs"
            >
              <Check className="h-3 w-3" />
              Apply
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
