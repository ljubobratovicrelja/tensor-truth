import { useState, useEffect, useMemo } from "react";
import {
  Database,
  Check,
  Book,
  FileText,
  Package,
  Folder,
  Link,
  Lock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { getShortModelId, inferDocType, generateDisplayName } from "@/lib/moduleUtils";
import { useModules, useConfig, useEmbeddingModels } from "@/hooks";
import type { ModuleInfo, DocumentInfo } from "@/api/types";

interface ModuleSelectorProps {
  selectedModules: string[];
  onModulesChange: (modules: string[]) => void;
  disabled?: boolean;
  /** Embedding model to filter modules by. If provided, only modules indexed with this model are shown. */
  embeddingModel?: string;
  /** Module names that are locked (from project catalog). Shown as checked + greyed out. */
  lockedModules?: string[];
  /** Documents attached to the project. Shown as a flat read-only list. */
  projectDocuments?: DocumentInfo[];
}

export function ModuleSelector({
  selectedModules,
  onModulesChange,
  disabled = false,
  embeddingModel,
  lockedModules,
  projectDocuments,
}: ModuleSelectorProps) {
  const [open, setOpen] = useState(false);
  const [localSelection, setLocalSelection] = useState<string[]>(selectedModules);
  const { data: modulesData, isLoading } = useModules();
  const { data: config } = useConfig();
  const { data: embeddingModelsData } = useEmbeddingModels();

  // Use prop if provided, otherwise fall back to config default
  const effectiveEmbeddingModel = embeddingModel ?? config?.rag?.default_embedding_model;
  const embeddingModelId = getShortModelId(effectiveEmbeddingModel);

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

  // Find the embedding model info matching the effective embedding model
  const selectedModelInfo = useMemo(() => {
    if (!embeddingModelsData || !effectiveEmbeddingModel) return null;

    const effectiveLower = effectiveEmbeddingModel.toLowerCase();
    const shortEffective = getShortModelId(effectiveEmbeddingModel);

    // Try exact matches first (case-insensitive), then short ID matches
    return (
      embeddingModelsData.models.find(
        (m) =>
          m.model_id.toLowerCase() === effectiveLower ||
          m.model_name?.toLowerCase() === effectiveLower
      ) ??
      embeddingModelsData.models.find(
        (m) =>
          m.model_id.toLowerCase() === shortEffective ||
          getShortModelId(m.model_name ?? "") === shortEffective
      ) ??
      null
    );
  }, [embeddingModelsData, effectiveEmbeddingModel]);

  // Get the set of available module names for the selected embedding model
  const availableModuleNames = useMemo(() => {
    return selectedModelInfo ? new Set(selectedModelInfo.modules) : null;
  }, [selectedModelInfo]);

  // Clear invalid module selections when embedding model changes
  useEffect(() => {
    if (!availableModuleNames || selectedModules.length === 0) return;
    const validSelection = selectedModules.filter((m) => availableModuleNames.has(m));
    if (validSelection.length !== selectedModules.length) {
      // Some selected modules are no longer valid - update parent
      onModulesChange(validSelection);
    }
  }, [availableModuleNames, selectedModules, onModulesChange]);

  // Build module list from embeddingModelsData (source of truth for available modules)
  // and enrich with display info from modulesData where available
  const filteredModules = useMemo(() => {
    // If no selected model info, we can't determine available modules - show nothing
    if (!selectedModelInfo) {
      return [];
    }

    // Create a lookup map from modulesData for display info
    const moduleInfoMap = new Map<string, ModuleInfo>();
    for (const m of modulesData?.modules ?? []) {
      moduleInfoMap.set(m.name, m);
    }

    // Build modules list from selectedModelInfo.modules (the actual available modules)
    const modules = selectedModelInfo.modules.map((moduleName): ModuleInfo => {
      const existingInfo = moduleInfoMap.get(moduleName);
      if (existingInfo) {
        return existingInfo;
      }
      // Create info for modules not in modulesData by inferring from name
      // This happens when session's embedding model differs from config's default
      const { doc_type, sort_order } = inferDocType(moduleName);
      return {
        name: moduleName,
        display_name: generateDisplayName(moduleName),
        doc_type,
        sort_order,
      };
    });

    // Sort by sort_order (type) then alphabetically by display_name
    return modules.sort((a, b) => {
      if (a.sort_order !== b.sort_order) {
        return a.sort_order - b.sort_order;
      }
      return a.display_name.toLowerCase().localeCompare(b.display_name.toLowerCase());
    });
  }, [selectedModelInfo, modulesData?.modules]);

  const modules = filteredModules;
  const selectedCount = localSelection.length;

  // Group modules by doc_type, sorted alphabetically within each group
  const groupedModules = useMemo(() => {
    const groups: Record<string, ModuleInfo[]> = {};
    for (const module of filteredModules) {
      const type = module.doc_type || "unknown";
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(module);
    }
    // Sort each group alphabetically by display_name
    for (const type of Object.keys(groups)) {
      groups[type].sort((a, b) =>
        a.display_name.toLowerCase().localeCompare(b.display_name.toLowerCase())
      );
    }
    return groups;
  }, [filteredModules]);

  // Type display config
  const typeConfig: Record<string, { label: string; icon: typeof Book }> = {
    book: { label: "Books", icon: Book },
    paper: { label: "Papers", icon: FileText },
    library_doc: { label: "Libraries", icon: Package },
    unknown: { label: "Other", icon: Folder },
  };

  // Order for displaying groups
  const typeOrder = ["book", "paper", "library_doc", "unknown"];

  // Project context: determine which modules are locked vs additional
  const isProjectContext = !!lockedModules;
  const lockedModuleSet = useMemo(() => new Set(lockedModules ?? []), [lockedModules]);
  const additionalModules = useMemo(() => {
    if (!isProjectContext) return filteredModules;
    return filteredModules.filter((m) => !lockedModuleSet.has(m.name));
  }, [isProjectContext, filteredModules, lockedModuleSet]);

  // Locked modules with display info
  const lockedModuleInfos = useMemo(() => {
    if (!isProjectContext || !lockedModules) return [];
    const moduleInfoMap = new Map<string, ModuleInfo>();
    for (const m of modulesData?.modules ?? []) {
      moduleInfoMap.set(m.name, m);
    }
    return lockedModules.map((name) => {
      const existing = moduleInfoMap.get(name);
      if (existing) return existing;
      const { doc_type, sort_order } = inferDocType(name);
      return {
        name,
        display_name: generateDisplayName(name),
        doc_type,
        sort_order,
      };
    });
  }, [isProjectContext, lockedModules, modulesData?.modules]);

  // Total effective count for display (locked + user-selected)
  const totalSelectedCount = isProjectContext
    ? (lockedModules?.length ?? 0) + localSelection.length
    : localSelection.length;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          disabled={disabled}
          className={cn(
            "h-8 w-8",
            totalSelectedCount > 0
              ? "text-primary"
              : "text-muted-foreground hover:text-foreground"
          )}
          title={
            totalSelectedCount > 0
              ? `${totalSelectedCount} module${totalSelectedCount !== 1 ? "s" : ""} selected`
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
            <span className="text-sm font-medium">
              Knowledge Modules
              {embeddingModelId && (
                <span className="text-muted-foreground font-normal">
                  {" "}
                  ({embeddingModelId})
                </span>
              )}
            </span>
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
            ) : isProjectContext ? (
              /* Project context: 3 sections */
              <div className="space-y-3">
                {/* 1. Project Knowledge (documents) */}
                {projectDocuments && projectDocuments.length > 0 && (
                  <div>
                    <div className="text-muted-foreground mb-1 flex items-center gap-1.5 px-2 text-xs font-medium">
                      <FileText className="h-3 w-3" />
                      <span>Project Knowledge</span>
                    </div>
                    <div className="space-y-0.5">
                      {projectDocuments.map((doc) => (
                        <div
                          key={doc.doc_id}
                          className="flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left"
                        >
                          {doc.type === "url" ? (
                            <Link className="text-muted-foreground h-4 w-4 shrink-0" />
                          ) : (
                            <FileText className="text-muted-foreground h-4 w-4 shrink-0" />
                          )}
                          <span className="text-muted-foreground min-w-0 flex-1 truncate text-sm">
                            {doc.filename || doc.url || doc.doc_id}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 2. Project Modules (locked) */}
                {lockedModuleInfos.length > 0 && (
                  <div>
                    <div className="text-muted-foreground mb-1 flex items-center gap-1.5 px-2 text-xs font-medium">
                      <Lock className="h-3 w-3" />
                      <span>Project Modules</span>
                    </div>
                    <div className="space-y-0.5">
                      {lockedModuleInfos.map((module) => (
                        <div
                          key={module.name}
                          className="bg-muted/30 flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left opacity-70"
                        >
                          <Checkbox
                            checked={true}
                            disabled
                            className="pointer-events-none"
                          />
                          <span className="min-w-0 flex-1 truncate text-sm">
                            {module.display_name || module.name}
                          </span>
                          <span className="text-muted-foreground shrink-0 text-xs">
                            (locked)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 3. Additional Modules (toggleable) */}
                {additionalModules.length > 0 && (
                  <div>
                    <div className="text-muted-foreground mb-1 flex items-center gap-1.5 px-2 text-xs font-medium">
                      <Database className="h-3 w-3" />
                      <span>Additional Modules</span>
                    </div>
                    <div className="space-y-0.5">
                      {additionalModules.map((module) => {
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
                )}

                {lockedModuleInfos.length === 0 &&
                  additionalModules.length === 0 &&
                  (!projectDocuments || projectDocuments.length === 0) && (
                    <div className="text-muted-foreground py-4 text-center text-sm">
                      No knowledge sources configured.
                    </div>
                  )}
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
              /* Default (non-project) view: grouped by type */
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
            {totalSelectedCount === 0
              ? "No modules selected (LLM only)"
              : isProjectContext
                ? `${lockedModules?.length ?? 0} locked + ${selectedCount} additional`
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
