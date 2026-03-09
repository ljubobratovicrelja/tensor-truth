import { useCallback } from "react";
import {
  Download,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Package,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useExtensions,
  useExtensionLibrary,
  useInstallExtension,
  useUninstallExtension,
} from "@/hooks";
import type { LibraryExtensionResponse } from "@/api/types";

function McpStatusDot({
  requires_mcp,
  mcp_available,
}: {
  requires_mcp: string | null;
  mcp_available: boolean;
}) {
  if (!requires_mcp) return null;
  return (
    <span className="inline-flex items-center gap-1 text-xs">
      {mcp_available ? (
        <CheckCircle2 className="h-3 w-3 text-green-500" />
      ) : (
        <AlertCircle className="h-3 w-3 text-amber-500" />
      )}
      <span
        className={
          mcp_available ? "text-muted-foreground" : "text-amber-600 dark:text-amber-400"
        }
      >
        {requires_mcp}
      </span>
    </span>
  );
}

export function ExtensionsSection() {
  const { data: installedData, isLoading: installedLoading } = useExtensions();
  const { data: libraryData, isLoading: libraryLoading } = useExtensionLibrary();
  const installExtension = useInstallExtension();
  const uninstallExtension = useUninstallExtension();

  const handleInstall = useCallback(
    (ext: LibraryExtensionResponse) => {
      installExtension.mutate(
        { type: ext.type, filename: ext.filename },
        {
          onSuccess: () => toast.success(`Installed "${ext.name}"`),
          onError: (e) => toast.error(e.message),
        }
      );
    },
    [installExtension]
  );

  const handleUninstall = useCallback(
    (type: string, filename: string, name: string) => {
      if (!confirm(`Uninstall "${name}"?`)) return;
      uninstallExtension.mutate(
        { type, filename },
        {
          onSuccess: () => toast.success(`Uninstalled "${name}"`),
          onError: (e) => toast.error(e.message),
        }
      );
    },
    [uninstallExtension]
  );

  const handleInstallGroup = useCallback(
    (extensions: LibraryExtensionResponse[]) => {
      const toInstall = extensions.filter((e) => !e.installed);
      for (const ext of toInstall) {
        installExtension.mutate(
          { type: ext.type, filename: ext.filename },
          {
            onError: (e) => toast.error(`Failed to install "${ext.name}": ${e.message}`),
          }
        );
      }
      if (toInstall.length > 0) {
        toast.success(`Installing ${toInstall.length} extensions...`);
      }
    },
    [installExtension]
  );

  const isLoading = installedLoading || libraryLoading;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  const installed = installedData?.extensions ?? [];
  const library = libraryData?.extensions ?? [];

  // Group library extensions by MCP dependency
  const groups = new Map<string, LibraryExtensionResponse[]>();
  for (const ext of library) {
    const key = ext.requires_mcp ?? "_builtin";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(ext);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">Extensions</h3>
      <p className="text-muted-foreground text-xs">
        Slash commands and agents that extend TensorTruth with new capabilities.
      </p>

      {/* Installed Extensions */}
      {installed.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Installed
          </h4>
          {installed.map((ext) => (
            <div
              key={`${ext.type}:${ext.filename}`}
              className="bg-muted/30 flex items-center justify-between rounded-lg border px-3 py-2"
            >
              <div className="flex min-w-0 items-center gap-2">
                <Package className="text-muted-foreground h-3.5 w-3.5 shrink-0" />
                <span className="truncate text-sm font-medium">{ext.name}</span>
                <Badge variant="outline" className="shrink-0 text-xs">
                  {ext.type}
                </Badge>
                <McpStatusDot
                  requires_mcp={ext.requires_mcp}
                  mcp_available={ext.mcp_available}
                />
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="text-destructive hover:text-destructive h-7 shrink-0 px-2 text-xs"
                onClick={() => handleUninstall(ext.type, ext.filename, ext.name)}
                disabled={uninstallExtension.isPending}
              >
                <Trash2 className="mr-1 h-3 w-3" />
                Remove
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Extension Library */}
      {library.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
              Extension Library
            </h4>
            <a
              href="https://github.com/ljubobratovicrelja/tensor-truth/tree/main/extension_library"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-0.5 text-xs transition-colors"
              title="Browse extensions on GitHub"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          {Array.from(groups.entries()).map(([groupKey, extensions]) => (
            <div key={groupKey} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {groupKey !== "_builtin" && (
                    <McpStatusDot
                      requires_mcp={groupKey}
                      mcp_available={extensions[0]?.mcp_available ?? false}
                    />
                  )}
                  <span className="text-muted-foreground text-xs">
                    {groupKey === "_builtin"
                      ? "Built-in tools"
                      : `Requires ${groupKey} MCP`}
                  </span>
                </div>
                {extensions.some((e) => !e.installed) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs"
                    onClick={() => handleInstallGroup(extensions)}
                  >
                    Install All
                  </Button>
                )}
              </div>
              {extensions.map((ext) => (
                <div
                  key={`${ext.type}:${ext.filename}`}
                  className="flex items-center justify-between rounded px-3 py-1.5"
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="truncate text-sm">{ext.name}</span>
                    <Badge variant="outline" className="shrink-0 text-xs">
                      {ext.type}
                    </Badge>
                  </div>
                  {ext.installed ? (
                    <Badge variant="secondary" className="shrink-0 text-xs">
                      Installed
                    </Badge>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 shrink-0 px-2 text-xs"
                      onClick={() => handleInstall(ext)}
                      disabled={installExtension.isPending}
                    >
                      <Download className="mr-1 h-3 w-3" />
                      Install
                    </Button>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {installed.length === 0 && library.length === 0 && (
        <p className="text-muted-foreground py-4 text-center text-sm">
          No extensions available.
        </p>
      )}
    </div>
  );
}
