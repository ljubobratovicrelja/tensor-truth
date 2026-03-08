import { useState, useEffect, useCallback } from "react";
import { Plus, Search, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  useProviders,
  useAddProvider,
  useUpdateProvider,
  useRemoveProvider,
  useDiscoverServers,
} from "@/hooks/useProviders";
import { ProviderCard } from "./ProviderCard";
import { DiscoveryBanner } from "./DiscoveryBanner";
import { AddProviderDialog } from "./AddProviderDialog";
import type { ProviderResponse, DiscoveredServer } from "@/api/types";

interface ProviderSetupPanelProps {
  mode: "startup" | "settings";
  onProvidersReady?: () => void;
}

export function ProviderSetupPanel({ mode, onProvidersReady }: ProviderSetupPanelProps) {
  const { data: providersData, isLoading } = useProviders();
  const addProvider = useAddProvider();
  const updateProvider = useUpdateProvider();
  const removeProvider = useRemoveProvider();
  const discoverServers = useDiscoverServers();

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingProvider, setEditingProvider] = useState<ProviderResponse | undefined>();
  const [prefill, setPrefill] = useState<
    { type: string; base_url: string; suggested_id: string } | undefined
  >();

  // Auto-discover in startup mode
  useEffect(() => {
    if (mode === "startup") {
      discoverServers.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // Notify parent when providers have models
  useEffect(() => {
    if (
      onProvidersReady &&
      providersData?.providers.some((p) => p.status === "connected" && p.model_count > 0)
    ) {
      onProvidersReady();
    }
  }, [providersData, onProvidersReady]);

  const handleAddFromDiscovery = useCallback((server: DiscoveredServer) => {
    setPrefill({
      type: server.type,
      base_url: server.base_url,
      suggested_id: server.suggested_id,
    });
    setEditingProvider(undefined);
    setDialogOpen(true);
  }, []);

  const handleOpenAdd = useCallback(() => {
    setPrefill(undefined);
    setEditingProvider(undefined);
    setDialogOpen(true);
  }, []);

  const handleEdit = useCallback((provider: ProviderResponse) => {
    setEditingProvider(provider);
    setPrefill(undefined);
    setDialogOpen(true);
  }, []);

  const handleDelete = useCallback(
    (provider: ProviderResponse) => {
      if (!confirm(`Remove provider "${provider.id}"?`)) return;
      setDeletingId(provider.id);
      removeProvider.mutate(provider.id, {
        onSuccess: () => toast.success(`Provider "${provider.id}" removed`),
        onError: (error) => toast.error(error.message),
        onSettled: () => setDeletingId(null),
      });
    },
    [removeProvider]
  );

  const handleSave = useCallback(
    (data: {
      id: string;
      type: string;
      base_url: string;
      api_key?: string;
      timeout?: number;
    }) => {
      if (editingProvider) {
        updateProvider.mutate(
          {
            id: editingProvider.id,
            request: {
              base_url: data.base_url,
              api_key: data.api_key,
              timeout: data.timeout,
            },
          },
          {
            onSuccess: () => {
              toast.success(`Provider "${editingProvider.id}" updated`);
              setDialogOpen(false);
            },
            onError: (error) => toast.error(error.message),
          }
        );
      } else {
        addProvider.mutate(
          {
            id: data.id,
            type: data.type,
            base_url: data.base_url,
            api_key: data.api_key,
            timeout: data.timeout,
          },
          {
            onSuccess: () => {
              toast.success(`Provider "${data.id}" added`);
              setDialogOpen(false);
              // Re-run discovery to update the banner
              discoverServers.mutate();
            },
            onError: (error) => toast.error(error.message),
          }
        );
      }
    },
    [editingProvider, addProvider, updateProvider, discoverServers]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="text-muted-foreground h-5 w-5 animate-spin" />
      </div>
    );
  }

  const providers = providersData?.providers || [];
  const discovered = discoverServers.data?.servers || [];

  return (
    <div className="space-y-4">
      {/* Startup mode header */}
      {mode === "startup" && (
        <div className="space-y-1">
          <h3 className="text-lg font-semibold">LLM Providers</h3>
          <p className="text-muted-foreground text-sm">
            Configure at least one LLM provider with available models to get started.
          </p>
        </div>
      )}

      {/* Discovery banner */}
      {discovered.length > 0 && (
        <DiscoveryBanner servers={discovered} onAdd={handleAddFromDiscovery} />
      )}

      {/* Provider cards */}
      <div className="space-y-2">
        {providers.map((provider) => (
          <ProviderCard
            key={provider.id}
            provider={provider}
            onEdit={() => handleEdit(provider)}
            onDelete={() => handleDelete(provider)}
            isDeleting={deletingId === provider.id}
          />
        ))}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleOpenAdd}>
          <Plus className="mr-1 h-3.5 w-3.5" />
          Add Provider
        </Button>
        {mode === "settings" && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => discoverServers.mutate()}
            disabled={discoverServers.isPending}
          >
            {discoverServers.isPending ? (
              <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Search className="mr-1 h-3.5 w-3.5" />
            )}
            Scan for local servers
          </Button>
        )}
      </div>

      {/* Add/Edit dialog */}
      <AddProviderDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onSave={handleSave}
        isSaving={addProvider.isPending || updateProvider.isPending}
        prefill={prefill}
        editProvider={editingProvider}
      />
    </div>
  );
}
