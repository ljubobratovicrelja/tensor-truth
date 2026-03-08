import { useState, useCallback } from "react";
import { Plus, Pencil, Trash2, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useMcpServers,
  useMcpServerPresets,
  useAddMcpServer,
  useUpdateMcpServer,
  useDeleteMcpServer,
  useToggleMcpServer,
} from "@/hooks";
import type { MCPServerResponse, MCPServerCreateRequest } from "@/api/types";

interface ServerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editServer?: MCPServerResponse;
}

function ServerDialog({ open, onOpenChange, editServer }: ServerDialogProps) {
  const { data: presetsData } = useMcpServerPresets();
  const addServer = useAddMcpServer();
  const updateServer = useUpdateMcpServer();

  const [preset, setPreset] = useState("custom");
  const [name, setName] = useState(editServer?.name ?? "");
  const [type, setType] = useState(editServer?.type ?? "stdio");
  const [command, setCommand] = useState(editServer?.command ?? "");
  const [args, setArgs] = useState(editServer?.args?.join("\n") ?? "");
  const [url, setUrl] = useState(editServer?.url ?? "");
  const [description, setDescription] = useState(editServer?.description ?? "");
  const [envText, setEnvText] = useState(
    editServer?.env
      ? Object.entries(editServer.env)
          .map(([k, v]) => `${k}=${v}`)
          .join("\n")
      : ""
  );

  const handlePresetChange = (presetKey: string) => {
    setPreset(presetKey);
    if (presetKey === "custom") return;
    const p = presetsData?.presets[presetKey];
    if (!p) return;
    setName(p.name ?? "");
    setType(p.type ?? "stdio");
    setCommand(p.command ?? "");
    setArgs(p.args?.join("\n") ?? "");
    setUrl(p.url ?? "");
    setDescription(p.description ?? "");
    if (p.env) {
      setEnvText(
        Object.entries(p.env)
          .map(([k, v]) => `${k}=${v}`)
          .join("\n")
      );
    } else {
      setEnvText("");
    }
  };

  const parseEnv = (): Record<string, string> | undefined => {
    if (!envText.trim()) return undefined;
    const env: Record<string, string> = {};
    for (const line of envText.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const idx = trimmed.indexOf("=");
      if (idx > 0) {
        env[trimmed.slice(0, idx)] = trimmed.slice(idx + 1);
      }
    }
    return Object.keys(env).length > 0 ? env : undefined;
  };

  const handleSubmit = () => {
    const parsedArgs = args
      .split("\n")
      .map((a) => a.trim())
      .filter(Boolean);

    if (editServer) {
      updateServer.mutate(
        {
          name: editServer.name,
          request: {
            type: type || undefined,
            command: command || undefined,
            args: parsedArgs.length ? parsedArgs : undefined,
            url: url || undefined,
            description: description || undefined,
            env: parseEnv(),
          },
        },
        {
          onSuccess: () => {
            toast.success(`Server "${editServer.name}" updated`);
            onOpenChange(false);
          },
          onError: (e) => toast.error(e.message),
        }
      );
    } else {
      const request: MCPServerCreateRequest = {
        name,
        type,
        command: command || undefined,
        args: parsedArgs.length ? parsedArgs : undefined,
        url: url || undefined,
        description: description || undefined,
        env: parseEnv(),
      };
      addServer.mutate(request, {
        onSuccess: () => {
          toast.success(`Server "${name}" added`);
          onOpenChange(false);
        },
        onError: (e) => toast.error(e.message),
      });
    }
  };

  const isPending = addServer.isPending || updateServer.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{editServer ? "Edit MCP Server" : "Add MCP Server"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {!editServer && (
            <div className="space-y-2">
              <Label>Preset</Label>
              <Select value={preset} onValueChange={handlePresetChange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="custom">Custom</SelectItem>
                  {presetsData &&
                    Object.entries(presetsData.presets).map(([key, p]) => (
                      <SelectItem key={key} value={key}>
                        {p.name}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="space-y-2">
            <Label>Name</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-server"
              disabled={!!editServer}
            />
          </div>

          <div className="space-y-2">
            <Label>Type</Label>
            <Select value={type} onValueChange={setType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="stdio">stdio</SelectItem>
                <SelectItem value="sse">sse</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {type === "stdio" ? (
            <>
              <div className="space-y-2">
                <Label>Command</Label>
                <Input
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  placeholder="npx"
                />
              </div>
              <div className="space-y-2">
                <Label>Arguments (one per line)</Label>
                <Textarea
                  value={args}
                  onChange={(e) => setArgs(e.target.value)}
                  placeholder={"-y\n@upstash/context7-mcp@latest"}
                  rows={3}
                  className="font-mono text-xs"
                />
              </div>
            </>
          ) : (
            <div className="space-y-2">
              <Label>URL</Label>
              <Input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="http://localhost:3000/sse"
              />
            </div>
          )}

          <div className="space-y-2">
            <Label>Description</Label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What this server does"
            />
          </div>

          <div className="space-y-2">
            <Label>Environment Variables (KEY=$VAR, one per line)</Label>
            <Textarea
              value={envText}
              onChange={(e) => setEnvText(e.target.value)}
              placeholder={"GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_PERSONAL_ACCESS_TOKEN"}
              rows={2}
              className="font-mono text-xs"
            />
            <p className="text-muted-foreground text-xs">
              Values starting with $ are resolved from your shell environment.
            </p>
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleSubmit} disabled={isPending || !name.trim()}>
              {isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : editServer ? (
                "Update"
              ) : (
                "Add"
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function McpServersSection() {
  const { data, isLoading } = useMcpServers();
  const toggleServer = useToggleMcpServer();
  const deleteServer = useDeleteMcpServer();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editServer, setEditServer] = useState<MCPServerResponse | undefined>();

  const handleEdit = useCallback((server: MCPServerResponse) => {
    setEditServer(server);
    setDialogOpen(true);
  }, []);

  const handleAdd = useCallback(() => {
    setEditServer(undefined);
    setDialogOpen(true);
  }, []);

  const handleDelete = useCallback(
    (name: string) => {
      if (!confirm(`Remove server "${name}"?`)) return;
      deleteServer.mutate(name, {
        onSuccess: () => toast.success(`Server "${name}" removed`),
        onError: (e) => toast.error(e.message),
      });
    },
    [deleteServer]
  );

  const handleToggle = useCallback(
    (name: string, enabled: boolean) => {
      toggleServer.mutate(
        { name, enabled },
        {
          onError: (e) => toast.error(e.message),
        }
      );
    },
    [toggleServer]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  const servers = data?.servers ?? [];

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">MCP Servers</h3>
      <p className="text-muted-foreground text-xs">
        Model Context Protocol servers provide tools for agents and slash commands.
      </p>

      <div className="space-y-2">
        {servers.map((server) => (
          <div key={server.name} className="bg-muted/30 space-y-2 rounded-lg border p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">{server.name}</span>
                <Badge variant="outline" className="text-xs">
                  {server.type}
                </Badge>
                {server.builtin && (
                  <Badge variant="secondary" className="text-xs">
                    built-in
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={server.enabled}
                  onCheckedChange={(checked) => handleToggle(server.name, checked)}
                />
              </div>
            </div>

            {server.description && (
              <p className="text-muted-foreground text-xs">{server.description}</p>
            )}

            {server.command && (
              <p className="text-muted-foreground truncate font-mono text-xs">
                {server.command} {server.args?.join(" ")}
              </p>
            )}
            {server.url && (
              <p className="text-muted-foreground truncate font-mono text-xs">
                {server.url}
              </p>
            )}

            {/* Env var status */}
            {server.env_status && Object.keys(server.env_status).length > 0 && (
              <div className="flex flex-wrap gap-2">
                {Object.entries(server.env_status).map(([varName, resolved]) => (
                  <span key={varName} className="inline-flex items-center gap-1 text-xs">
                    {resolved ? (
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-amber-500" />
                    )}
                    <span
                      className={
                        resolved
                          ? "text-muted-foreground"
                          : "text-amber-600 dark:text-amber-400"
                      }
                    >
                      ${varName}
                    </span>
                  </span>
                ))}
              </div>
            )}

            {/* Actions for non-builtin */}
            {!server.builtin && (
              <div className="flex gap-1 pt-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => handleEdit(server)}
                >
                  <Pencil className="mr-1 h-3 w-3" />
                  Edit
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive h-7 px-2 text-xs"
                  onClick={() => handleDelete(server.name)}
                >
                  <Trash2 className="mr-1 h-3 w-3" />
                  Remove
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>

      <Button variant="outline" size="sm" className="w-full" onClick={handleAdd}>
        <Plus className="mr-2 h-3.5 w-3.5" />
        Add MCP Server
      </Button>

      {dialogOpen && (
        <ServerDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          editServer={editServer}
        />
      )}
    </div>
  );
}
