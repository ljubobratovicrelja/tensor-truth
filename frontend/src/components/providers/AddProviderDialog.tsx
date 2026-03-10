import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useTestProviderUrl } from "@/hooks/useProviders";
import { PROVIDER_TYPE_LABELS } from "@/lib/constants";
import type { ProviderResponse } from "@/api/types";

const TYPE_OPTIONS = Object.entries(PROVIDER_TYPE_LABELS).map(([value, label]) => ({
  value,
  label,
}));

const URL_PLACEHOLDERS: Record<string, string> = {
  ollama: "http://localhost:11434",
  llama_cpp: "http://localhost:8080",
  openai_compatible: "https://api.example.com/v1",
};

const ID_SUGGESTIONS: Record<string, string> = {
  ollama: "ollama",
  llama_cpp: "llama-cpp",
  openai_compatible: "openai",
};

interface AddProviderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (data: {
    id: string;
    type: string;
    base_url: string;
    api_key?: string;
    timeout?: number;
    default_capabilities?: string[];
  }) => void;
  isSaving: boolean;
  prefill?: { type: string; base_url: string; suggested_id: string };
  editProvider?: ProviderResponse;
}

export function AddProviderDialog({
  open,
  onOpenChange,
  onSave,
  isSaving,
  prefill,
  editProvider,
}: AddProviderDialogProps) {
  const [providerType, setProviderType] = useState("ollama");
  const [id, setId] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [providerTimeout, setProviderTimeout] = useState(300);
  const [assumeTools, setAssumeTools] = useState(true);
  const [assumeThinking, setAssumeThinking] = useState(true);

  const testUrl = useTestProviderUrl();

  const isEdit = !!editProvider;

  // Reset form on open
  useEffect(() => {
    if (!open) return;

    if (editProvider) {
      setProviderType(editProvider.type);
      setId(editProvider.id);
      setBaseUrl(editProvider.base_url);
      setApiKey(editProvider.api_key === "***" ? "***" : editProvider.api_key);
      setProviderTimeout(editProvider.timeout);
      setAssumeTools((editProvider.default_capabilities ?? []).includes("tools"));
      setAssumeThinking((editProvider.default_capabilities ?? []).includes("thinking"));
    } else if (prefill) {
      setProviderType(prefill.type);
      setId(prefill.suggested_id);
      setBaseUrl(prefill.base_url);
      setApiKey("");
      setProviderTimeout(300);
      setAssumeTools(true);
      setAssumeThinking(true);
    } else {
      setProviderType("ollama");
      setId("");
      setBaseUrl("");
      setApiKey("");
      setProviderTimeout(300);
      setAssumeTools(true);
      setAssumeThinking(true);
    }
    testUrl.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, editProvider, prefill]);

  // Auto-suggest ID from type (only in add mode, and only if user hasn't typed)
  useEffect(() => {
    if (!isEdit && !prefill) {
      setId(ID_SUGGESTIONS[providerType] || providerType);
    }
  }, [providerType, isEdit, prefill]);

  const handleTest = () => {
    testUrl.mutate({
      type: providerType,
      base_url: baseUrl,
      api_key: apiKey && apiKey !== "***" ? apiKey : undefined,
    });
  };

  const handleSave = () => {
    const caps: string[] = [];
    if (assumeTools) caps.push("tools");
    if (assumeThinking) caps.push("thinking");
    onSave({
      id,
      type: providerType,
      base_url: baseUrl,
      api_key: apiKey && apiKey !== "***" ? apiKey : undefined,
      timeout: providerTimeout,
      default_capabilities: caps,
    });
  };

  const canSave = id.trim() && baseUrl.trim();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Provider" : "Add Provider"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>ID</Label>
            <Input
              value={id}
              onChange={(e) =>
                setId(
                  e.target.value
                    .toLowerCase()
                    .replace(/[^a-z0-9_-]/g, "-")
                    .replace(/^-+/, "")
                )
              }
              placeholder="my-provider"
              disabled={isEdit}
            />
            {!isEdit && (
              <p className="text-muted-foreground text-xs">
                Lowercase letters, numbers, hyphens, and underscores only.
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label>Type</Label>
            <Select
              value={providerType}
              onValueChange={setProviderType}
              disabled={isEdit}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TYPE_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={URL_PLACEHOLDERS[providerType] || "http://localhost:8080"}
            />
          </div>

          {(providerType === "openai_compatible" || apiKey) && (
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                type={/^\$\{.+\}$/.test(apiKey) ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-... or ${ENV_VAR_NAME}"
              />
              <p className="text-muted-foreground text-xs">
                Use <code className="text-xs">{"${ENV_VAR_NAME}"}</code> to reference an
                environment variable.
              </p>
            </div>
          )}

          {providerType === "openai_compatible" && (
            <div className="space-y-3">
              <Label>Assume model capabilities</Label>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="assume-tools"
                  checked={assumeTools}
                  onCheckedChange={(v) => setAssumeTools(v === true)}
                />
                <label htmlFor="assume-tools" className="text-sm">
                  Tool calling
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="assume-thinking"
                  checked={assumeThinking}
                  onCheckedChange={(v) => setAssumeThinking(v === true)}
                />
                <label htmlFor="assume-thinking" className="text-sm">
                  Thinking / reasoning
                </label>
              </div>
              <p className="text-muted-foreground text-xs">
                Applied to all models that don&apos;t have explicit capabilities set.
              </p>
            </div>
          )}

          <div className="space-y-2">
            <Label>Timeout (seconds)</Label>
            <Input
              type="number"
              value={providerTimeout}
              onChange={(e) => setProviderTimeout(Number(e.target.value))}
              min={1}
            />
          </div>

          {/* Test result */}
          {testUrl.data && (
            <div
              className={`flex items-center gap-2 rounded-md p-2 text-sm ${
                testUrl.data.success
                  ? "bg-green-50 text-green-700 dark:bg-green-950/20 dark:text-green-400"
                  : "bg-red-50 text-red-700 dark:bg-red-950/20 dark:text-red-400"
              }`}
            >
              {testUrl.data.success ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              <span>{testUrl.data.message}</span>
            </div>
          )}

          <div className="flex justify-between gap-2">
            <Button
              variant="outline"
              onClick={handleTest}
              disabled={!baseUrl.trim() || testUrl.isPending}
            >
              {testUrl.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Testing...
                </>
              ) : (
                "Test Connection"
              )}
            </Button>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={!canSave || isSaving}>
                {isSaving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Save"
                )}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
