import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { useUpdateSource } from "@/hooks";
import { toast } from "sonner";

interface EditSourceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceType: string;
  name: string;
  config: Record<string, unknown>;
}

export function EditSourceDialog({
  open,
  onOpenChange,
  sourceType,
  name,
  config,
}: EditSourceDialogProps) {
  const [jsonText, setJsonText] = useState("");
  const [parseError, setParseError] = useState("");
  const updateSource = useUpdateSource();

  /* eslint-disable react-hooks/set-state-in-effect -- reset on open */
  useEffect(() => {
    if (open) {
      setJsonText(JSON.stringify(config, null, 2));
      setParseError("");
    }
  }, [open, config]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const handleSave = () => {
    try {
      const parsed = JSON.parse(jsonText);
      setParseError("");
      updateSource.mutate(
        { sourceType, name, config: parsed },
        {
          onSuccess: () => {
            toast.success(`Source "${name}" updated`);
            onOpenChange(false);
          },
          onError: (err) => toast.error(err.message),
        }
      );
    } catch {
      setParseError("Invalid JSON");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            Edit Source: {sourceType}/{name}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-2">
          <Label>Configuration (JSON)</Label>
          <textarea
            className="border-input bg-background flex min-h-[200px] w-full rounded-md border px-3 py-2 font-mono text-xs"
            value={jsonText}
            onChange={(e) => {
              setJsonText(e.target.value);
              setParseError("");
            }}
          />
          {parseError && <p className="text-xs text-red-500">{parseError}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={updateSource.isPending}>
            {updateSource.isPending && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
