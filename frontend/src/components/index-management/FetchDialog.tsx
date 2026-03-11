import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Loader2 } from "lucide-react";
import { useFetchSource } from "@/hooks";
import { toast } from "sonner";
import { TaskProgressCard } from "./TaskProgressCard";
import { useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";

interface FetchDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceType: string;
  name: string;
}

export function FetchDialog({ open, onOpenChange, sourceType, name }: FetchDialogProps) {
  const [workers, setWorkers] = useState(20);
  const [converter, setConverter] = useState("marker");
  const [outputFormat, setOutputFormat] = useState("markdown");
  const [enableCleanup, setEnableCleanup] = useState(false);
  const [minSize, setMinSize] = useState(0);
  const [pagesPerChunk, setPagesPerChunk] = useState(15);
  const [taskId, setTaskId] = useState<string | null>(null);
  const fetchSource = useFetchSource();
  const qc = useQueryClient();

  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset on open
      setTaskId(null);
    }
  }, [open]);

  const handleFetch = () => {
    fetchSource.mutate(
      {
        source_type: sourceType,
        name,
        workers,
        converter,
        output_format: outputFormat,
        enable_cleanup: enableCleanup,
        min_size: minSize,
        pages_per_chunk: pagesPerChunk,
      },
      {
        onSuccess: (data) => {
          setTaskId(data.task_id);
          toast.success("Fetch started");
        },
        onError: (err) => toast.error(err.message),
      }
    );
  };

  const isLibrary = sourceType === "libraries";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Fetch Docs: {name}</DialogTitle>
        </DialogHeader>

        {taskId ? (
          <TaskProgressCard
            taskId={taskId}
            label={`Fetching ${name}`}
            onComplete={() => {
              toast.success(`Fetch complete: ${name}`);
              qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
            }}
            onError={(task) => toast.error(`Fetch failed: ${task.error}`)}
          />
        ) : (
          <div className="space-y-3">
            {isLibrary && (
              <>
                <div className="space-y-1.5">
                  <Label>Workers</Label>
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={workers}
                    onChange={(e) => setWorkers(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Output Format</Label>
                  <Select value={outputFormat} onValueChange={setOutputFormat}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="markdown">Markdown</SelectItem>
                      <SelectItem value="html">HTML</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <Label>Cleanup</Label>
                  <Switch checked={enableCleanup} onCheckedChange={setEnableCleanup} />
                </div>
                <div className="space-y-1.5">
                  <Label>Min Size (bytes)</Label>
                  <Input
                    type="number"
                    min={0}
                    value={minSize}
                    onChange={(e) => setMinSize(Number(e.target.value))}
                  />
                </div>
              </>
            )}
            {!isLibrary && (
              <>
                <div className="space-y-1.5">
                  <Label>Converter</Label>
                  <Select value={converter} onValueChange={setConverter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="marker">Marker</SelectItem>
                      <SelectItem value="pymupdf">PyMuPDF</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Pages per Chunk</Label>
                  <Input
                    type="number"
                    min={1}
                    value={pagesPerChunk}
                    onChange={(e) => setPagesPerChunk(Number(e.target.value))}
                  />
                </div>
              </>
            )}
          </div>
        )}

        <DialogFooter>
          {!taskId && (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={handleFetch} disabled={fetchSource.isPending}>
                {fetchSource.isPending && (
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                )}
                Fetch
              </Button>
            </>
          )}
          {taskId && (
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
