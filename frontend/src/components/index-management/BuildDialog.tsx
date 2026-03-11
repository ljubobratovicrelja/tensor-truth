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
import { Loader2 } from "lucide-react";
import { useBuildModuleIndex, useEmbeddingModels } from "@/hooks";
import { toast } from "sonner";
import { TaskProgressCard } from "./TaskProgressCard";
import { useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";

interface BuildDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  moduleName: string;
}

export function BuildDialog({ open, onOpenChange, moduleName }: BuildDialogProps) {
  const [strategy, setStrategy] = useState("hierarchical");
  const [chunkSizes, setChunkSizes] = useState("2048, 512, 256");
  const [chunkOverlap, setChunkOverlap] = useState(64);
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [semanticBufferSize, setSemanticBufferSize] = useState(1);
  const [semanticThreshold, setSemanticThreshold] = useState(95);
  const [taskId, setTaskId] = useState<string | null>(null);
  const buildIndex = useBuildModuleIndex();
  const { data: embeddingModels } = useEmbeddingModels();
  const qc = useQueryClient();

  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset on open
      setTaskId(null);
    }
  }, [open]);

  const handleBuild = () => {
    const sizes = chunkSizes
      .split(",")
      .map((s) => parseInt(s.trim()))
      .filter((n) => !isNaN(n));

    if (sizes.length === 0) {
      toast.error("At least one chunk size is required");
      return;
    }

    buildIndex.mutate(
      {
        module_name: moduleName,
        chunking_strategy: strategy,
        chunk_sizes: sizes,
        chunk_overlap: chunkOverlap,
        embedding_model: embeddingModel || undefined,
        semantic_buffer_size: semanticBufferSize,
        semantic_breakpoint_threshold: semanticThreshold,
      },
      {
        onSuccess: (data) => {
          setTaskId(data.task_id);
          toast.success("Build started");
        },
        onError: (err) => toast.error(err.message),
      }
    );
  };

  const showSemantic = strategy === "semantic" || strategy === "semantic_hierarchical";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Build Index: {moduleName}</DialogTitle>
        </DialogHeader>

        {taskId ? (
          <TaskProgressCard
            taskId={taskId}
            label={`Building ${moduleName}`}
            onComplete={() => {
              toast.success(`Build complete: ${moduleName}`);
              qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
              qc.invalidateQueries({ queryKey: QUERY_KEYS.modules });
            }}
            onError={(task) => toast.error(`Build failed: ${task.error}`)}
          />
        ) : (
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label>Chunking Strategy</Label>
              <Select value={strategy} onValueChange={setStrategy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hierarchical">Hierarchical</SelectItem>
                  <SelectItem value="semantic">Semantic</SelectItem>
                  <SelectItem value="semantic_hierarchical">
                    Semantic + Hierarchical
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Chunk Sizes (comma-separated)</Label>
              <Input value={chunkSizes} onChange={(e) => setChunkSizes(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                min={0}
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Embedding Model</Label>
              <Select value={embeddingModel} onValueChange={setEmbeddingModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Default (configured)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Default</SelectItem>
                  {embeddingModels?.models.map((m) => (
                    <SelectItem key={m.model_id} value={m.model_name || m.model_id}>
                      {m.model_name || m.model_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {showSemantic && (
              <>
                <div className="space-y-1.5">
                  <Label>Semantic Buffer Size</Label>
                  <Input
                    type="number"
                    min={1}
                    value={semanticBufferSize}
                    onChange={(e) => setSemanticBufferSize(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Semantic Breakpoint Threshold (%)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={semanticThreshold}
                    onChange={(e) => setSemanticThreshold(Number(e.target.value))}
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
              <Button onClick={handleBuild} disabled={buildIndex.isPending}>
                {buildIndex.isPending && (
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                )}
                Build
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
