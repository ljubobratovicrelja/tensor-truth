import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface IndexMetadataViewProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  moduleName: string;
  metadata: Record<string, unknown> | null;
}

export function IndexMetadataView({
  open,
  onOpenChange,
  moduleName,
  metadata,
}: IndexMetadataViewProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Index Metadata: {moduleName}</DialogTitle>
        </DialogHeader>
        {metadata ? (
          <div className="space-y-2 text-sm">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key} className="flex gap-2">
                <span className="text-muted-foreground min-w-[140px] font-mono text-xs">
                  {key}:
                </span>
                <span className="font-mono text-xs break-all">
                  {typeof value === "object" ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">No metadata available.</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
