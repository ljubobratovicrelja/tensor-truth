import { FileUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { ScopeType } from "@/api/types";
import { DocumentPanel } from "./DocumentPanel";

interface DocumentDialogProps {
  scopeId: string;
  scopeType: ScopeType;
}

export function DocumentDialog({ scopeId, scopeType }: DocumentDialogProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <FileUp className="mr-2 h-4 w-4" />
          Documents
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Manage Documents</DialogTitle>
        </DialogHeader>
        <DocumentPanel scopeId={scopeId} scopeType={scopeType} />
      </DialogContent>
    </Dialog>
  );
}
