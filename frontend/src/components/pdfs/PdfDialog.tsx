import { FileUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { PdfList } from "./PdfList";

interface PdfDialogProps {
  sessionId: string;
}

export function PdfDialog({ sessionId }: PdfDialogProps) {
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
        <PdfList sessionId={sessionId} />
      </DialogContent>
    </Dialog>
  );
}
