import { FileText, Trash2, RefreshCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { usePdfs, useDeletePdf, useReindexPdfs } from "@/hooks";
import { PdfUploader } from "./PdfUploader";

interface PdfListProps {
  sessionId: string;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function PdfList({ sessionId }: PdfListProps) {
  const { data, isLoading } = usePdfs(sessionId);
  const deletePdf = useDeletePdf();
  const reindex = useReindexPdfs();

  const handleDelete = async (pdfId: string) => {
    try {
      await deletePdf.mutateAsync({ sessionId, pdfId });
    } catch (error) {
      console.error("Failed to delete PDF:", error);
    }
  };

  const handleReindex = async () => {
    try {
      await reindex.mutateAsync(sessionId);
    } catch (error) {
      console.error("Failed to reindex:", error);
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg">Documents</CardTitle>
        {data && data.pdfs.length > 0 && (
          <div className="flex items-center gap-2">
            {data.has_index ? (
              <Badge variant="secondary">Indexed</Badge>
            ) : (
              <Badge variant="outline">Not indexed</Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleReindex}
              disabled={reindex.isPending || data.pdfs.length === 0}
            >
              {reindex.isPending ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-1 h-4 w-4" />
              )}
              Reindex
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <PdfUploader sessionId={sessionId} />

        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
          </div>
        ) : data?.pdfs.length === 0 ? (
          <p className="text-muted-foreground py-4 text-center text-sm">
            No documents uploaded yet
          </p>
        ) : (
          <div className="space-y-2">
            {data?.pdfs.map((pdf) => (
              <div
                key={pdf.pdf_id}
                className="flex items-center justify-between rounded-lg border p-3"
              >
                <div className="flex items-center gap-3">
                  <FileText className="text-muted-foreground h-5 w-5" />
                  <div>
                    <p className="text-sm font-medium">{pdf.filename}</p>
                    <p className="text-muted-foreground text-xs">
                      {pdf.page_count} pages &middot; {formatFileSize(pdf.file_size)}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDelete(pdf.pdf_id)}
                  disabled={deletePdf.isPending}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
