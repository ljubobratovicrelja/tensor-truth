import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUploadPdf } from "@/hooks";

interface PdfUploaderProps {
  sessionId: string;
  onUploadComplete?: () => void;
}

export function PdfUploader({ sessionId, onUploadComplete }: PdfUploaderProps) {
  const uploadPdf = useUploadPdf();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      for (const file of acceptedFiles) {
        try {
          await uploadPdf.mutateAsync({ sessionId, file });
        } catch (error) {
          console.error("Failed to upload PDF:", error);
        }
      }
      onUploadComplete?.();
    },
    [sessionId, uploadPdf, onUploadComplete]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    multiple: true,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors",
        isDragActive
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/25 hover:border-primary/50"
      )}
    >
      <input {...getInputProps()} />
      {uploadPdf.isPending ? (
        <>
          <Loader2 className="text-muted-foreground mb-2 h-8 w-8 animate-spin" />
          <p className="text-muted-foreground text-sm">Uploading...</p>
        </>
      ) : isDragActive ? (
        <>
          <File className="text-primary mb-2 h-8 w-8" />
          <p className="text-primary text-sm">Drop the PDF here</p>
        </>
      ) : (
        <>
          <Upload className="text-muted-foreground mb-2 h-8 w-8" />
          <p className="text-muted-foreground text-center text-sm">
            Drag & drop PDFs here, or click to select
          </p>
        </>
      )}
    </div>
  );
}
