import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, Loader2, Link, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useUploadDocument, useUploadText, useUploadUrl } from "@/hooks";
import type { ScopeType } from "@/api/types";

interface DocumentUploaderProps {
  scopeId: string;
  scopeType: ScopeType;
}

const TEXT_EXTENSIONS = [".txt", ".md", ".markdown"];

function isTextFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return TEXT_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function DocumentUploader({ scopeId, scopeType }: DocumentUploaderProps) {
  const uploadDoc = useUploadDocument();
  const uploadText = useUploadText();
  const uploadUrlMutation = useUploadUrl();

  const [urlExpanded, setUrlExpanded] = useState(false);
  const [urlValue, setUrlValue] = useState("");
  const [urlContext, setUrlContext] = useState("");

  const isUploading = uploadDoc.isPending || uploadText.isPending;

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      for (const file of acceptedFiles) {
        try {
          if (isTextFile(file.name)) {
            // Read text file client-side and upload as text
            const text = await file.text();
            await uploadText.mutateAsync({
              scopeId,
              scopeType,
              data: { content: text, filename: file.name },
            });
          } else {
            // PDF or other binary files - upload as FormData
            await uploadDoc.mutateAsync({ scopeId, scopeType, file });
          }
        } catch (error) {
          const msg = error instanceof Error ? error.message : "Upload failed";
          toast.error(`Failed to upload ${file.name}: ${msg}`);
        }
      }
    },
    [scopeId, scopeType, uploadDoc, uploadText]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
      "text/markdown": [".md", ".markdown"],
    },
    multiple: true,
  });

  const handleUrlSubmit = async () => {
    const trimmedUrl = urlValue.trim();
    if (!trimmedUrl) return;

    try {
      await uploadUrlMutation.mutateAsync({
        scopeId,
        scopeType,
        data: { url: trimmedUrl, context: urlContext.trim() || undefined },
      });
      setUrlValue("");
      setUrlContext("");
      toast.success("URL content added");
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Failed to fetch URL";
      toast.error(msg);
    }
  };

  return (
    <div className="space-y-3">
      {/* Drag-and-drop zone */}
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
        {isUploading ? (
          <>
            <Loader2 className="text-muted-foreground mb-2 h-8 w-8 animate-spin" />
            <p className="text-muted-foreground text-sm">Uploading...</p>
          </>
        ) : isDragActive ? (
          <>
            <File className="text-primary mb-2 h-8 w-8" />
            <p className="text-primary text-sm">Drop files here</p>
          </>
        ) : (
          <>
            <Upload className="text-muted-foreground mb-2 h-8 w-8" />
            <p className="text-muted-foreground text-center text-sm">
              Drag & drop files here, or click to select
            </p>
            <p className="text-muted-foreground/60 mt-1 text-center text-xs">
              PDF, TXT, MD files supported
            </p>
          </>
        )}
      </div>

      {/* URL input section */}
      <div>
        <button
          type="button"
          onClick={() => setUrlExpanded(!urlExpanded)}
          className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-xs transition-colors"
        >
          {urlExpanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          <Link className="h-3 w-3" />
          Add from URL
        </button>

        {urlExpanded && (
          <div className="mt-2 space-y-2">
            <Input
              placeholder="https://example.com/page"
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
              className="text-sm"
            />
            <Textarea
              placeholder="Optional: Describe what this URL is about..."
              value={urlContext}
              onChange={(e) => setUrlContext(e.target.value)}
              rows={2}
              className="text-sm"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleUrlSubmit}
              disabled={!urlValue.trim() || uploadUrlMutation.isPending}
            >
              {uploadUrlMutation.isPending ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <Link className="mr-1 h-4 w-4" />
              )}
              Add URL
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
