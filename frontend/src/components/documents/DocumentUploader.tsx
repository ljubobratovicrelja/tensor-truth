import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, Loader2, Link, BookOpen } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useUploadDocument,
  useUploadText,
  useUploadUrl,
  useArxivLookup,
  useUploadArxiv,
} from "@/hooks";
import type { ScopeType } from "@/api/types";

interface DocumentUploaderProps {
  scopeId: string;
  scopeType: ScopeType;
}

const TEXT_EXTENSIONS = [".txt", ".md", ".markdown"];
const ARXIV_ID_RE = /^\d{4}\.\d{4,5}$/;

function isTextFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return TEXT_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function useDebounce(value: string, delay: number): string {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

export function DocumentUploader({ scopeId, scopeType }: DocumentUploaderProps) {
  const uploadDoc = useUploadDocument();
  const uploadText = useUploadText();
  const uploadUrlMutation = useUploadUrl();
  const uploadArxivMutation = useUploadArxiv();

  const [urlValue, setUrlValue] = useState("");
  const [urlContext, setUrlContext] = useState("");
  const [arxivInput, setArxivInput] = useState("");
  const debouncedArxivId = useDebounce(arxivInput.trim(), 500);
  const arxivLookup = useArxivLookup(debouncedArxivId);

  const isUploading = uploadDoc.isPending || uploadText.isPending;

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      for (const file of acceptedFiles) {
        try {
          if (isTextFile(file.name)) {
            const text = await file.text();
            await uploadText.mutateAsync({
              scopeId,
              scopeType,
              data: { content: text, filename: file.name },
            });
          } else {
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

  const handleArxivSubmit = async () => {
    const id = arxivInput.trim();
    if (!id) return;

    try {
      await uploadArxivMutation.mutateAsync({
        scopeId,
        scopeType,
        data: { arxiv_id: id },
      });
      setArxivInput("");
      toast.success("arXiv paper added");
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Failed to add paper";
      toast.error(msg);
    }
  };

  const showArxivError =
    arxivInput.trim().length > 0 &&
    !ARXIV_ID_RE.test(arxivInput.trim()) &&
    arxivInput.trim().length >= 4;

  return (
    <Tabs defaultValue="files" className="w-full">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="files" className="text-xs">
          <Upload className="mr-1 h-3 w-3" />
          Files
        </TabsTrigger>
        <TabsTrigger value="url" className="text-xs">
          <Link className="mr-1 h-3 w-3" />
          URL
        </TabsTrigger>
        <TabsTrigger value="arxiv" className="text-xs">
          <BookOpen className="mr-1 h-3 w-3" />
          arXiv
        </TabsTrigger>
      </TabsList>

      {/* Files tab */}
      <TabsContent value="files" className="mt-3">
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
      </TabsContent>

      {/* URL tab */}
      <TabsContent value="url" className="mt-3">
        <div className="space-y-2">
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
      </TabsContent>

      {/* arXiv tab */}
      <TabsContent value="arxiv" className="mt-3">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Input
              placeholder="2301.12345"
              value={arxivInput}
              onChange={(e) => setArxivInput(e.target.value)}
              className="text-sm"
            />
            {arxivLookup.isFetching && (
              <Loader2 className="text-muted-foreground h-4 w-4 shrink-0 animate-spin" />
            )}
          </div>

          {showArxivError && (
            <p className="text-destructive text-xs">
              Invalid arXiv ID (expected format: 2301.12345)
            </p>
          )}

          {arxivLookup.isError && ARXIV_ID_RE.test(debouncedArxivId) && (
            <p className="text-destructive text-xs">Paper not found</p>
          )}

          {arxivLookup.data && (
            <div className="bg-muted/50 rounded-md border p-3">
              <p className="text-sm leading-snug font-semibold">
                {arxivLookup.data.title}
              </p>
              <p className="text-muted-foreground mt-1 text-xs">
                {arxivLookup.data.authors.slice(0, 5).join(", ")}
                {arxivLookup.data.authors.length > 5 && " et al."}
              </p>
              <p className="text-muted-foreground mt-0.5 text-xs">
                {arxivLookup.data.published.slice(0, 4)}
                {" \u00b7 "}
                {arxivLookup.data.categories[0]}
              </p>
            </div>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={handleArxivSubmit}
            disabled={!arxivLookup.data || uploadArxivMutation.isPending}
          >
            {uploadArxivMutation.isPending ? (
              <Loader2 className="mr-1 h-4 w-4 animate-spin" />
            ) : (
              <BookOpen className="mr-1 h-4 w-4" />
            )}
            Add Preprint
          </Button>
        </div>
      </TabsContent>
    </Tabs>
  );
}
