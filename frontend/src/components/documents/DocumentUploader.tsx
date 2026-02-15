import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File as FileIcon, Loader2, Link, BookOpen, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
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
  useFileUrlInfo,
  useUploadFileUrl,
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

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getTypeBadge(filename: string, contentType?: string): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf") || contentType === "application/pdf") return "PDF";
  if (
    lower.endsWith(".md") ||
    lower.endsWith(".markdown") ||
    contentType === "text/markdown"
  )
    return "MD";
  if (lower.endsWith(".txt") || contentType === "text/plain") return "TXT";
  return contentType?.split("/")[1]?.toUpperCase() || "FILE";
}

export function DocumentUploader({ scopeId, scopeType }: DocumentUploaderProps) {
  const uploadDoc = useUploadDocument();
  const uploadText = useUploadText();
  const uploadUrlMutation = useUploadUrl();
  const uploadArxivMutation = useUploadArxiv();
  const uploadFileUrlMutation = useUploadFileUrl();

  // Files tab state
  const [uploadingCount, setUploadingCount] = useState(0);
  const [fileUrlInput, setFileUrlInput] = useState("");
  const debouncedFileUrl = useDebounce(fileUrlInput.trim(), 500);
  const fileUrlInfo = useFileUrlInfo(debouncedFileUrl);

  // URL tab state
  const [urlValue, setUrlValue] = useState("");
  const [urlContext, setUrlContext] = useState("");

  // arXiv tab state
  const [arxivInput, setArxivInput] = useState("");
  const debouncedArxivId = useDebounce(arxivInput.trim(), 500);
  const arxivLookup = useArxivLookup(debouncedArxivId);

  const isFileUploading = uploadingCount > 0 || uploadFileUrlMutation.isPending;

  // Determine what's staged for preview (URL files only)
  const hasUrlStaged = fileUrlInfo.data?.supported === true;
  const hasStaged = hasUrlStaged;

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setFileUrlInput("");
      setUploadingCount((c) => c + acceptedFiles.length);

      const results = await Promise.allSettled(
        acceptedFiles.map(async (file) => {
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
        })
      );

      setUploadingCount((c) => c - acceptedFiles.length);

      const succeeded = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected");

      if (succeeded > 0) {
        toast.success(`${succeeded} file${succeeded === 1 ? "" : "s"} added`);
      }
      for (const f of failed) {
        const msg = f.reason instanceof Error ? f.reason.message : "Upload failed";
        toast.error(msg);
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
    disabled: uploadingCount > 0,
  });

  const handleFileUrlChange = (value: string) => {
    setFileUrlInput(value);
  };

  const clearStaged = () => {
    setFileUrlInput("");
  };

  const handleAddFile = async () => {
    if (hasUrlStaged && fileUrlInfo.data) {
      try {
        await uploadFileUrlMutation.mutateAsync({
          scopeId,
          scopeType,
          data: { url: fileUrlInfo.data.url },
        });
        clearStaged();
        toast.success("File added");
      } catch (error) {
        const msg = error instanceof Error ? error.message : "Download failed";
        toast.error(msg);
      }
    }
  };

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

  // File URL error states
  const showFileUrlError = fileUrlInfo.isError && /^https?:\/\/.+/.test(debouncedFileUrl);
  const showFileUrlUnsupported = fileUrlInfo.data?.supported === false;

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
        <div className="space-y-3">
          {/* Drop zone — compact single row */}
          <div
            {...getRootProps()}
            className={cn(
              "flex items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-3 transition-colors",
              uploadingCount > 0
                ? "border-muted-foreground/25 pointer-events-none opacity-60"
                : "cursor-pointer",
              isDragActive
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25 hover:border-primary/50"
            )}
          >
            <input {...getInputProps()} />
            {uploadingCount > 0 ? (
              <>
                <Loader2 className="text-muted-foreground h-4 w-4 shrink-0 animate-spin" />
                <p className="text-muted-foreground text-sm">
                  Uploading {uploadingCount} file{uploadingCount === 1 ? "" : "s"}...
                </p>
              </>
            ) : isDragActive ? (
              <>
                <FileIcon className="text-primary h-4 w-4 shrink-0" />
                <p className="text-primary text-sm">Drop files here</p>
              </>
            ) : (
              <>
                <Upload className="text-muted-foreground h-4 w-4 shrink-0" />
                <p className="text-muted-foreground text-sm">
                  Drag & drop or click to browse
                  <span className="text-muted-foreground/60 ml-1 text-xs">
                    PDF, TXT, MD
                  </span>
                </p>
              </>
            )}
          </div>

          {/* Divider */}
          <div className="flex items-center gap-3">
            <div className="bg-border h-px flex-1" />
            <span className="text-muted-foreground text-xs">or paste a file URL</span>
            <div className="bg-border h-px flex-1" />
          </div>

          {/* URL input */}
          <div className="flex items-center gap-2">
            <Input
              placeholder="https://example.com/paper.pdf"
              value={fileUrlInput}
              onChange={(e) => handleFileUrlChange(e.target.value)}
              className="text-sm"
            />
            {fileUrlInfo.isFetching && (
              <Loader2 className="text-muted-foreground h-4 w-4 shrink-0 animate-spin" />
            )}
          </div>

          {/* Error states */}
          {showFileUrlError && (
            <p className="text-destructive text-xs">Could not reach file</p>
          )}
          {showFileUrlUnsupported && fileUrlInfo.data && (
            <p className="text-destructive text-xs">
              Unsupported file type: {fileUrlInfo.data.content_type}
            </p>
          )}

          {/* Preview card (URL files only) */}
          {hasUrlStaged && fileUrlInfo.data && (
            <div className="bg-muted/50 flex items-start gap-3 rounded-md border p-3">
              <FileIcon className="text-muted-foreground mt-0.5 h-5 w-5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-medium">
                    {fileUrlInfo.data.filename}
                  </p>
                  <Badge variant="secondary" className="shrink-0 px-1.5 py-0 text-[10px]">
                    {getTypeBadge(
                      fileUrlInfo.data.filename,
                      fileUrlInfo.data.content_type
                    )}
                  </Badge>
                </div>
                <p className="text-muted-foreground mt-0.5 text-xs">
                  {fileUrlInfo.data.file_size != null
                    ? formatFileSize(fileUrlInfo.data.file_size)
                    : "Size unknown"}
                </p>
                <p className="text-muted-foreground mt-0.5 truncate text-xs">
                  {fileUrlInfo.data.url}
                </p>
              </div>
              <button
                onClick={clearStaged}
                className="text-muted-foreground hover:text-foreground shrink-0 p-0.5"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Add File button */}
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            onClick={handleAddFile}
            disabled={!hasStaged || isFileUploading}
          >
            {isFileUploading ? (
              <Loader2 className="mr-1 h-4 w-4 animate-spin" />
            ) : (
              <Upload className="mr-1 h-4 w-4" />
            )}
            Add File
          </Button>
        </div>
      </TabsContent>

      {/* URL tab */}
      <TabsContent value="url" className="mt-3">
        <div className="space-y-2">
          <p className="text-muted-foreground text-xs">
            Fetches a web page and converts its HTML content to text.
          </p>
          <Input
            placeholder="https://docs.example.com/guide"
            value={urlValue}
            onChange={(e) => setUrlValue(e.target.value)}
            className="text-sm"
          />
          <Textarea
            placeholder="Optional: Describe what this page is about..."
            value={urlContext}
            onChange={(e) => setUrlContext(e.target.value)}
            rows={2}
            className="text-sm"
          />
          <Button
            variant="outline"
            size="sm"
            className="w-full"
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
            className="w-full"
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
