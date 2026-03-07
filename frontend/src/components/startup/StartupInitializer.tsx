/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  Loader2,
  CheckCircle2,
  AlertCircle,
  Download,
  Server,
  AlertTriangle,
  WifiOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useStartupStatus, useDownloadIndexes } from "@/hooks/useStartup";

interface StartupInitializerProps {
  onComplete: () => void;
}

const DOWNLOAD_START_KEY = "tensortruth-download-start";
const INDEXES_SKIPPED_KEY = "tensortruth-indexes-skipped";

export function StartupInitializer({ onComplete }: StartupInitializerProps) {
  const [indexesSkipped, setIndexesSkipped] = useState(
    () => localStorage.getItem(INDEXES_SKIPPED_KEY) === "true"
  );
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadStartTime, setDownloadStartTime] = useState<number | null>(null);
  const [downloadElapsedSeconds, setDownloadElapsedSeconds] = useState(0);

  // Poll every 1s when downloading, otherwise every 5s
  const pollingInterval = isDownloading ? 1000 : 5000;
  const {
    data: status,
    isLoading,
    isError,
    error,
  } = useStartupStatus({ pollingInterval });
  const downloadIndexes = useDownloadIndexes();

  // Apply theme on mount from localStorage or system preference
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");

    try {
      // Try to read theme from localStorage (same key as Zustand store)
      const stored = localStorage.getItem("tensortruth-ui");
      const theme = stored ? JSON.parse(stored).state?.theme : "system";

      if (theme === "system" || !theme) {
        // Use system preference
        const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
        root.classList.add(systemTheme);
      } else {
        // Use stored preference
        root.classList.add(theme);
      }
    } catch {
      // Fallback to system preference if localStorage read fails
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
      root.classList.add(systemTheme);
    }
  }, []);

  // Restore download/pull state from localStorage on mount
  useEffect(() => {
    if (!status) return;

    // Check if download was in progress before reload
    const storedDownloadStart = localStorage.getItem(DOWNLOAD_START_KEY);
    if (storedDownloadStart && !status.indexes_ok) {
      const startTime = parseInt(storedDownloadStart, 10);
      setIsDownloading(true);
      setDownloadStartTime(startTime);
    } else if (status.indexes_ok) {
      // Clean up if indexes are now available
      localStorage.removeItem(DOWNLOAD_START_KEY);
    }
  }, [status]);

  // Update elapsed time every second while downloading
  useEffect(() => {
    if (!isDownloading || !downloadStartTime) {
      setDownloadElapsedSeconds(0);
      return;
    }

    const interval = setInterval(() => {
      setDownloadElapsedSeconds(Math.round((Date.now() - downloadStartTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isDownloading, downloadStartTime]);

  // Auto-complete when all resources are available
  useEffect(() => {
    if (status?.ready && (status.indexes_ok || indexesSkipped) && status.models_ok) {
      // All resources available (or indexes skipped), complete initialization
      setIsDownloading(false);
      onComplete();
    }
  }, [status, indexesSkipped, onComplete]);

  // Detect when indexes become available during download
  useEffect(() => {
    if (isDownloading && status?.indexes_ok) {
      setIsDownloading(false);
      localStorage.removeItem(DOWNLOAD_START_KEY);
      localStorage.removeItem(INDEXES_SKIPPED_KEY);
      setIndexesSkipped(false);
      const elapsed = downloadStartTime ? Date.now() - downloadStartTime : 0;
      const elapsedSeconds = Math.round(elapsed / 1000);
      toast.success(`Indexes downloaded successfully! (${elapsedSeconds}s)`);
    }
  }, [isDownloading, status?.indexes_ok, downloadStartTime]);

  const handleSkipIndexes = () => {
    localStorage.setItem(INDEXES_SKIPPED_KEY, "true");
    setIndexesSkipped(true);
  };

  const handleDownloadIndexes = () => {
    const startTime = Date.now();
    setIsDownloading(true);
    setDownloadStartTime(startTime);
    localStorage.setItem(DOWNLOAD_START_KEY, startTime.toString());

    downloadIndexes.mutate(undefined, {
      onSuccess: (response) => {
        toast.info(response.message);
      },
      onError: (error) => {
        setIsDownloading(false);
        localStorage.removeItem(DOWNLOAD_START_KEY);
        toast.error(`Failed to download indexes: ${error.message}`);
      },
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="bg-background flex h-screen items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="text-primary h-12 w-12 animate-spin" />
          <p className="text-muted-foreground text-lg">Initializing TensorTruth...</p>
        </div>
      </div>
    );
  }

  // Error state (API unreachable)
  if (isError) {
    return (
      <div className="bg-background flex h-screen items-center justify-center px-4">
        <div className="max-w-md space-y-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Connection Error</AlertTitle>
            <AlertDescription>
              Cannot connect to TensorTruth API. Please ensure the backend server is
              running.
              {error instanceof Error && (
                <p className="mt-2 font-mono text-xs">{error.message}</p>
              )}
            </AlertDescription>
          </Alert>
          <Button
            onClick={() => window.location.reload()}
            className="w-full"
            variant="outline"
          >
            Retry Connection
          </Button>
        </div>
      </div>
    );
  }

  // Critical infrastructure failure
  if (status && !status.ready) {
    return (
      <div className="bg-background flex h-screen items-center justify-center px-4">
        <div className="max-w-md space-y-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Initialization Failed</AlertTitle>
            <AlertDescription>
              Critical infrastructure setup failed. Please check the logs and try again.
            </AlertDescription>
          </Alert>
          {status.warnings.map((warning, idx) => (
            <Alert key={idx} variant="default">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{warning}</AlertDescription>
            </Alert>
          ))}
          <Button
            onClick={() => window.location.reload()}
            className="w-full"
            variant="outline"
          >
            Retry Initialization
          </Button>
        </div>
      </div>
    );
  }

  // Setup required for optional resources
  if (
    status &&
    ((!status.indexes_ok && !indexesSkipped) ||
      !status.models_ok ||
      !status.ollama_running)
  ) {
    return (
      <div className="bg-background flex h-screen items-center justify-center px-4">
        <div className="w-full max-w-2xl space-y-6">
          {/* Header */}
          <div className="text-center">
            <div className="mb-4 flex items-center justify-center gap-3">
              <img src="/logo.png" alt="TensorTruth" className="h-12 w-12" />
            </div>
            <h1 className="mb-2 text-3xl font-bold">Welcome to TensorTruth</h1>
            <p className="text-muted-foreground">
              Let's get you set up with the resources you need
            </p>
          </div>

          {/* Indexes Section */}
          {!status.indexes_ok && (
            <div className="bg-card rounded-lg border p-6">
              <div className="mb-4 flex items-start gap-3">
                <Download className="text-muted-foreground mt-1 h-5 w-5" />
                <div className="flex-1">
                  <h3 className="mb-1 text-lg font-semibold">Knowledge Base Indexes</h3>
                  <p className="text-muted-foreground text-sm">
                    Download pre-built vector indexes from HuggingFace Hub to enable RAG
                    queries over documentation and research papers.
                  </p>
                </div>
              </div>
              <Button
                onClick={handleDownloadIndexes}
                disabled={isDownloading}
                className="w-full"
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="mr-2 h-4 w-4" />
                    Download Indexes
                  </>
                )}
              </Button>
              {isDownloading && (
                <div className="mt-4 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Download in progress...</span>
                    <span className="text-muted-foreground font-mono">
                      {downloadElapsedSeconds}s
                    </span>
                  </div>
                  <div className="bg-secondary relative h-2 overflow-hidden rounded-full">
                    <div className="from-primary/50 via-primary to-primary/50 absolute inset-0 animate-pulse bg-gradient-to-r" />
                    <div className="via-primary/30 animate-shimmer absolute inset-0 bg-gradient-to-r from-transparent to-transparent" />
                  </div>
                  <p className="text-muted-foreground text-xs">
                    This may take a few minutes. The indexes are approximately 3.9GB.
                  </p>
                </div>
              )}
              {!isDownloading && (
                <div className="mt-3 text-center">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleSkipIndexes}
                    className="text-muted-foreground text-xs"
                  >
                    Continue without indexes
                  </Button>
                  <p className="text-muted-foreground mt-1 text-xs">
                    You can download indexes later from Settings (gear icon, top-right).
                  </p>
                </div>
              )}
            </div>
          )}

          {status.indexes_ok && (
            <div className="bg-card rounded-lg border p-6">
              <div className="flex items-center gap-3 text-green-600 dark:text-green-400">
                <CheckCircle2 className="h-5 w-5" />
                <span className="font-semibold">Indexes Ready</span>
              </div>
            </div>
          )}

          {/* Embedding Model Mismatch Warning */}
          {status.embedding_mismatch && (
            <Alert className="border-amber-500 bg-amber-50 dark:bg-amber-950/20">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
              <AlertTitle className="text-amber-700 dark:text-amber-400">
                Embedding Model Mismatch
              </AlertTitle>
              <AlertDescription className="text-amber-600 dark:text-amber-300">
                <p className="mb-2">{status.embedding_mismatch.message}</p>
                <div className="text-sm">
                  <p>
                    <strong>Configured:</strong>{" "}
                    <code className="rounded bg-amber-100 px-1 dark:bg-amber-900/50">
                      {status.embedding_mismatch.config_model}
                    </code>
                  </p>
                  <p>
                    <strong>Available:</strong>{" "}
                    {status.embedding_mismatch.available_model_ids.map((id, i) => (
                      <code
                        key={id}
                        className="rounded bg-amber-100 px-1 dark:bg-amber-900/50"
                      >
                        {id}
                        {i < status.embedding_mismatch!.available_model_ids.length - 1 &&
                          ", "}
                      </code>
                    ))}
                  </p>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Models Section */}
          {!status.ollama_running && (
            <div className="bg-card rounded-lg border p-6">
              <div className="mb-3 flex items-start gap-3">
                <WifiOff className="text-destructive mt-1 h-5 w-5" />
                <div className="flex-1">
                  <h3 className="mb-1 text-lg font-semibold">Ollama Not Reachable</h3>
                  <p className="text-muted-foreground mb-3 text-sm">
                    TensorTruth cannot connect to Ollama. Make sure it is installed and
                    running.
                  </p>
                  <div className="text-muted-foreground space-y-1 text-sm">
                    <p>
                      Start Ollama:{" "}
                      <code className="bg-muted rounded px-1.5 py-0.5 text-xs">
                        ollama serve
                      </code>
                    </p>
                    <p>
                      To use a custom URL, set{" "}
                      <code className="bg-muted rounded px-1.5 py-0.5 text-xs">
                        ollama.base_url
                      </code>{" "}
                      in{" "}
                      <code className="bg-muted rounded px-1.5 py-0.5 text-xs">
                        ~/.tensortruth/config.yaml
                      </code>
                    </p>
                  </div>
                </div>
              </div>
              <Button
                onClick={() => window.location.reload()}
                variant="outline"
                className="w-full"
              >
                Retry
              </Button>
            </div>
          )}

          {status.ollama_running && !status.models_ok && (
            <div className="bg-card rounded-lg border p-6">
              <div className="mb-3 flex items-start gap-3">
                <Server className="text-muted-foreground mt-1 h-5 w-5" />
                <div className="flex-1">
                  <h3 className="mb-1 text-lg font-semibold">No Ollama Models Found</h3>
                  <p className="text-muted-foreground mb-3 text-sm">
                    Ollama is running but has no models installed. Pull any model to get
                    started.
                  </p>
                  <div className="text-muted-foreground space-y-1 text-sm">
                    <p>
                      Example:{" "}
                      <code className="bg-muted rounded px-1.5 py-0.5 text-xs">
                        ollama pull llama3.2
                      </code>{" "}
                      or{" "}
                      <code className="bg-muted rounded px-1.5 py-0.5 text-xs">
                        ollama pull qwen2.5:7b
                      </code>
                    </p>
                    <p className="text-xs">
                      Models with tool support (llama3.1+, qwen2.5+, etc.) enable agentic
                      orchestration — web search, MCP tools, and autonomous reasoning.
                    </p>
                  </div>
                </div>
              </div>
              <Button
                onClick={() => window.location.reload()}
                variant="outline"
                className="w-full"
              >
                Reload after pulling
              </Button>
            </div>
          )}

          {status.models_ok && (
            <div className="bg-card rounded-lg border p-6">
              <div className="flex items-center gap-3 text-green-600 dark:text-green-400">
                <CheckCircle2 className="h-5 w-5" />
                <span className="font-semibold">
                  Ollama ready — {status.models_status.available.length} model
                  {status.models_status.available.length !== 1 ? "s" : ""} available
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Shouldn't reach here (status.ready would have triggered onComplete)
  return null;
}
