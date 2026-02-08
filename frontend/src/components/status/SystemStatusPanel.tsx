/**
 * System Status Panel Component
 *
 * Comprehensive system status panel showing:
 * - Memory usage (via MemoryMonitor)
 * - Available devices
 * - Current configuration
 * - Ollama runtime status
 */

import * as React from "react";
import { Activity, Cpu, Server, Database, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { MemoryMonitor } from "./MemoryMonitor";
import {
  useDevices,
  useOllamaStatus,
  useRAGStatus,
  useSessionStats,
  useConfig,
} from "@/hooks";
import { useSessionStore } from "@/stores/sessionStore";

interface SystemStatusPanelProps {
  /** Optional trigger element (if not provided, uses default button) */
  trigger?: React.ReactNode;
}

export function SystemStatusPanel({ trigger }: SystemStatusPanelProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const { activeSessionId } = useSessionStore();

  const { data: devicesData, isLoading: devicesLoading } = useDevices();
  const { data: ollamaData, isLoading: ollamaLoading } = useOllamaStatus(isOpen);
  const { data: ragData, isLoading: ragLoading } = useRAGStatus(isOpen);
  const { data: sessionStats, isLoading: sessionStatsLoading } = useSessionStats(
    activeSessionId,
    isOpen && !!activeSessionId
  );
  const { data: config } = useConfig();

  const devices = devicesData?.devices || [];

  // Helper to format model name (extract short name after last /)
  const formatModelName = (name: string | null) => {
    if (!name) return "Unknown";
    return name.split("/").pop() || name;
  };

  // Helper to format large numbers with K suffix
  const formatNumber = (n: number) => {
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
  };

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        {trigger || (
          <Button variant="ghost" size="sm" className="gap-2">
            <Activity className="h-4 w-4" />
            <span className="hidden sm:inline">System Status</span>
          </Button>
        )}
      </SheetTrigger>

      <SheetContent className="w-[400px] overflow-y-auto sm:w-[540px]">
        <SheetHeader>
          <SheetTitle>System Status</SheetTitle>
          <SheetDescription>Monitor system resources and configuration</SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Memory Monitor */}
          <section>
            <MemoryMonitor enabled={isOpen} />
          </section>

          <Separator />

          {/* RAG Status */}
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Database className="text-muted-foreground h-4 w-4" />
              <h3 className="text-foreground text-sm font-semibold">RAG Engine</h3>
            </div>

            {ragLoading ? (
              <div className="bg-muted h-20 w-full animate-pulse rounded" />
            ) : !ragData?.active ? (
              <div className="space-y-2">
                <Badge variant="secondary">Inactive</Badge>
                <p className="text-muted-foreground text-xs">
                  RAG models will load when you start a session with modules
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Badge variant="default">Active</Badge>
                  {ragData.total_memory_gb > 0 && (
                    <span className="text-muted-foreground text-xs">
                      {ragData.total_memory_gb.toFixed(2)} GB total
                    </span>
                  )}
                </div>

                {/* Embedder */}
                {ragData.embedder.loaded && (
                  <div className="rounded border p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground text-xs font-medium">
                        Embedder
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {ragData.embedder.device?.toUpperCase() || "CPU"}
                      </Badge>
                    </div>
                    <p className="mt-1 font-mono text-sm">
                      {formatModelName(ragData.embedder.model_name)}
                    </p>
                    {ragData.embedder.memory_gb != null &&
                      ragData.embedder.memory_gb > 0 && (
                        <p className="text-muted-foreground mt-0.5 text-xs">
                          {ragData.embedder.memory_gb.toFixed(2)} GB
                        </p>
                      )}
                  </div>
                )}

                {/* Reranker */}
                {ragData.reranker.loaded && (
                  <div className="rounded border p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground text-xs font-medium">
                        Reranker
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {ragData.reranker.device?.toUpperCase() || "CPU"}
                      </Badge>
                    </div>
                    <p className="mt-1 font-mono text-sm">
                      {formatModelName(ragData.reranker.model_name)}
                    </p>
                    {ragData.reranker.memory_gb != null &&
                      ragData.reranker.memory_gb > 0 && (
                        <p className="text-muted-foreground mt-0.5 text-xs">
                          {ragData.reranker.memory_gb.toFixed(2)} GB
                        </p>
                      )}
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Chat Session Stats - only shown when there's an active session */}
          {activeSessionId && (
            <>
              <Separator />

              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <MessageSquare className="text-muted-foreground h-4 w-4" />
                  <h3 className="text-foreground text-sm font-semibold">
                    Active Session
                  </h3>
                </div>

                {sessionStatsLoading ? (
                  <div className="bg-muted h-16 w-full animate-pulse rounded" />
                ) : sessionStats ? (
                  <div className="space-y-2">
                    {/* Model & Context */}
                    <div className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground text-xs font-medium">
                          LLM Model
                        </span>
                        {sessionStats.context_length > 0 && (
                          <Badge variant="outline" className="text-xs">
                            {formatNumber(sessionStats.context_length)} ctx
                          </Badge>
                        )}
                      </div>
                      <p className="mt-1 font-mono text-sm">
                        {sessionStats.model_name || "Unknown"}
                      </p>
                    </div>

                    {/* History Stats - Compiled (what LLM sees) */}
                    <div className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground text-xs font-medium">
                          Context Sent to LLM
                        </span>
                        {sessionStats.max_history_turns > 0 && (
                          <Badge
                            variant="outline"
                            className="cursor-help text-xs"
                            title="1 turn = 1 user query + 1 assistant response"
                          >
                            max {sessionStats.max_history_turns} turns
                          </Badge>
                        )}
                      </div>
                      <div className="mt-1 flex items-baseline gap-3">
                        <span className="text-foreground text-sm font-medium">
                          {formatNumber(sessionStats.compiled_history_chars)} chars
                        </span>
                        <span
                          className="text-muted-foreground cursor-help text-xs"
                          title="Estimated tokens (characters ÷ 4)"
                        >
                          ~{formatNumber(sessionStats.compiled_history_tokens_estimate)}{" "}
                          tokens
                        </span>
                      </div>
                      <div className="text-muted-foreground mt-1 text-xs">
                        {sessionStats.compiled_history_messages} of{" "}
                        {sessionStats.history_messages} messages included
                      </div>
                    </div>
                  </div>
                ) : null}
              </section>
            </>
          )}

          <Separator />

          {/* Available Devices */}
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Cpu className="text-muted-foreground h-4 w-4" />
              <h3 className="text-foreground text-sm font-semibold">Available Devices</h3>
            </div>

            {devicesLoading ? (
              <div className="flex gap-2">
                {[1, 2].map((i) => (
                  <div key={i} className="bg-muted h-6 w-16 animate-pulse rounded" />
                ))}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {devices.map((device) => (
                  <Badge key={device} variant="secondary">
                    {device.toUpperCase()}
                  </Badge>
                ))}
              </div>
            )}

            {config && (
              <div className="text-muted-foreground mt-2 space-y-1 text-xs">
                <p>
                  <span className="font-medium">Default RAG Device:</span>{" "}
                  {config.rag.default_device || "auto"}
                </p>
              </div>
            )}
          </section>

          <Separator />

          {/* Ollama Status */}
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Server className="text-muted-foreground h-4 w-4" />
              <h3 className="text-foreground text-sm font-semibold">Ollama Runtime</h3>
            </div>

            {ollamaLoading ? (
              <div className="bg-muted h-16 w-full animate-pulse rounded" />
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant={ollamaData?.running ? "default" : "secondary"}>
                    {ollamaData?.running ? "Running" : "Idle"}
                  </Badge>
                  {ollamaData?.running && ollamaData.models.length > 0 && (
                    <span className="text-muted-foreground text-xs">
                      {ollamaData.models.length} model(s) loaded
                    </span>
                  )}
                </div>

                {ollamaData?.running && ollamaData.models.length > 0 && (
                  <div className="text-muted-foreground space-y-2 text-xs">
                    {ollamaData.models.map((model, idx) => (
                      <div key={idx} className="rounded border p-2">
                        <div className="flex items-center justify-between">
                          <p className="font-mono text-sm">{model.name}</p>
                          {model.context_length != null && model.context_length > 0 && (
                            <Badge variant="outline" className="text-xs">
                              {formatNumber(model.context_length)} ctx
                            </Badge>
                          )}
                        </div>
                        <div className="mt-1 flex gap-3">
                          {model.size_vram_gb > 0 && (
                            <span>VRAM: {model.size_vram_gb.toFixed(2)} GB</span>
                          )}
                          {model.size_gb > 0 && (
                            <span>Size: {model.size_gb.toFixed(2)} GB</span>
                          )}
                          {model.parameters && <span>{model.parameters}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {!ollamaData?.running && (
                  <p className="text-muted-foreground text-xs">
                    No models currently loaded
                  </p>
                )}
              </div>
            )}

            {config && (
              <div className="text-muted-foreground mt-2 space-y-1 text-xs">
                <p>
                  <span className="font-medium">Base URL:</span>{" "}
                  {config.ollama.base_url || "http://localhost:11434"}
                </p>
              </div>
            )}
          </section>

          <Separator />

          {/* Default Models */}
          {config && (
            <section className="space-y-3">
              <h3 className="text-foreground text-sm font-semibold">Default Models</h3>

              <div className="text-muted-foreground space-y-1.5 text-xs">
                <div>
                  <span className="font-medium">RAG Model:</span>{" "}
                  <span className="font-mono">{config.models.default_rag_model}</span>
                </div>
                {config.agent.router_model && (
                  <div>
                    <span className="font-medium">Agent Reasoning:</span>{" "}
                    <span className="font-mono">{config.agent.router_model}</span>
                  </div>
                )}
                {config.agent.function_agent_model && (
                  <div>
                    <span className="font-medium">Function Agent:</span>{" "}
                    <span className="font-mono">{config.agent.function_agent_model}</span>
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
