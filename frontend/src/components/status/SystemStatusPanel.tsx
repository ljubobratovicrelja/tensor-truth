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
import { Activity, Cpu, Server } from "lucide-react";
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
import { useDevices, useOllamaStatus, useConfig } from "@/hooks";

interface SystemStatusPanelProps {
  /** Optional trigger element (if not provided, uses default button) */
  trigger?: React.ReactNode;
}

export function SystemStatusPanel({ trigger }: SystemStatusPanelProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  const { data: devicesData, isLoading: devicesLoading } = useDevices();
  const { data: ollamaData, isLoading: ollamaLoading } = useOllamaStatus(isOpen);
  const { data: config } = useConfig();

  const devices = devicesData?.devices || [];

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
                        <p className="font-mono text-sm">{model.name}</p>
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
                {config.models.default_fallback_model && (
                  <div>
                    <span className="font-medium">Fallback Model:</span>{" "}
                    <span className="font-mono">
                      {config.models.default_fallback_model}
                    </span>
                  </div>
                )}
                {config.models.default_agent_reasoning_model && (
                  <div>
                    <span className="font-medium">Agent Reasoning:</span>{" "}
                    <span className="font-mono">
                      {config.models.default_agent_reasoning_model}
                    </span>
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
