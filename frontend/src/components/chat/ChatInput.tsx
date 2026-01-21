import { useState, useRef, useEffect } from "react";
import { Send, Square, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useModels, useConfig } from "@/hooks";
import { ModuleSelector } from "./ModuleSelector";
import { SessionSettingsPanel } from "@/components/config";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  placeholder?: string;
  selectedModules?: string[];
  onModulesChange?: (modules: string[]) => void;
  selectedModel?: string;
  onModelChange?: (model: string | null) => void;
  sessionId?: string;
  sessionParams?: Record<string, unknown>;
}

export function ChatInput({
  onSend,
  onStop,
  isStreaming = false,
  placeholder = "Type your message...",
  selectedModules = [],
  onModulesChange,
  selectedModel,
  onModelChange,
  sessionId,
  sessionParams = {},
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: config } = useConfig();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const scrollHeight = textareaRef.current.scrollHeight;
      // Clamp between min (80px) and max (200px)
      textareaRef.current.style.height = `${Math.max(80, Math.min(scrollHeight, 200))}px`;
    }
  }, [message]);

  const handleSend = () => {
    const trimmed = message.trim();
    if (trimmed && !isStreaming) {
      onSend(trimmed);
      setMessage("");
    }
  };

  const handleStop = () => {
    onStop?.();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Send on Enter (without Shift) or Cmd/Ctrl+Enter
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) {
        handleSend();
      }
    }
  };

  const canSend = message.trim().length > 0 && !isStreaming;

  return (
    <div className="space-y-2">
      <div className="bg-muted/50 border-input relative rounded-2xl border">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isStreaming}
          className={cn(
            "w-full resize-none bg-transparent px-4 pt-4 pb-14 text-base",
            "placeholder:text-muted-foreground focus:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50"
          )}
          rows={1}
        />

        {/* Bottom toolbar */}
        <div className="absolute right-2 bottom-2 left-2 flex items-center justify-between">
          {/* Left side - module selector, session settings, and model selector */}
          <div className="flex items-center gap-1">
            {onModulesChange && (
              <ModuleSelector
                selectedModules={selectedModules}
                onModulesChange={onModulesChange}
                disabled={isStreaming}
                embeddingModel={sessionParams.embedding_model as string | undefined}
              />
            )}
            {sessionId && (
              <SessionSettingsPanel
                sessionId={sessionId}
                currentParams={sessionParams}
                disabled={isStreaming}
              />
            )}
            {onModelChange && (
              <Select
                value={selectedModel || "__none__"}
                onValueChange={(value) =>
                  onModelChange(value === "__none__" ? null : value)
                }
                disabled={isStreaming}
              >
                <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
                  <Bot className="h-3.5 w-3.5" />
                  <span className="text-xs">
                    {selectedModel || config?.models.default_rag_model || "Model"}
                  </span>
                </SelectTrigger>
                <SelectContent position="popper" side="top" className="max-h-[300px]">
                  <SelectItem value="__none__">
                    <span className="text-muted-foreground">
                      Default ({config?.models.default_rag_model || "..."})
                    </span>
                  </SelectItem>
                  {modelsLoading ? (
                    <SelectItem value="loading" disabled>
                      Loading...
                    </SelectItem>
                  ) : (
                    modelsData?.models
                      .slice()
                      .sort((a, b) => a.name.localeCompare(b.name))
                      .map((model) => (
                        <SelectItem key={model.name} value={model.name}>
                          {model.name}
                        </SelectItem>
                      ))
                  )}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* Right side - send/stop button */}
          <div className="flex items-center gap-2">
            {isStreaming ? (
              <Button
                onClick={handleStop}
                size="icon"
                variant="destructive"
                className="h-9 w-9 rounded-xl"
                title="Stop generating"
              >
                <Square className="h-4 w-4 fill-current" />
              </Button>
            ) : (
              <Button
                onClick={handleSend}
                disabled={!canSend}
                size="icon"
                className="h-9 w-9 rounded-xl"
                title="Send message"
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
      <p className="text-muted-foreground text-center text-xs">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
