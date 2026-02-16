import { useState, useRef, useEffect } from "react";
import { Send, Square, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useModels, useConfig, useCommandDetection } from "@/hooks";
import { ModuleSelector } from "./ModuleSelector";
import { SessionSettingsPanel } from "@/components/config";
import { CommandAutocomplete } from "./CommandAutocomplete";
import type { CommandDefinition } from "@/types/commands";
import type { DocumentInfo } from "@/api/types";

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
  /** Module names locked by the project (shown as non-toggleable). */
  lockedModules?: string[];
  /** Documents attached to the project (shown in module selector). */
  projectDocuments?: DocumentInfo[];
  /** Whether this is a project session (hides session-level system prompt). */
  isProjectSession?: boolean;
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
  lockedModules,
  projectDocuments,
  isProjectSession,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [autocompleteHasResults, setAutocompleteHasResults] = useState(false);
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: config } = useConfig();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const detection = useCommandDetection(message);

  // Show autocomplete only if command detected AND no space after command name
  // (hide when user is typing arguments, only show when typing command name)
  const commandEndPos =
    detection.commandPosition + (detection.commandName?.length || 0) + 1;
  const hasSpaceAfterCommand = message.charAt(commandEndPos) === " ";
  const showAutocomplete = detection.hasCommand && !hasSpaceAfterCommand;

  // Auto-resize textarea
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
      setMessage(""); // Autocomplete will hide automatically when message clears
    }
  };

  const handleStop = () => {
    onStop?.();
  };

  const handleCommandSelect = (command: CommandDefinition) => {
    // Replace the current command with the selected one
    if (detection.commandPosition >= 0) {
      const before = message.slice(0, detection.commandPosition);
      const after = message.slice(
        detection.commandPosition + (detection.commandName?.length || 0) + 1
      );
      // Replace with just the command name (e.g., "/web "), user adds args manually
      const commandName = command.usage.split(" ")[0];
      setMessage(`${before}${commandName} ${after}`.trim() + " ");
    } else {
      // Fallback: append command at the end
      const commandName = command.usage.split(" ")[0];
      setMessage(`${message} ${commandName} `);
    }
    // Autocomplete will hide automatically when command pattern changes
    // Focus back on textarea
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Only let autocomplete handle keys if it's open AND has results
    if (
      showAutocomplete &&
      autocompleteHasResults &&
      ["Enter", "Tab", "ArrowUp", "ArrowDown", "Escape"].includes(e.key)
    ) {
      // Let autocomplete handle these keys
      return;
    }

    // Send on Enter (without Shift)
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
        {/* Command Autocomplete */}
        <CommandAutocomplete
          input={message}
          isOpen={showAutocomplete}
          onSelect={handleCommandSelect}
          onHasResultsChange={setAutocompleteHasResults}
          onClose={() => {
            // Remove the command when user presses Escape
            if (detection.commandPosition >= 0) {
              const before = message.slice(0, detection.commandPosition);
              const after = message.slice(
                detection.commandPosition + (detection.commandName?.length || 0) + 1
              );
              setMessage((before + after).trim());
            }
            textareaRef.current?.focus();
          }}
        />

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
                lockedModules={lockedModules}
                projectDocuments={projectDocuments}
              />
            )}
            {sessionId && (
              <SessionSettingsPanel
                sessionId={sessionId}
                currentParams={sessionParams}
                disabled={isStreaming}
                hideSystemPrompt={isProjectSession}
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
                    {selectedModel || config?.llm.default_model || "Model"}
                  </span>
                </SelectTrigger>
                <SelectContent position="popper" side="top" className="max-h-[300px]">
                  <SelectItem value="__none__">
                    <span className="text-muted-foreground">
                      Default ({config?.llm.default_model || "..."})
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
