import { useState, useRef, useEffect } from "react";
import { Send, Square, Bot, ImagePlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectItem, SelectTrigger } from "@/components/ui/select";
import { ModelSelectContent, decodeModelValue } from "./ModelSelectContent";
import { ImagePreviewStrip } from "./ImagePreviewStrip";
import { cn } from "@/lib/utils";
import {
  useModels,
  useConfig,
  useCommandDetection,
  useThinkingSupport,
  useModelActions,
  useImageAttachment,
} from "@/hooks";
import { ModuleSelector } from "./ModuleSelector";
import { ThinkingSelect } from "./ThinkingSelect";
import { SessionSettingsPanel } from "@/components/config";
import { CommandAutocomplete } from "./CommandAutocomplete";
import type { CommandDefinition } from "@/types/commands";
import type { DocumentInfo } from "@/api/types";
import type { AttachedImage } from "@/hooks/useWebSocket";

interface ChatInputProps {
  onSend: (message: string, images?: AttachedImage[]) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  placeholder?: string;
  selectedModules?: string[];
  onModulesChange?: (modules: string[]) => void;
  selectedModel?: string;
  onModelChange?: (model: string | null) => void;
  thinking?: string;
  onThinkingChange?: (thinking: string) => void;
  sessionId?: string;
  sessionParams?: Record<string, unknown>;
  /** Module names locked by the project (shown as non-toggleable). */
  lockedModules?: string[];
  /** Documents attached to the project (shown in module selector). */
  projectDocuments?: DocumentInfo[];
  /** Indexed documents in the current session (shown in module selector for non-project sessions). */
  sessionDocuments?: DocumentInfo[];
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
  thinking = "auto",
  onThinkingChange,
  sessionId,
  sessionParams = {},
  lockedModules,
  projectDocuments,
  sessionDocuments,
  isProjectSession,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const {
    attachedImages,
    isDragOver,
    dragProps,
    handlePaste,
    addImageFiles,
    removeImage,
    detach,
    hasImages,
  } = useImageAttachment();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [autocompleteHasResults, setAutocompleteHasResults] = useState(false);
  const [selectOpen, setSelectOpen] = useState(false);
  const { data: modelsData, isLoading: modelsLoading } = useModels(
    selectOpen ? 2000 : false
  );
  const { data: config } = useConfig();
  const { actionsInFlight, handleLoadModel, handleUnloadModel } = useModelActions();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const detection = useCommandDetection(message);

  const activeModelName = selectedModel
    ? decodeModelValue(selectedModel).modelName
    : config?.llm.default_model || modelsData?.models[0]?.name || "";
  const thinkingSupport = useThinkingSupport(modelsData, activeModelName);

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
      textareaRef.current.style.height = `${Math.max(48, Math.min(scrollHeight, 200))}px`;
    }
  }, [message]);

  const handleSend = () => {
    const trimmed = message.trim();
    if ((trimmed || hasImages) && !isStreaming) {
      // Detach images (clear state without revoking blob URLs) so the
      // pending message display can still use the preview URLs.
      const images = hasImages ? detach() : undefined;
      onSend(trimmed, images);
      setMessage("");
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

  const canSend = (message.trim().length > 0 || hasImages) && !isStreaming;

  return (
    <div className="space-y-1">
      <div
        className={cn("relative flex flex-col", isDragOver && "ring-primary ring-2")}
        {...dragProps}
      >
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

        {/* Image preview strip — above textarea */}
        <ImagePreviewStrip images={attachedImages} onRemove={removeImage} />

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={placeholder}
          className={cn(
            "w-full resize-none border-0 bg-transparent px-1 pt-1 pb-2 text-base shadow-none",
            "placeholder:text-muted-foreground focus:ring-0 focus:outline-none"
          )}
          rows={1}
        />

        {/* Hidden file input for image attachment */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            if (files.length > 0) addImageFiles(files);
            e.target.value = "";
          }}
        />

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between px-0 pb-1">
          {/* Left side - module selector, session settings, and model selector */}
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              title="Attach image"
              disabled={isStreaming}
              onClick={() => fileInputRef.current?.click()}
            >
              <ImagePlus className="h-4 w-4" />
            </Button>
            {onModulesChange && (
              <ModuleSelector
                selectedModules={selectedModules}
                onModulesChange={onModulesChange}
                disabled={isStreaming}
                embeddingModel={sessionParams.embedding_model as string | undefined}
                lockedModules={lockedModules}
                projectDocuments={projectDocuments}
                sessionDocuments={sessionDocuments}
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
                onOpenChange={setSelectOpen}
              >
                <SelectTrigger className="hover:bg-muted h-8 w-auto gap-2 border-0 bg-transparent px-2 text-xs">
                  <Bot className="h-3.5 w-3.5" />
                  <span className="text-xs">
                    {activeModelName || config?.llm.default_model || "Model"}
                  </span>
                </SelectTrigger>
                <ModelSelectContent
                  models={modelsData?.models ?? []}
                  isLoading={modelsLoading}
                  position="popper"
                  side="top"
                  className="!max-h-[300px]"
                  onLoadModel={handleLoadModel}
                  onUnloadModel={handleUnloadModel}
                  actionsInFlight={actionsInFlight}
                  extraItems={
                    <SelectItem value="__none__">
                      <span className="text-muted-foreground">
                        Default ({activeModelName || "..."})
                      </span>
                    </SelectItem>
                  }
                />
              </Select>
            )}
            {onThinkingChange && thinkingSupport.thinking && (
              <ThinkingSelect
                value={thinking}
                onValueChange={onThinkingChange}
                disabled={isStreaming}
                supportsLevels={thinkingSupport.levels}
              />
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
      <p className="text-muted-foreground hidden text-center text-xs md:block">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
