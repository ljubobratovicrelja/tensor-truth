import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square, Bot, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectItem, SelectTrigger } from "@/components/ui/select";
import { ModelSelectContent, decodeModelValue } from "./ModelSelectContent";
import { cn } from "@/lib/utils";
import {
  useModels,
  useConfig,
  useCommandDetection,
  useThinkingSupport,
  useModelActions,
} from "@/hooks";
import { ModuleSelector } from "./ModuleSelector";
import { ThinkingSelect } from "./ThinkingSelect";
import { SessionSettingsPanel } from "@/components/config";
import { CommandAutocomplete } from "./CommandAutocomplete";
import type { CommandDefinition } from "@/types/commands";
import type { DocumentInfo } from "@/api/types";
import type { AttachedImage } from "@/hooks/useWebSocket";

const MAX_IMAGES = 4;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp"];

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
  isProjectSession,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
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

  // --- Image attachment helpers ---
  const addImageFiles = useCallback(
    (files: File[]) => {
      const imageFiles = files.filter(
        (f) => ALLOWED_IMAGE_TYPES.includes(f.type) && f.size <= MAX_IMAGE_SIZE
      );
      const remaining = MAX_IMAGES - attachedImages.length;
      const toAdd = imageFiles.slice(0, remaining);

      const newImages: AttachedImage[] = toAdd.map((file) => ({
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
        mimetype: file.type,
      }));
      setAttachedImages((prev) => [...prev, ...newImages]);
    },
    [attachedImages.length]
  );

  const removeImage = useCallback((id: string) => {
    setAttachedImages((prev) => {
      const img = prev.find((i) => i.id === id);
      if (img) URL.revokeObjectURL(img.previewUrl);
      return prev.filter((i) => i.id !== id);
    });
  }, []);

  // Cleanup preview URLs on unmount
  useEffect(() => {
    return () => {
      attachedImages.forEach((img) => URL.revokeObjectURL(img.previewUrl));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      addImageFiles(files);
    },
    [addImageFiles]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      const imageFiles: File[] = [];
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) imageFiles.push(file);
        }
      }
      if (imageFiles.length > 0) {
        addImageFiles(imageFiles);
      }
    },
    [addImageFiles]
  );

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
    if ((trimmed || attachedImages.length > 0) && !isStreaming) {
      onSend(trimmed, attachedImages.length > 0 ? attachedImages : undefined);
      setMessage("");
      setAttachedImages([]);
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

  const canSend =
    (message.trim().length > 0 || attachedImages.length > 0) && !isStreaming;

  return (
    <div className="space-y-2">
      <div
        className={cn(
          "bg-muted/50 border-input relative flex flex-col rounded-2xl border",
          isDragOver && "ring-primary ring-2"
        )}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
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

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={placeholder}
          className={cn(
            "w-full resize-none bg-transparent px-4 pt-4 pb-2 text-base",
            "placeholder:text-muted-foreground focus:outline-none"
          )}
          rows={1}
        />

        {/* Image preview strip */}
        {attachedImages.length > 0 && (
          <div className="flex gap-2 overflow-x-auto px-3 py-2">
            {attachedImages.map((img) => (
              <div key={img.id} className="relative shrink-0">
                <img
                  src={img.previewUrl}
                  alt={img.file.name}
                  className="h-16 w-16 rounded-lg border object-cover"
                />
                <button
                  onClick={() => removeImage(img.id)}
                  className="bg-background/80 hover:bg-destructive hover:text-destructive-foreground absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full border text-xs shadow-sm"
                  title="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Bottom toolbar */}
        <div className="flex items-center justify-between px-2 pb-2">
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
      <p className="text-muted-foreground text-center text-xs">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
