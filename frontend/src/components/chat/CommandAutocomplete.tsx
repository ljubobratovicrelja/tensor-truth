/**
 * Command autocomplete dropdown component
 *
 * Shows available commands when user types "/" in the chat input.
 * Features:
 * - Filters commands as user types the command name
 * - Keyboard navigation (↑/↓ arrows, Enter to select, Esc to close)
 * - Shows command description and usage
 * - Positioned near the cursor
 */

import { useEffect, useState, useCallback } from "react";
import { Command as CommandIcon, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { useCommands, useCommandDetection } from "@/hooks";
import type { CommandDefinition } from "@/types/commands";

interface CommandAutocompleteProps {
  /** Current input text */
  input: string;

  /** Whether the autocomplete is open */
  isOpen: boolean;

  /** Callback when user selects a command */
  onSelect: (command: CommandDefinition) => void;

  /** Callback when user wants to close autocomplete */
  onClose: () => void;

  /** Callback to inform parent whether autocomplete has results */
  onHasResultsChange?: (hasResults: boolean) => void;

  /** Optional class name for the container */
  className?: string;
}

export function CommandAutocomplete({
  input,
  isOpen,
  onSelect,
  onClose,
  onHasResultsChange,
  className,
}: CommandAutocompleteProps) {
  const { data: commands = [], isLoading } = useCommands();
  const detection = useCommandDetection(input);
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Filter commands based on what user has typed
  const filteredCommands = commands.filter((cmd) => {
    if (!detection.hasCommand || !detection.commandName) {
      return true; // Show all commands if user just typed "/"
    }

    const query = detection.commandName.toLowerCase();
    const nameMatch = cmd.name.toLowerCase().startsWith(query);
    const aliasMatch = cmd.aliases.some((alias) => alias.toLowerCase().startsWith(query));

    return nameMatch || aliasMatch;
  });

  // Notify parent when results change
  useEffect(() => {
    onHasResultsChange?.(filteredCommands.length > 0);
  }, [filteredCommands.length, onHasResultsChange]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => {
            // Clamp to valid range and reset to 0 if out of bounds
            const clamped = prev >= filteredCommands.length ? 0 : prev;
            return clamped < filteredCommands.length - 1 ? clamped + 1 : 0;
          });
          break;

        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => {
            // Clamp to valid range and reset to last if out of bounds
            const clamped =
              prev >= filteredCommands.length ? filteredCommands.length - 1 : prev;
            return clamped > 0 ? clamped - 1 : filteredCommands.length - 1;
          });
          break;

        case "Enter":
        case "Tab":
          if (filteredCommands.length > 0) {
            e.preventDefault();
            onSelect(filteredCommands[selectedIndex]);
          }
          break;

        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [isOpen, filteredCommands, selectedIndex, onSelect, onClose]
  );

  // Attach keyboard listener
  useEffect(() => {
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [isOpen, handleKeyDown]);

  // Don't render if not open or no commands
  if (!isOpen || (!isLoading && filteredCommands.length === 0)) {
    return null;
  }

  return (
    <div
      className={cn(
        "border-border bg-popover absolute bottom-full left-0 z-50 mb-2 w-full max-w-md rounded-lg border shadow-lg",
        "animate-in fade-in slide-in-from-bottom-2 duration-200",
        className
      )}
    >
      {/* Header */}
      <div className="border-border flex items-center gap-2 border-b px-3 py-2">
        <Search className="text-muted-foreground h-4 w-4" />
        <span className="text-foreground text-sm font-medium">
          {detection.commandName
            ? `Commands matching "/${detection.commandName}"`
            : "Available Commands"}
        </span>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="text-muted-foreground px-3 py-4 text-center text-sm">
          Loading commands...
        </div>
      )}

      {/* Command list */}
      {!isLoading && filteredCommands.length > 0 && (
        <div className="max-h-64 overflow-y-auto">
          {filteredCommands.map((cmd, index) => (
            <button
              key={cmd.name}
              onClick={() => onSelect(cmd)}
              className={cn(
                "w-full px-3 py-2 text-left transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                "focus:bg-accent focus:text-accent-foreground focus:outline-none",
                selectedIndex === index && "bg-accent text-accent-foreground"
              )}
              onMouseEnter={() => setSelectedIndex(index)}
            >
              <div className="flex items-start gap-2">
                <CommandIcon className="text-muted-foreground mt-0.5 h-4 w-4 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  {/* Command name and aliases */}
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-sm font-semibold">
                      {cmd.usage.split(" ")[0]}
                    </span>
                    {cmd.aliases.length > 0 && (
                      <span className="text-muted-foreground text-xs">
                        ({cmd.aliases.map((a) => `/${a}`).join(", ")})
                      </span>
                    )}
                  </div>

                  {/* Description */}
                  <p className="text-muted-foreground mt-0.5 text-xs">
                    {cmd.description}
                  </p>

                  {/* Usage hint */}
                  {cmd.usage.includes("<") && (
                    <p className="text-muted-foreground/70 mt-1 font-mono text-xs">
                      {cmd.usage}
                    </p>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Footer hint */}
      <div className="border-border text-muted-foreground border-t px-3 py-1.5 text-xs">
        <span className="font-medium">↑↓</span> Navigate{" "}
        <span className="font-medium">Tab</span> or{" "}
        <span className="font-medium">Enter</span> Select{" "}
        <span className="font-medium">Esc</span> Close
      </div>
    </div>
  );
}
