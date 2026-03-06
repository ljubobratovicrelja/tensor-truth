/**
 * Hook for detecting commands in user input
 *
 * Commands must be preceded by whitespace or appear at the start of the input.
 * Pattern: /commandName optional arguments
 *
 * Examples:
 * - "/help" → command: "help", args: ""
 * - "/web search query" → command: "web", args: "search query"
 * - "I need help - /help please" → command: "help", args: "please"
 */

import { useMemo } from "react";
import type { CommandDetection } from "@/types/commands";

/**
 * Detect command in user input string
 *
 * @param input - User input string to analyze
 * @returns CommandDetection object with command info
 */
export function useCommandDetection(input: string): CommandDetection {
  return useMemo(() => {
    // Match pattern: /commandName optionalArgs
    // (?<!\S) ensures / is at start of string or after whitespace
    // \w+ matches command name (letters, numbers, underscore)
    // (?:\s+(.+))? optionally matches space + remaining text
    const match = input.match(/(?<!\S)\/(\w+)(?:\s+(.+))?/);

    if (!match) {
      return {
        hasCommand: false,
        commandName: null,
        commandArgs: "",
        commandPosition: -1,
      };
    }

    return {
      hasCommand: true,
      commandName: match[1], // First capture group: command name
      commandArgs: match[2] || "", // Second capture group: args (or empty)
      commandPosition: match.index ?? -1, // Position in string
    };
  }, [input]);
}
