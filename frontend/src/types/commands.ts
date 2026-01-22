/**
 * Command system type definitions
 *
 * Commands are tool/agent triggers that can appear anywhere in user input.
 * The backend detects and executes commands, streaming responses back via WebSocket.
 */

/**
 * Command definition from backend API
 */
export interface CommandDefinition {
  /** Primary command name (e.g., "web", "help") */
  name: string;

  /** Alternative names for the command (e.g., ["search", "websearch"]) */
  aliases: string[];

  /** Human-readable description of what the command does */
  description: string;

  /** Usage string with example (e.g., "/web <search query>") */
  usage: string;
}

/**
 * Command suggestion for autocomplete with match scoring
 */
export interface CommandSuggestion extends CommandDefinition {
  /** Match score for ranking suggestions (0-1, higher is better) */
  matchScore: number;
}

/**
 * Result of command detection in user input
 */
export interface CommandDetection {
  /** Whether a command was detected in the input */
  hasCommand: boolean;

  /** The command name (without /) if detected, null otherwise */
  commandName: string | null;

  /** Arguments after the command name, or empty string */
  commandArgs: string;

  /** Character position where command starts, or -1 */
  commandPosition: number;
}
