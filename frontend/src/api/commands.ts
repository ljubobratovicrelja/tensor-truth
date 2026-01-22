/**
 * Commands API client
 *
 * Provides functions to interact with the commands API endpoints.
 */

import { apiGet } from "./client";
import type { CommandDefinition } from "@/types/commands";

/**
 * Response from GET /api/commands
 */
interface CommandsResponse {
  commands: CommandDefinition[];
}

/**
 * Fetch all available commands from the backend
 *
 * @returns Promise resolving to array of command definitions
 */
export async function getCommands(): Promise<CommandDefinition[]> {
  const response = await apiGet<CommandsResponse>("/commands");
  return response.commands;
}
