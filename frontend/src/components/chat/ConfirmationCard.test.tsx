import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfirmationCard } from "./ConfirmationCard";
import type { StreamConfirmationRequest } from "@/api/types";

const resolveMock = vi.fn();
const approveMock = vi.fn();

vi.mock("@/api/client", () => ({
  approveToolConfirmation: (...a: unknown[]) => approveMock(...a),
  rejectToolConfirmation: vi.fn(),
  resolveToolConfirmation: (...a: unknown[]) => resolveMock(...a),
  getToolConfirmationStatus: vi.fn().mockResolvedValue(null),
}));

function makeRequest(details: Record<string, unknown>): StreamConfirmationRequest {
  return {
    type: "confirmation_request",
    confirmation_id: "c-1",
    tool_name: "add_arxiv_paper",
    action_type: "add_document",
    title: "Add arXiv paper 2511.08544",
    summary: "Download and index",
    details,
  };
}

function renderCard(request: StreamConfirmationRequest) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ConfirmationCard request={request} isLive />
    </QueryClientProvider>
  );
}

describe("ConfirmationCard alternatives", () => {
  it("renders a third button for a tool-offered alternative", async () => {
    const user = userEvent.setup();
    renderCard(
      makeRequest({
        alternatives: [{ id: "context", label: "Add directly to context" }],
      })
    );

    const alt = screen.getByRole("button", { name: /Add directly to context/ });
    expect(alt).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Approve/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Reject/ })).toBeInTheDocument();

    await user.click(alt);
    expect(resolveMock).toHaveBeenCalledWith("c-1", "context");
    expect(approveMock).not.toHaveBeenCalled();
  });

  it("shows only approve/reject when no alternative is offered", () => {
    renderCard(makeRequest({}));
    expect(screen.queryByRole("button", { name: /Add directly/ })).toBeNull();
    expect(screen.getByRole("button", { name: /Approve/ })).toBeInTheDocument();
  });
});
