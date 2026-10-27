import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DocumentList } from "./DocumentList";
import type { DocumentListItem } from "@/api/types";

const mockMutateAsync = vi.fn();
const mockUseDocuments = vi.fn();

vi.mock("@/hooks", () => ({
  useDocuments: (...args: unknown[]) => mockUseDocuments(...args),
  useDeleteDocument: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useBuildIndex: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useSetDocumentContext: () => ({
    mutateAsync: mockMutateAsync,
    isPending: false,
  }),
  useUploadDocument: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadText: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadUrl: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadArxiv: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadFileUrl: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useArxivLookup: () => ({
    data: undefined,
    isFetching: false,
    isError: false,
  }),
  useFileUrlInfo: () => ({
    data: undefined,
    isFetching: false,
    isError: false,
  }),
  useIndexingConfig: () => ({ data: undefined, dataUpdatedAt: 0 }),
  useUpdateIndexingConfig: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/useTasks", () => ({
  useTaskProgress: () => ({ data: undefined }),
}));

function makeDoc(overrides: Partial<DocumentListItem>): DocumentListItem {
  return {
    doc_id: "doc_abc",
    filename: "paper.pdf",
    file_size: 2048,
    page_count: 4,
    is_indexed: true,
    ...overrides,
  };
}

function renderList(scopeType: "session" | "project") {
  return render(
    <DocumentList
      scopeId="s1"
      scopeType={scopeType}
      buildTaskId={null}
      onBuildTaskIdChange={() => {}}
    />
  );
}

describe("DocumentList in-context badge", () => {
  beforeEach(() => {
    mockMutateAsync.mockReset();
  });

  it("shows an In context badge with approximate token cost", () => {
    mockUseDocuments.mockReturnValue({
      data: {
        documents: [
          makeDoc({ in_context: true, context_tokens: 12345 }),
          makeDoc({ doc_id: "doc_def", filename: "other.pdf" }),
        ],
        has_index: true,
        unindexed_count: 0,
      },
      isLoading: false,
    });

    renderList("session");

    expect(screen.getByText("In context")).toBeInTheDocument();
    expect(screen.getByText(/~12\.3k tok/)).toBeInTheDocument();
  });

  it("does not show the badge for indexed-only documents", () => {
    mockUseDocuments.mockReturnValue({
      data: {
        documents: [makeDoc({ in_context: false })],
        has_index: true,
        unindexed_count: 0,
      },
      isLoading: false,
    });

    renderList("session");

    expect(screen.queryByText("In context")).not.toBeInTheDocument();
  });

  it("toggles the attachment for session documents", async () => {
    const user = userEvent.setup();
    mockUseDocuments.mockReturnValue({
      data: {
        documents: [makeDoc({ in_context: true, context_tokens: 100 })],
        has_index: true,
        unindexed_count: 0,
      },
      isLoading: false,
    });

    renderList("session");

    await user.click(screen.getByTitle("Remove from chat context (keeps the document)"));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      scopeId: "s1",
      scopeType: "session",
      docId: "doc_abc",
      inContext: false,
    });
  });

  it("hides the attach action for project documents", () => {
    mockUseDocuments.mockReturnValue({
      data: {
        documents: [makeDoc({})],
        has_index: true,
        unindexed_count: 0,
      },
      isLoading: false,
    });

    renderList("project");

    expect(
      screen.queryByTitle("Attach directly to chat context (no index)")
    ).not.toBeInTheDocument();
  });
});
