import { describe, it, expect, vi, beforeAll } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DocumentUploader } from "./DocumentUploader";

const uploadTextMutate = vi.fn();

vi.mock("@/hooks", () => ({
  useUploadDocument: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadText: () => ({ mutateAsync: uploadTextMutate, isPending: false }),
  useUploadUrl: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadArxiv: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUploadFileUrl: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useArxivLookup: () => ({ data: undefined, isFetching: false, isError: false }),
  useFileUrlInfo: () => ({
    data: undefined,
    isFetching: false,
    isError: false,
  }),
}));

beforeAll(() => {
  // jsdom's File lacks .text(); the uploader reads text files that way.
  Object.defineProperty(File.prototype, "text", {
    configurable: true,
    value: () => Promise.resolve("hello context"),
  });
});

describe("DocumentUploader in-context toggle", () => {
  it("offers the direct-to-context checkbox for session scope", () => {
    render(<DocumentUploader scopeId="s1" scopeType="session" />);
    expect(
      screen.getByLabelText("Add directly to context (no indexing)")
    ).toBeInTheDocument();
    expect(screen.getByText(/large-context models/i)).toBeInTheDocument();
  });

  it("hides the checkbox for project scope", () => {
    render(<DocumentUploader scopeId="p1" scopeType="project" />);
    expect(
      screen.queryByLabelText("Add directly to context (no indexing)")
    ).not.toBeInTheDocument();
  });

  it("sends in_context when the toggle is checked", async () => {
    const user = userEvent.setup();
    uploadTextMutate.mockReset();
    const { container } = render(<DocumentUploader scopeId="s1" scopeType="session" />);

    await user.click(screen.getByLabelText("Add directly to context (no indexing)"));

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["hello context"], "notes.txt", { type: "text/plain" });
    await user.upload(input, file);

    await waitFor(() =>
      expect(uploadTextMutate).toHaveBeenCalledWith({
        scopeId: "s1",
        scopeType: "session",
        data: {
          content: "hello context",
          filename: "notes.txt",
          in_context: true,
        },
      })
    );
  });
});
