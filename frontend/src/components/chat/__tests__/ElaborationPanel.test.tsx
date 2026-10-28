import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ElaborationPanel } from "../ElaborationPanel";
import { useElaborationStore } from "@/stores/elaborationStore";

const anchor = {
  messageIndex: 1,
  selectedText: "the passage",
  occurrenceIndex: 0,
};

function renderPanel(onFollowUp = vi.fn(), onCancel = vi.fn()) {
  render(<ElaborationPanel onFollowUp={onFollowUp} onCancel={onCancel} />);
  return { onFollowUp, onCancel };
}

function composer() {
  return screen.getByLabelText("Ask a follow-up") as HTMLTextAreaElement;
}

describe("ElaborationPanel composer", () => {
  beforeEach(() => {
    useElaborationStore.getState().reset();
  });

  it("disables the composer until the explanation is persisted", () => {
    // Still streaming and no id yet: the panel cannot attach a follow-up.
    useElaborationStore.getState().openFor(anchor);
    renderPanel();

    expect(composer()).toBeDisabled();
    expect(screen.queryByLabelText("Send follow-up")).not.toBeInTheDocument();

    act(() => {
      useElaborationStore.setState({ isStreaming: false });
    });
    expect(composer()).toBeDisabled();
    expect(screen.getByLabelText("Send follow-up")).toBeDisabled();
  });

  it("sends on Enter and clears the draft", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    const { onFollowUp } = renderPanel();
    const user = userEvent.setup();

    await user.type(composer(), "why?");
    await user.keyboard("{Enter}");

    expect(onFollowUp).toHaveBeenCalledWith("why?");
    expect(composer()).toHaveValue("");
  });

  it("inserts a newline on Shift+Enter instead of sending", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    const { onFollowUp } = renderPanel();
    const user = userEvent.setup();

    await user.type(composer(), "why?");
    await user.keyboard("{Shift>}{Enter}{/Shift}");

    expect(onFollowUp).not.toHaveBeenCalled();
    expect(composer()).toHaveValue("why?\n");
  });

  it("disables input and shows Stop while a reply streams", () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.");
    useElaborationStore.getState().startFollowUp("why?");
    const { onCancel } = renderPanel();

    expect(composer()).toBeDisabled();
    const stop = screen.getByLabelText("Stop");
    stop.click();
    expect(onCancel).toHaveBeenCalled();
  });

  it("renders the stored thread and the streaming partial reply", async () => {
    useElaborationStore.getState().openStored(anchor, "elab-1", "Explained.", [
      { role: "user", content: "why?" },
      { role: "assistant", content: "Because." },
    ]);
    useElaborationStore.getState().startFollowUp("and then?");
    useElaborationStore.getState().appendReplyToken("Then this.");

    renderPanel();

    expect(screen.getByTestId("elaboration-explanation")).toHaveTextContent("Explained.");
    expect(screen.getAllByTestId("elaboration-turn-user")[0]).toHaveTextContent("why?");
    expect(screen.getByTestId("elaboration-turn-assistant")).toHaveTextContent(
      "Because."
    );
    await waitFor(() =>
      expect(screen.getByTestId("elaboration-reply")).toHaveTextContent("Then this.")
    );
  });
});
