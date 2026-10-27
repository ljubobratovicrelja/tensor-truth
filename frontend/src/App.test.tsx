import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

const mocks = vi.hoisted(() => ({
  triggerModelPreload: vi.fn(),
}));

vi.mock("@/api/system", () => ({
  getModelPreloadStatus: vi.fn(),
  triggerModelPreload: mocks.triggerModelPreload,
  // The panel/other hooks are not mounted in the startup view.
  getMemoryInfo: vi.fn(),
  getDevices: vi.fn(),
  getOllamaStatus: vi.fn(),
  getLlamaCppStatus: vi.fn(),
  getRAGStatus: vi.fn(),
  restartEngine: vi.fn(),
}));

// Keep the startup view inert so only the preload trigger mounts.
vi.mock("@/components/startup", () => ({
  StartupInitializer: () => null,
}));
vi.mock("@/components/chat/ChatContainer", () => ({
  ChatContainer: () => null,
}));
vi.mock("@/components/welcome", () => ({
  WelcomePage: () => null,
}));
vi.mock("@/components/projects", () => ({
  ProjectsListPage: () => null,
  ProjectsNewPage: () => null,
  ProjectViewPage: () => null,
}));
vi.mock("@/components/layout", () => ({
  GlobalLayout: () => null,
  ProjectLayout: () => null,
}));
vi.mock("@/components/ui/sonner", () => ({
  Toaster: () => null,
}));

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <App />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("App model preload", () => {
  beforeEach(() => {
    mocks.triggerModelPreload.mockReset();
    mocks.triggerModelPreload.mockResolvedValue({
      embedder: { kind: "embedder", state: "loading", elapsed_seconds: 0 },
      reranker: { kind: "reranker", state: "loading", elapsed_seconds: 0 },
      any_loading: true,
      all_ready: false,
    });
  });

  it("triggers exactly one model preload when the app mounts", async () => {
    renderApp();

    await waitFor(() => {
      expect(mocks.triggerModelPreload).toHaveBeenCalledTimes(1);
    });

    // Give effects another tick: no duplicate trigger.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mocks.triggerModelPreload).toHaveBeenCalledTimes(1);
  });
});
