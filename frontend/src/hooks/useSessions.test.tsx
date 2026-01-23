import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useSessionMessages } from "./useSessions";

// Mock the sessions API
const mockGetSessionMessages = vi.fn();

vi.mock("@/api/sessions", () => ({
  listSessions: vi.fn(),
  createSession: vi.fn(),
  getSession: vi.fn(),
  updateSession: vi.fn(),
  deleteSession: vi.fn(),
  getSessionMessages: (...args: unknown[]) => mockGetSessionMessages(...args),
  getSessionStats: vi.fn(),
}));

// Test wrapper with QueryClient
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        // Disable global staleTime for testing
        staleTime: 0,
      },
    },
  });
  return {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
    queryClient,
  };
}

describe("useSessionMessages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch messages when sessionId is provided", async () => {
    const mockMessages = {
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
      ],
    };
    mockGetSessionMessages.mockResolvedValue(mockMessages);

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useSessionMessages("session-123"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(mockGetSessionMessages).toHaveBeenCalledWith("session-123");
    expect(result.current.data).toEqual(mockMessages);
  });

  it("should not fetch when sessionId is null", async () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useSessionMessages(null), { wrapper });

    // Wait a tick for React Query to process
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(mockGetSessionMessages).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  it("should refetch when sessionId changes", async () => {
    const mockMessagesA = {
      messages: [{ role: "user", content: "Session A message" }],
    };
    const mockMessagesB = {
      messages: [{ role: "user", content: "Session B message" }],
    };

    mockGetSessionMessages
      .mockResolvedValueOnce(mockMessagesA)
      .mockResolvedValueOnce(mockMessagesB);

    const { wrapper } = createWrapper();
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) => useSessionMessages(sessionId),
      {
        wrapper,
        initialProps: { sessionId: "session-A" },
      }
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockMessagesA);
    expect(mockGetSessionMessages).toHaveBeenCalledWith("session-A");

    // Switch to session B
    rerender({ sessionId: "session-B" });

    await waitFor(() => {
      expect(mockGetSessionMessages).toHaveBeenCalledWith("session-B");
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(mockMessagesB);
    });
  });

  it("should always refetch on mount (refetchOnMount: always)", async () => {
    const mockMessages = {
      messages: [{ role: "user", content: "Hello" }],
    };
    mockGetSessionMessages.mockResolvedValue(mockMessages);

    const { wrapper, queryClient } = createWrapper();

    // Pre-populate the cache
    queryClient.setQueryData(["sessions", "session-123", "messages"], mockMessages);

    // Render the hook - it should refetch despite cache existing
    const { result } = renderHook(() => useSessionMessages("session-123"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should have fetched despite cache being present (refetchOnMount: 'always')
    expect(mockGetSessionMessages).toHaveBeenCalledWith("session-123");
  });

  it("should have staleTime of 0 to prevent showing stale data", async () => {
    const mockMessages = {
      messages: [{ role: "user", content: "Hello" }],
    };
    mockGetSessionMessages.mockResolvedValue(mockMessages);

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useSessionMessages("session-123"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Query should be immediately stale (staleTime: 0)
    expect(result.current.isStale).toBe(true);
  });

  it("should show loading state during fetch", async () => {
    // Create a promise that we can control
    let resolvePromise: (value: unknown) => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    mockGetSessionMessages.mockReturnValue(pendingPromise);

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useSessionMessages("session-123"), {
      wrapper,
    });

    // Should be loading initially
    expect(result.current.isLoading).toBe(true);

    // Resolve the promise
    resolvePromise!({ messages: [] });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
  });

  it("should differentiate between isLoading and isFetching", async () => {
    // This test verifies that useSessionMessages returns both isLoading and isFetching
    // isLoading: true only on initial load (no cached data)
    // isFetching: true whenever a fetch is in progress (including refetches)
    // This distinction is important for showing proper loading states on session switch

    let resolvePromise: (value: unknown) => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockGetSessionMessages.mockReturnValue(pendingPromise);

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useSessionMessages("session-123"), {
      wrapper,
    });

    // On initial load, both should be true
    expect(result.current.isLoading).toBe(true);
    expect(result.current.isFetching).toBe(true);

    // Resolve the promise
    resolvePromise!({ messages: [{ role: "user", content: "Hello" }] });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isFetching).toBe(false);
    });

    // After data is loaded, isLoading should be false but data should be present
    expect(result.current.data).toBeDefined();
  });
});
