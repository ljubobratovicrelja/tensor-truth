import { useEffect, useRef } from "react";
import { useParams, useNavigate, useLocation, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useSessionStore, useChatStore, useUIStore } from "@/stores";
import {
  useSessionMessages,
  useSession,
  useUpdateSession,
  useWebSocketChat,
  useIsMobile,
} from "@/hooks";
import { cn } from "@/lib/utils";
import { PdfDialog } from "@/components/pdfs";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

export function ChatContainer() {
  const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { setActiveSessionId } = useSessionStore();
  const {
    streamingContent,
    streamingThinking,
    streamingSources,
    streamingMetrics,
    isStreaming,
    pipelineStatus,
    error,
    pendingUserMessage,
  } = useChatStore();
  const autoSendTriggered = useRef(false);
  const isMobile = useIsMobile();
  const setHeaderHidden = useUIStore((state) => state.setHeaderHidden);
  const inputHidden = useUIStore((state) => state.inputHidden);

  // Hide header initially when entering chat on mobile
  // The scroll logic in MessageList will show it if we're at top
  useEffect(() => {
    if (isMobile && urlSessionId) {
      setHeaderHidden(true);
    }
  }, [isMobile, urlSessionId, setHeaderHidden]);

  // Sync URL param with store (for other components that need it)
  useEffect(() => {
    setActiveSessionId(urlSessionId ?? null);
  }, [urlSessionId, setActiveSessionId]);

  // Reset autoSend trigger when session changes
  useEffect(() => {
    autoSendTriggered.current = false;
  }, [urlSessionId]);

  // Use urlSessionId directly for queries (source of truth is the URL)
  const { data: sessionData, error: sessionError } = useSession(urlSessionId ?? null);
  const { data: messagesData, isLoading: messagesLoading } = useSessionMessages(
    urlSessionId ?? null
  );
  const updateSession = useUpdateSession();

  // Redirect to home if session doesn't exist
  useEffect(() => {
    if (sessionError && urlSessionId) {
      toast.error("Session not found");
      navigate("/", { replace: true });
    }
  }, [sessionError, urlSessionId, navigate]);

  // Scroll to hash anchor after messages load
  useEffect(() => {
    if (!messagesLoading && location.hash) {
      const id = location.hash.slice(1);
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [messagesLoading, location.hash]);

  const { sendMessage, cancelStreaming } = useWebSocketChat({
    sessionId: urlSessionId ?? null,
    onError: (err) => toast.error(err),
  });

  // Keep sendMessage in a ref to avoid effect re-runs when it changes
  const sendMessageRef = useRef(sendMessage);
  useEffect(() => {
    sendMessageRef.current = sendMessage;
  }, [sendMessage]);

  // Auto-send pending message when coming from welcome page
  // Wait for session data to confirm session exists before sending
  useEffect(() => {
    const shouldAutoSend = searchParams.get("autoSend") === "true";
    if (
      shouldAutoSend &&
      pendingUserMessage &&
      urlSessionId &&
      sessionData && // Wait for session to be loaded
      !autoSendTriggered.current
    ) {
      autoSendTriggered.current = true;
      // Use ref to get the latest sendMessage function
      sendMessageRef.current(pendingUserMessage);
      // Clear the autoSend param from URL
      setSearchParams({}, { replace: true });
    }
  }, [searchParams, pendingUserMessage, urlSessionId, sessionData, setSearchParams]);

  if (!urlSessionId) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center">
        <p>Select or create a chat session to get started</p>
      </div>
    );
  }

  const handleSend = (message: string) => {
    sendMessage(message);
  };

  const handleModulesChange = (modules: string[]) => {
    if (!urlSessionId) return;
    updateSession.mutate(
      { sessionId: urlSessionId, data: { modules } },
      {
        onError: () => toast.error("Failed to update modules"),
      }
    );
  };

  const handleModelChange = (model: string | null) => {
    if (!urlSessionId) return;
    const currentParams = sessionData?.params ?? {};
    const newParams = model
      ? { ...currentParams, model }
      : Object.fromEntries(Object.entries(currentParams).filter(([k]) => k !== "model"));
    updateSession.mutate(
      { sessionId: urlSessionId, data: { params: newParams } },
      {
        onError: () => toast.error("Failed to update model"),
      }
    );
  };

  const currentModules = sessionData?.modules ?? [];
  const currentModel = (sessionData?.params?.model as string) || undefined;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <h2 className="text-muted-foreground text-sm font-medium">
          {sessionData?.title ?? "Chat"}
        </h2>
        <PdfDialog sessionId={urlSessionId} />
      </div>
      <MessageList
        messages={messagesData?.messages ?? []}
        isLoading={messagesLoading}
        pendingUserMessage={pendingUserMessage}
        streamingContent={streamingContent || undefined}
        streamingThinking={streamingThinking || undefined}
        streamingSources={streamingSources.length > 0 ? streamingSources : undefined}
        streamingMetrics={streamingMetrics}
        isStreaming={isStreaming}
        pipelineStatus={pipelineStatus}
      />
      {error && (
        <div className="border-destructive bg-destructive/10 text-destructive border-t py-2 text-sm">
          <div className="chat-content-width">{error}</div>
        </div>
      )}
      <div
        className={cn(
          "bg-background border-t pt-4 pb-[calc(1rem+env(safe-area-inset-bottom))]",
          "max-h-[320px] transition-all duration-300 ease-in-out",
          isMobile && inputHidden && "max-h-0 overflow-hidden border-t-0 !p-0 opacity-0"
        )}
      >
        <div className="chat-content-width md:pr-[5%]">
          <ChatInput
            onSend={handleSend}
            onStop={cancelStreaming}
            isStreaming={isStreaming}
            placeholder={isStreaming ? "Generating response..." : undefined}
            selectedModules={currentModules}
            onModulesChange={handleModulesChange}
            selectedModel={currentModel}
            onModelChange={handleModelChange}
            sessionId={urlSessionId}
            sessionParams={sessionData?.params ?? {}}
          />
        </div>
      </div>
    </div>
  );
}
