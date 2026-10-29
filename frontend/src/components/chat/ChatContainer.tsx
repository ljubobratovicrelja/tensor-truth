import { useEffect, useRef } from "react";
import { useParams, useNavigate, useLocation, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useSessionStore, useChatStore, useUIStore, useElaborationStore } from "@/stores";
import {
  useSessionMessages,
  useSession,
  useUpdateSession,
  useWebSocketChat,
  useIsMobile,
  useProject,
  useModels,
  useDocuments,
  useCompactSession,
  useElaboration,
  useElaborationClick,
  useElaborationHighlights,
  useTextSelection,
  paramToThinking,
  thinkingToParam,
} from "@/hooks";
import { cn } from "@/lib/utils";
import { DocumentDialog } from "@/components/documents";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { ElaborationSidebar } from "./ElaborationSidebar";
import { ElaborationTrigger } from "./ElaborationTrigger";
import { decodeModelValue, encodeModelValue } from "./ModelSelectContent";

export function ChatContainer() {
  const { sessionId: urlSessionId, projectId } = useParams<{
    sessionId: string;
    projectId: string;
  }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { setActiveSessionId } = useSessionStore();
  const {
    streamingContent,
    streamingThinking,
    narrationSegments,
    streamingSources,
    streamingMetrics,
    isStreaming,
    pipelineStatus,
    error,
    pendingUserMessage,
    clearPendingUserMessage,
    agentProgress,
    streamingToolSteps,
    confidenceLevel,
    streamingReasoning,
    pendingUserImages,
    pendingAttachedImages,
    lastResponseStats,
    confirmationRequests,
  } = useChatStore();
  const autoSendTriggered = useRef(false);
  const prevSessionIdRef = useRef<string | undefined>(undefined);
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

  // Reset autoSend trigger and clear stale pending message when session changes
  // This prevents showing a pending message from a different session
  useEffect(() => {
    const prevSessionId = prevSessionIdRef.current;
    prevSessionIdRef.current = urlSessionId;

    autoSendTriggered.current = false;

    // Only clear pending message when switching FROM one session TO another
    // Don't clear when coming from welcome page (prevSessionId undefined)
    // Don't clear on initial mount
    if (prevSessionId && prevSessionId !== urlSessionId) {
      clearPendingUserMessage();
    }
  }, [urlSessionId, clearPendingUserMessage]);

  // Use urlSessionId directly for queries (source of truth is the URL)
  const { data: sessionData, error: sessionError } = useSession(urlSessionId ?? null);
  const {
    data: messagesData,
    isLoading: messagesLoading,
    isFetching: messagesFetching,
  } = useSessionMessages(urlSessionId ?? null);

  // --- Selection-driven elaboration (independent of the chat stream) ---
  const elaborationOpen = useElaborationStore((state) => state.open);
  const elaborationAnchor = useElaborationStore((state) => state.anchor);
  const { selection, clear: clearSelection } = useTextSelection();
  const {
    start: startElaboration,
    followUp: followUpElaboration,
    cancel: cancelElaboration,
  } = useElaboration(urlSessionId ?? null);
  useElaborationHighlights(
    messagesData?.messages,
    elaborationOpen ? elaborationAnchor : null
  );
  useElaborationClick(messagesData?.messages);

  // Show loading state during initial load OR when switching sessions (refetching)
  // This prevents showing stale cached messages from a different session
  const showMessagesLoading = messagesLoading || (messagesFetching && !messagesData);
  const updateSession = useUpdateSession();

  // Load project data if session belongs to a project
  const sessionProjectId = sessionData?.project_id ?? null;
  const { data: projectData } = useProject(sessionProjectId);
  const { data: modelsData } = useModels();

  // Load session documents for non-project sessions (shown in ModuleSelector)
  const { data: sessionDocsData } = useDocuments(
    !sessionProjectId ? (urlSessionId ?? null) : null,
    "session"
  );
  const sessionDocuments = sessionDocsData?.has_index
    ? sessionDocsData.documents
        .filter((d) => d.is_indexed !== false)
        .map((d) => ({
          doc_id: d.doc_id,
          type: d.doc_id.startsWith("url_") ? "url" : "pdf",
          filename: d.filename,
          status: "indexed",
        }))
    : undefined;

  const modelThinkingSupport = (modelName: string) => {
    const info = modelsData?.models.find((m) => m.name === modelName);
    const caps = info?.capabilities ?? [];
    return {
      thinking: caps.includes("thinking"),
      levels: (info?.reasoning_levels ?? []).length > 0,
    };
  };

  // Redirect to home (or project view) if session doesn't exist
  useEffect(() => {
    if (sessionError && urlSessionId) {
      toast.error("Session not found");
      const fallback = projectId ? `/projects/${projectId}` : "/";
      navigate(fallback, { replace: true });
    }
  }, [sessionError, urlSessionId, projectId, navigate]);

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

  const compact = useCompactSession(urlSessionId);

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
    const hasPendingContent =
      pendingUserMessage || (pendingAttachedImages && pendingAttachedImages.length > 0);
    if (
      shouldAutoSend &&
      hasPendingContent &&
      urlSessionId &&
      sessionData && // Wait for session to be loaded
      !autoSendTriggered.current
    ) {
      autoSendTriggered.current = true;
      // Use ref to get the latest sendMessage function
      sendMessageRef.current(
        pendingUserMessage || "",
        pendingAttachedImages ?? undefined
      );
      // Clear the autoSend param from URL
      setSearchParams({}, { replace: true });
    }
  }, [
    searchParams,
    pendingUserMessage,
    pendingAttachedImages,
    urlSessionId,
    sessionData,
    setSearchParams,
  ]);

  if (!urlSessionId) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center">
        <p>Select or create a chat session to get started</p>
      </div>
    );
  }

  const handleCompact = () => {
    if (!urlSessionId) return;
    compact.mutate({});
  };

  const handleSend = (
    message: string,
    images?: import("@/hooks/useWebSocket").AttachedImage[]
  ) => {
    sendMessage(message, images);
  };

  const handleStartElaboration = () => {
    if (!selection) return;
    void startElaboration(selection.anchor);
    // The trigger must not linger over the freshly opened panel.
    clearSelection();
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

  const handleModuleModeChange = (mode: "manual" | "auto") => {
    if (!urlSessionId) return;
    // Switching to auto keeps whatever the agent has loaded so far; switching
    // back to manual leaves that set in place, now user-editable.
    updateSession.mutate(
      {
        sessionId: urlSessionId,
        data: {
          module_mode: mode,
          modules: mode === "auto" ? (sessionData?.modules ?? []) : currentModules,
        },
      },
      {
        onError: () => toast.error("Failed to update module mode"),
      }
    );
  };

  const handleModelChange = (encodedValue: string | null) => {
    if (!urlSessionId) return;
    const currentParams = sessionData?.params ?? {};

    let model: string | null = null;
    let providerId: string | undefined;
    if (encodedValue) {
      const decoded = decodeModelValue(encodedValue);
      model = decoded.modelName;
      providerId = decoded.providerId;
    }

    let newParams = model
      ? { ...currentParams, model, provider_id: providerId || "ollama" }
      : Object.fromEntries(
          Object.entries(currentParams).filter(
            ([k]) => k !== "model" && k !== "provider_id"
          )
        );

    // Reset thinking if incompatible with new model
    if (model && newParams.thinking !== undefined) {
      const support = modelThinkingSupport(model);
      // Level support for OpenAI-compatible models is probed lazily, so an
      // empty list here does not mean unsupported — only drop the preference
      // when the model cannot reason at all.
      if (!support.thinking) {
        newParams = Object.fromEntries(
          Object.entries(newParams).filter(([k]) => k !== "thinking")
        );
      }
    }

    updateSession.mutate(
      { sessionId: urlSessionId, data: { params: newParams } },
      {
        onError: () => toast.error("Failed to update model"),
      }
    );
  };

  const handleThinkingChange = (value: string) => {
    if (!urlSessionId) return;
    const currentParams = sessionData?.params ?? {};
    const paramValue = thinkingToParam(value);
    const newParams =
      paramValue === undefined
        ? Object.fromEntries(
            Object.entries(currentParams).filter(([k]) => k !== "thinking")
          )
        : { ...currentParams, thinking: paramValue };
    updateSession.mutate(
      { sessionId: urlSessionId, data: { params: newParams } },
      {
        onError: () => toast.error("Failed to update thinking"),
      }
    );
  };

  const currentModules = sessionData?.modules ?? [];
  const currentModuleMode = sessionData?.module_mode ?? "auto";
  const currentModelName = (sessionData?.params?.model as string) || undefined;
  const currentProviderId = (() => {
    const stored = sessionData?.params?.provider_id as string | undefined;
    if (stored) return stored;
    // Resolve provider from models list for sessions without provider_id
    if (currentModelName && modelsData?.models) {
      const info = modelsData.models.find((m) => m.name === currentModelName);
      if (info?.provider_id) return info.provider_id;
    }
    return "ollama";
  })();
  const currentModel = currentModelName
    ? encodeModelValue(currentProviderId, currentModelName)
    : undefined;
  const currentThinking = paramToThinking(sessionData?.params?.thinking);

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b px-4 py-2">
          <h2 className="text-muted-foreground text-sm font-medium">
            {sessionData?.title ?? "Chat"}
          </h2>
          <DocumentDialog scopeId={urlSessionId} scopeType="session" />
        </div>
        <MessageList
          sessionId={urlSessionId}
          messages={messagesData?.messages ?? []}
          isLoading={showMessagesLoading}
          pendingUserMessage={pendingUserMessage}
          streamingContent={streamingContent || undefined}
          streamingThinking={streamingThinking || undefined}
          narrationSegments={narrationSegments}
          streamingSources={streamingSources.length > 0 ? streamingSources : undefined}
          streamingMetrics={streamingMetrics}
          isStreaming={isStreaming}
          streamingToolSteps={
            streamingToolSteps.length > 0 ? streamingToolSteps : undefined
          }
          pipelineStatus={pipelineStatus}
          agentProgress={agentProgress}
          confidenceLevel={confidenceLevel}
          streamingReasoning={streamingReasoning || undefined}
          pendingUserImages={pendingUserImages}
          lastResponseStats={lastResponseStats}
          confirmationRequests={
            confirmationRequests.length > 0 ? confirmationRequests : undefined
          }
        />
        {error && (
          <div className="border-destructive bg-destructive/10 text-destructive border-t py-2 text-sm">
            <div className="chat-content-width">{error}</div>
          </div>
        )}
        <div
          className={cn(
            "bg-background border-t pt-4 pb-[calc(1rem+env(safe-area-inset-bottom))]",
            "max-h-[320px] overflow-visible transition-all duration-300 ease-in-out",
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
              moduleMode={currentModuleMode}
              onModuleModeChange={handleModuleModeChange}
              selectedModel={currentModel}
              onModelChange={handleModelChange}
              thinking={currentThinking}
              onThinkingChange={handleThinkingChange}
              sessionId={urlSessionId}
              sessionParams={sessionData?.params ?? {}}
              lockedModules={
                projectData ? Object.keys(projectData.catalog_modules) : undefined
              }
              projectDocuments={projectData?.documents}
              sessionDocuments={sessionDocuments}
              isProjectSession={!!sessionProjectId}
              onCompact={handleCompact}
              compactPending={compact.isPending}
            />
          </div>
        </div>
      </div>
      <ElaborationSidebar onCancel={cancelElaboration} onFollowUp={followUpElaboration} />
      <ElaborationTrigger
        rect={isStreaming ? null : (selection?.rect ?? null)}
        onOpen={handleStartElaboration}
      />
    </div>
  );
}
