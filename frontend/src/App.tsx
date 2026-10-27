import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { GlobalLayout, ProjectLayout } from "@/components/layout";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { WelcomePage } from "@/components/welcome";
import {
  ProjectsListPage,
  ProjectsNewPage,
  ProjectViewPage,
} from "@/components/projects";
import { StartupInitializer } from "@/components/startup";
import { ModelPreloadTrigger } from "@/components/startup/ModelPreloadTrigger";
import { Toaster } from "@/components/ui/sonner";

function App() {
  const [initComplete, setInitComplete] = useState(false);

  // The preload trigger stays mounted across the startup → app transition so
  // exactly one idempotent preload request is issued per app load.
  return (
    <>
      <ModelPreloadTrigger />
      {!initComplete ? (
        <StartupInitializer onComplete={() => setInitComplete(true)} />
      ) : (
        <Routes>
          {/* Global context: chats sidebar */}
          <Route element={<GlobalLayout />}>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/chat/:sessionId" element={<ChatContainer />} />
            <Route path="/projects" element={<ProjectsListPage />} />
            <Route path="/projects/new" element={<ProjectsNewPage />} />
          </Route>

          {/* Project context: project-scoped sidebar */}
          <Route path="/projects/:projectId" element={<ProjectLayout />}>
            <Route index element={<ProjectViewPage />} />
            <Route path="chat/:sessionId" element={<ChatContainer />} />
          </Route>
        </Routes>
      )}
      <Toaster />
    </>
  );
}

export default App;
