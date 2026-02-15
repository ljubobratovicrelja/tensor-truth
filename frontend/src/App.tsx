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
import { Toaster } from "@/components/ui/sonner";

function App() {
  const [initComplete, setInitComplete] = useState(false);

  // Show startup initializer until initialization is complete
  if (!initComplete) {
    return (
      <>
        <StartupInitializer onComplete={() => setInitComplete(true)} />
        <Toaster />
      </>
    );
  }

  // Show main app once initialization is complete
  return (
    <>
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
      <Toaster />
    </>
  );
}

export default App;
