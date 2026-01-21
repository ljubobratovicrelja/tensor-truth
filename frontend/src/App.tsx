import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/layout";
import { SessionList } from "@/components/sessions/SessionList";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { WelcomePage } from "@/components/welcome";
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
      <AppLayout sidebar={<SessionList />}>
        <Routes>
          <Route path="/" element={<WelcomePage />} />
          <Route path="/chat/:sessionId" element={<ChatContainer />} />
        </Routes>
      </AppLayout>
      <Toaster />
    </>
  );
}

export default App;
