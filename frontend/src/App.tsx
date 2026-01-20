import { Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/layout";
import { SessionList } from "@/components/sessions/SessionList";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <>
      <AppLayout sidebar={<SessionList />}>
        <Routes>
          <Route path="/" element={<ChatContainer />} />
          <Route path="/chat/:sessionId" element={<ChatContainer />} />
        </Routes>
      </AppLayout>
      <Toaster />
    </>
  );
}

export default App;
