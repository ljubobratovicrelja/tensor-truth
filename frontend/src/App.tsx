import { AppLayout } from "@/components/layout";
import { SessionList } from "@/components/sessions/SessionList";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <>
      <AppLayout sidebar={<SessionList />}>
        <ChatContainer />
      </AppLayout>
      <Toaster />
    </>
  );
}

export default App;
