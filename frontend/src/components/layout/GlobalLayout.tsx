import { Outlet } from "react-router-dom";
import { AppLayout } from "./AppLayout";
import { SidebarNav } from "./SidebarNav";
import { SessionList } from "@/components/sessions/SessionList";

export function GlobalLayout() {
  return (
    <AppLayout
      sidebar={
        <>
          <SidebarNav />
          <SessionList />
        </>
      }
    >
      <Outlet />
    </AppLayout>
  );
}
