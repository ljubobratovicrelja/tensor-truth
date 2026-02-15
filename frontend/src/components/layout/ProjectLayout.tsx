import { Outlet, useParams } from "react-router-dom";
import { AppLayout } from "./AppLayout";
import { SidebarNav } from "./SidebarNav";
import { ProjectSessionList } from "@/components/sessions/ProjectSessionList";
import { ProjectRightSidebar } from "@/components/projects/ProjectRightSidebar";

export function ProjectLayout() {
  const { projectId } = useParams<{ projectId: string }>();

  return (
    <AppLayout
      sidebar={
        <>
          <SidebarNav />
          <ProjectSessionList projectId={projectId!} />
        </>
      }
      rightSidebar={<ProjectRightSidebar projectId={projectId!} />}
    >
      <Outlet />
    </AppLayout>
  );
}
