import { useState } from "react";
import { Database, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ModulesOverview } from "./ModulesOverview";
import { ActiveTasks } from "./ActiveTasks";
import { AddSourceDialog } from "./AddSourceDialog";

export function IndexManagementPanel() {
  const [showAddSource, setShowAddSource] = useState(false);

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="shrink-0" title="Index Management">
          <Database className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="flex h-[100dvh] w-full flex-col md:h-[85vh] md:w-[80vw] md:max-w-6xl">
        <DialogHeader>
          <div className="flex items-center justify-between pr-6">
            <DialogTitle>Index Management</DialogTitle>
            <Button size="sm" variant="outline" onClick={() => setShowAddSource(true)}>
              <Plus className="mr-1 h-3.5 w-3.5" />
              Add Source
            </Button>
          </div>
        </DialogHeader>

        <Tabs defaultValue="modules" className="flex min-h-0 flex-1 flex-col">
          <TabsList className="w-fit">
            <TabsTrigger value="modules">Modules</TabsTrigger>
            <TabsTrigger value="tasks">Active Tasks</TabsTrigger>
          </TabsList>

          <ScrollArea className="min-h-0 flex-1">
            <TabsContent value="modules" className="mt-0 p-1">
              <ModulesOverview />
            </TabsContent>
            <TabsContent value="tasks" className="mt-0 p-1">
              <ActiveTasks />
            </TabsContent>
          </ScrollArea>
        </Tabs>

        <AddSourceDialog open={showAddSource} onOpenChange={setShowAddSource} />
      </DialogContent>
    </Dialog>
  );
}
