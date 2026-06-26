import { useState } from "react";
import {
  MoreHorizontal,
  Download,
  Hammer,
  RefreshCw,
  Info,
  Trash2,
  Pencil,
  X,
  CheckCircle,
  FolderOpen,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { useDeleteSource, useDeleteIndex } from "@/hooks";
import { toast } from "sonner";
import type { ModuleOverview } from "@/api/types";
import { FetchDialog } from "./FetchDialog";
import { BuildDialog } from "./BuildDialog";
import { EditSourceDialog } from "./EditSourceDialog";
import { IndexMetadataView } from "./IndexMetadataView";
import { DocumentBrowserDialog } from "./DocumentBrowserDialog";

interface ModuleCardProps {
  module: ModuleOverview;
  embeddingModelId: string;
}

export function ModuleCard({ module, embeddingModelId }: ModuleCardProps) {
  const [showFetch, setShowFetch] = useState(false);
  const [showBuild, setShowBuild] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  const [showDeleteIndex, setShowDeleteIndex] = useState(false);
  const [showDeleteSource, setShowDeleteSource] = useState(false);
  const [showDocBrowser, setShowDocBrowser] = useState(false);

  const deleteSource = useDeleteSource();
  const deleteIdx = useDeleteIndex();

  const handleDeleteIndex = () => {
    deleteIdx.mutate(
      { embeddingModelId, moduleName: module.name },
      {
        onSuccess: () => {
          toast.success(`Index deleted: ${module.name}`);
          setShowDeleteIndex(false);
        },
        onError: (err) => toast.error(err.message),
      }
    );
  };

  const handleDeleteSource = () => {
    deleteSource.mutate(
      { sourceType: module.source_type, name: module.name },
      {
        onSuccess: () => {
          toast.success(`Source removed: ${module.name}`);
          setShowDeleteSource(false);
        },
        onError: (err) => toast.error(err.message),
      }
    );
  };

  const version = module.source_config.version as string | undefined;

  return (
    <>
      <div
        className={
          "bg-muted/50 flex items-center justify-between rounded-lg border p-3" +
          (module.has_index ? " cursor-pointer" : "")
        }
        onDoubleClick={() => module.has_index && setShowDocBrowser(true)}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{module.display_name}</span>
            {version && (
              <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                v{version}
              </Badge>
            )}
          </div>
          <div className="mt-1 flex items-center gap-2">
            {module.in_sources_config && (
              <span className="flex items-center gap-1 text-xs">
                {module.has_fetched_docs ? (
                  <>
                    <FolderOpen className="h-3 w-3 text-green-500" />
                    <span className="text-muted-foreground">
                      {module.doc_dir_size_mb != null
                        ? `${module.doc_dir_size_mb} MB`
                        : "Docs fetched"}
                    </span>
                  </>
                ) : (
                  <>
                    <FolderOpen className="text-muted-foreground h-3 w-3" />
                    <span className="text-muted-foreground">Not fetched</span>
                  </>
                )}
              </span>
            )}
            <span className="flex items-center gap-1 text-xs">
              {module.has_index ? (
                <>
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span className="text-muted-foreground">
                    {module.index_size_mb != null
                      ? `${module.index_size_mb} MB`
                      : "Indexed"}
                  </span>
                </>
              ) : (
                <>
                  <CheckCircle className="text-muted-foreground h-3 w-3" />
                  <span className="text-muted-foreground">Not indexed</span>
                </>
              )}
            </span>
          </div>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {module.in_sources_config && (
              <DropdownMenuItem onClick={() => setShowFetch(true)}>
                <Download className="mr-2 h-3.5 w-3.5" />
                Fetch Docs
              </DropdownMenuItem>
            )}
            {module.has_fetched_docs && !module.has_index && (
              <DropdownMenuItem onClick={() => setShowBuild(true)}>
                <Hammer className="mr-2 h-3.5 w-3.5" />
                Build Index
              </DropdownMenuItem>
            )}
            {module.has_index && (
              <DropdownMenuItem onClick={() => setShowBuild(true)}>
                <RefreshCw className="mr-2 h-3.5 w-3.5" />
                Rebuild Index
              </DropdownMenuItem>
            )}
            {module.has_index && (
              <DropdownMenuItem onClick={() => setShowDocBrowser(true)}>
                <FileText className="mr-2 h-3.5 w-3.5" />
                Browse Documents
              </DropdownMenuItem>
            )}
            {module.has_index && (
              <DropdownMenuItem onClick={() => setShowMetadata(true)}>
                <Info className="mr-2 h-3.5 w-3.5" />
                View Metadata
              </DropdownMenuItem>
            )}
            {module.in_sources_config && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => setShowEdit(true)}>
                  <Pencil className="mr-2 h-3.5 w-3.5" />
                  Edit Source
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setShowDeleteSource(true)}
                  className="text-red-600"
                >
                  <X className="mr-2 h-3.5 w-3.5" />
                  Remove Source
                </DropdownMenuItem>
              </>
            )}
            {module.has_index && (
              <DropdownMenuItem
                onClick={() => setShowDeleteIndex(true)}
                className="text-red-600"
              >
                <Trash2 className="mr-2 h-3.5 w-3.5" />
                Delete Index
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <FetchDialog
        open={showFetch}
        onOpenChange={setShowFetch}
        sourceType={module.source_type}
        name={module.name}
      />
      <BuildDialog
        open={showBuild}
        onOpenChange={setShowBuild}
        moduleName={module.name}
      />
      <EditSourceDialog
        open={showEdit}
        onOpenChange={setShowEdit}
        sourceType={module.source_type}
        name={module.name}
        config={module.source_config}
      />
      <DocumentBrowserDialog
        open={showDocBrowser}
        onOpenChange={setShowDocBrowser}
        embeddingModelId={embeddingModelId}
        moduleName={module.name}
        moduleDisplayName={module.display_name}
      />
      <IndexMetadataView
        open={showMetadata}
        onOpenChange={setShowMetadata}
        moduleName={module.name}
        metadata={module.index_metadata}
      />

      <AlertDialog open={showDeleteIndex} onOpenChange={setShowDeleteIndex}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Index</AlertDialogTitle>
            <AlertDialogDescription>
              Delete the built vector index for &quot;{module.name}&quot;? The source docs
              will be preserved. You can rebuild later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteIndex}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={showDeleteSource} onOpenChange={setShowDeleteSource}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Source</AlertDialogTitle>
            <AlertDialogDescription>
              Remove &quot;{module.name}&quot; from sources.json? This does not delete
              fetched docs or built indexes.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteSource}
              className="bg-red-600 hover:bg-red-700"
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
