import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2 } from "lucide-react";
import { useAddSource } from "@/hooks";
import { toast } from "sonner";

interface AddSourceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddSourceDialog({ open, onOpenChange }: AddSourceDialogProps) {
  const [tab, setTab] = useState("library");
  const addSource = useAddSource();

  // Library fields
  const [libName, setLibName] = useState("");
  const [libType, setLibType] = useState("sphinx");
  const [libVersion, setLibVersion] = useState("");
  const [libBaseUrl, setLibBaseUrl] = useState("");
  const [libStartUrl, setLibStartUrl] = useState("");

  // Paper category fields
  const [paperName, setPaperName] = useState("");
  const [paperDisplayName, setPaperDisplayName] = useState("");
  const [paperDescription, setPaperDescription] = useState("");
  const [paperArxivIds, setPaperArxivIds] = useState("");

  // Book fields
  const [bookName, setBookName] = useState("");
  const [bookTitle, setBookTitle] = useState("");
  const [bookAuthors, setBookAuthors] = useState("");
  const [bookUrl, setBookUrl] = useState("");
  const [bookSplitMethod, setBookSplitMethod] = useState("toc");

  /* eslint-disable react-hooks/set-state-in-effect -- reset form on open */
  useEffect(() => {
    if (open) {
      setLibName("");
      setLibType("sphinx");
      setLibVersion("");
      setLibBaseUrl("");
      setLibStartUrl("");
      setPaperName("");
      setPaperDisplayName("");
      setPaperDescription("");
      setPaperArxivIds("");
      setBookName("");
      setBookTitle("");
      setBookAuthors("");
      setBookUrl("");
      setBookSplitMethod("toc");
    }
  }, [open]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const handleSubmit = () => {
    let sourceType: string;
    let name: string;
    let config: Record<string, unknown>;

    if (tab === "library") {
      if (!libName || !libBaseUrl) {
        toast.error("Name and base URL are required");
        return;
      }
      sourceType = "libraries";
      name = libName.toLowerCase().replace(/[^a-z0-9_]/g, "_");
      config = {
        type: libType,
        version: libVersion || undefined,
        base_url: libBaseUrl,
        start_url: libStartUrl || libBaseUrl,
      };
    } else if (tab === "papers") {
      if (!paperName) {
        toast.error("Category name is required");
        return;
      }
      sourceType = "papers";
      name = paperName.toLowerCase().replace(/[^a-z0-9_]/g, "_");
      const items = paperArxivIds
        .split(/[\n,]+/)
        .map((s) => s.trim())
        .filter(Boolean)
        .map((id) => ({ arxiv_id: id }));
      config = {
        type: "arxiv",
        display_name: paperDisplayName || name,
        description: paperDescription || undefined,
        items,
      };
    } else {
      if (!bookName || !bookUrl) {
        toast.error("Name and URL are required");
        return;
      }
      sourceType = "papers"; // books are stored under papers in sources.json with type pdf_book
      name = bookName.toLowerCase().replace(/[^a-z0-9_]/g, "_");
      config = {
        type: "pdf_book",
        title: bookTitle || name,
        authors: bookAuthors
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        url: bookUrl,
        split_method: bookSplitMethod,
      };
    }

    addSource.mutate(
      { source_type: sourceType, name, config },
      {
        onSuccess: () => {
          toast.success(`Source "${name}" added`);
          onOpenChange(false);
        },
        onError: (err) => {
          toast.error(err.message);
        },
      }
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Add Source</DialogTitle>
        </DialogHeader>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="w-full">
            <TabsTrigger value="library" className="flex-1">
              Library
            </TabsTrigger>
            <TabsTrigger value="papers" className="flex-1">
              Paper Category
            </TabsTrigger>
            <TabsTrigger value="book" className="flex-1">
              Book
            </TabsTrigger>
          </TabsList>

          <TabsContent value="library" className="space-y-3 pt-3">
            <div className="space-y-1.5">
              <Label htmlFor="lib-name">Name</Label>
              <Input
                id="lib-name"
                placeholder="e.g. pytorch"
                value={libName}
                onChange={(e) => setLibName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="lib-type">Doc Type</Label>
              <Select value={libType} onValueChange={setLibType}>
                <SelectTrigger id="lib-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sphinx">Sphinx</SelectItem>
                  <SelectItem value="doxygen">Doxygen</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="lib-version">Version</Label>
              <Input
                id="lib-version"
                placeholder="e.g. 2.5"
                value={libVersion}
                onChange={(e) => setLibVersion(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="lib-base-url">Base URL</Label>
              <Input
                id="lib-base-url"
                placeholder="https://docs.example.com/"
                value={libBaseUrl}
                onChange={(e) => setLibBaseUrl(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="lib-start-url">Start URL (optional)</Label>
              <Input
                id="lib-start-url"
                placeholder="Leave blank to use base URL"
                value={libStartUrl}
                onChange={(e) => setLibStartUrl(e.target.value)}
              />
            </div>
          </TabsContent>

          <TabsContent value="papers" className="space-y-3 pt-3">
            <div className="space-y-1.5">
              <Label htmlFor="paper-name">Category Name</Label>
              <Input
                id="paper-name"
                placeholder="e.g. attention_mechanisms"
                value={paperName}
                onChange={(e) => setPaperName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="paper-display">Display Name</Label>
              <Input
                id="paper-display"
                placeholder="e.g. Attention Mechanisms"
                value={paperDisplayName}
                onChange={(e) => setPaperDisplayName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="paper-desc">Description</Label>
              <Input
                id="paper-desc"
                placeholder="Optional description"
                value={paperDescription}
                onChange={(e) => setPaperDescription(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="paper-ids">
                ArXiv IDs (one per line or comma-separated)
              </Label>
              <textarea
                id="paper-ids"
                className="border-input bg-background placeholder:text-muted-foreground flex min-h-[80px] w-full rounded-md border px-3 py-2 text-sm"
                placeholder={"1706.03762\n1810.04805"}
                value={paperArxivIds}
                onChange={(e) => setPaperArxivIds(e.target.value)}
              />
            </div>
          </TabsContent>

          <TabsContent value="book" className="space-y-3 pt-3">
            <div className="space-y-1.5">
              <Label htmlFor="book-name">Name</Label>
              <Input
                id="book-name"
                placeholder="e.g. dl_foundations"
                value={bookName}
                onChange={(e) => setBookName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="book-title">Title</Label>
              <Input
                id="book-title"
                placeholder="e.g. Deep Learning Foundations"
                value={bookTitle}
                onChange={(e) => setBookTitle(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="book-authors">Authors (comma-separated)</Label>
              <Input
                id="book-authors"
                placeholder="Author One, Author Two"
                value={bookAuthors}
                onChange={(e) => setBookAuthors(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="book-url">PDF URL</Label>
              <Input
                id="book-url"
                placeholder="https://example.com/book.pdf"
                value={bookUrl}
                onChange={(e) => setBookUrl(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="book-split">Split Method</Label>
              <Select value={bookSplitMethod} onValueChange={setBookSplitMethod}>
                <SelectTrigger id="book-split">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="toc">Table of Contents</SelectItem>
                  <SelectItem value="pages">Fixed Pages</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </TabsContent>
        </Tabs>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={addSource.isPending}>
            {addSource.isPending && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
            Add Source
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
