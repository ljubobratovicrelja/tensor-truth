import { X } from "lucide-react";
import type { AttachedImage } from "@/hooks/useWebSocket";

interface ImagePreviewStripProps {
  images: AttachedImage[];
  onRemove: (id: string) => void;
}

export function ImagePreviewStrip({ images, onRemove }: ImagePreviewStripProps) {
  if (images.length === 0) return null;

  return (
    <div className="flex gap-2 overflow-x-auto px-3 py-2">
      {images.map((img) => (
        <div key={img.id} className="relative shrink-0">
          <img
            src={img.previewUrl}
            alt={img.file.name}
            className="h-16 w-16 rounded-lg border object-cover"
          />
          <button
            onClick={() => onRemove(img.id)}
            className="bg-background/80 hover:bg-destructive hover:text-destructive-foreground absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full border text-xs shadow-sm"
            title="Remove image"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ))}
    </div>
  );
}
