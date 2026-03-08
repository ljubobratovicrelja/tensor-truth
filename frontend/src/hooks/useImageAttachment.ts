import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import type { AttachedImage } from "./useWebSocket";

const MAX_IMAGES = 4;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp"];

export function useImageAttachment() {
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const detachedRef = useRef(false);

  const addImageFiles = useCallback(
    (files: File[]) => {
      const imageFiles: File[] = [];
      for (const f of files) {
        if (!ALLOWED_IMAGE_TYPES.includes(f.type)) {
          toast.error("Unsupported image format");
        } else if (f.size > MAX_IMAGE_SIZE) {
          toast.error("Image exceeds 10MB limit");
        } else {
          imageFiles.push(f);
        }
      }
      const remaining = MAX_IMAGES - attachedImages.length;
      const toAdd = imageFiles.slice(0, remaining);

      const newImages: AttachedImage[] = toAdd.map((file) => ({
        id: Math.random().toString(36).slice(2) + Date.now().toString(36),
        file,
        previewUrl: URL.createObjectURL(file),
        mimetype: file.type,
      }));
      setAttachedImages((prev) => [...prev, ...newImages]);
    },
    [attachedImages.length]
  );

  const removeImage = useCallback((id: string) => {
    setAttachedImages((prev) => {
      const img = prev.find((i) => i.id === id);
      if (img) URL.revokeObjectURL(img.previewUrl);
      return prev.filter((i) => i.id !== id);
    });
  }, []);

  const clearImages = useCallback(() => {
    setAttachedImages((prev) => {
      prev.forEach((img) => URL.revokeObjectURL(img.previewUrl));
      return [];
    });
  }, []);

  // Detach images: returns current images and resets state WITHOUT revoking URLs.
  // Use before navigation so the receiving component can still use the blob URLs.
  const detach = useCallback(() => {
    detachedRef.current = true;
    const current = attachedImages;
    setAttachedImages([]);
    return current;
  }, [attachedImages]);

  // Cleanup preview URLs on unmount (skip if detached)
  useEffect(() => {
    return () => {
      if (!detachedRef.current) {
        attachedImages.forEach((img) => URL.revokeObjectURL(img.previewUrl));
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drag handlers for the container element
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      addImageFiles(files);
    },
    [addImageFiles]
  );

  // Paste handler for textarea
  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      const imageFiles: File[] = [];
      // Use index-based access — DataTransferItemList may not be iterable
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === "file" && item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) imageFiles.push(file);
        }
      }
      if (imageFiles.length > 0) {
        addImageFiles(imageFiles);
      }
    },
    [addImageFiles]
  );

  // Props to spread on the container div
  const dragProps = {
    onDragOver: handleDragOver,
    onDragEnter: handleDragOver,
    onDragLeave: handleDragLeave,
    onDrop: handleDrop,
  };

  return {
    attachedImages,
    isDragOver,
    dragProps,
    handlePaste,
    removeImage,
    clearImages,
    detach,
    hasImages: attachedImages.length > 0,
  };
}
