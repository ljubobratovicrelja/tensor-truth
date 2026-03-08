"""Image storage service for chat image attachments."""

import base64
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Optional

from tensortruth.app_utils.paths import get_session_dir

logger = logging.getLogger(__name__)

# Allowed image MIME types
ALLOWED_MIMETYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)

# MIME type to file extension mapping
MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _get_images_dir(session_id: str) -> Path:
    """Get the images directory for a session, creating it if needed."""
    images_dir = get_session_dir(session_id) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


class ImageService:
    """Service for saving and retrieving chat images."""

    def save_image(
        self, session_id: str, image_b64: str, mimetype: str, filename: str
    ) -> str:
        """Decode and save a base64-encoded image to disk.

        Args:
            session_id: The session this image belongs to.
            image_b64: Base64-encoded image data.
            mimetype: MIME type (e.g. "image/png").
            filename: Original filename from the client.

        Returns:
            The UUID assigned to the saved image.

        Raises:
            ValueError: If the MIME type is not allowed.
        """
        if mimetype not in ALLOWED_MIMETYPES:
            raise ValueError(
                f"Unsupported image type: {mimetype}. "
                f"Allowed: {', '.join(sorted(ALLOWED_MIMETYPES))}"
            )

        image_id = uuid.uuid4().hex
        ext = MIME_TO_EXT.get(mimetype, ".png")
        images_dir = _get_images_dir(session_id)
        image_path = images_dir / f"{image_id}{ext}"

        image_data = base64.b64decode(image_b64)
        image_path.write_bytes(image_data)

        logger.info(
            "Saved image %s (%s, %d bytes) for session %s",
            image_id,
            filename,
            len(image_data),
            session_id,
        )
        return image_id

    def get_image_path(self, session_id: str, image_id: str) -> Optional[Path]:
        """Find the stored image file by ID.

        Args:
            session_id: The session the image belongs to.
            image_id: The UUID of the image.

        Returns:
            Path to the image file, or None if not found.
        """
        images_dir = _get_images_dir(session_id)
        matches = list(images_dir.glob(f"{image_id}.*"))
        if matches:
            return matches[0]
        return None

    def get_mimetype(self, image_path: Path) -> str:
        """Determine the MIME type of an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            MIME type string.
        """
        mime, _ = mimetypes.guess_type(str(image_path))
        return mime or "application/octet-stream"
