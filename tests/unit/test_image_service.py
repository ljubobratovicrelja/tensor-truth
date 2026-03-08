"""Tests for image storage service."""

import base64

import pytest

from tensortruth.services.image_service import ImageService


@pytest.fixture
def image_service():
    return ImageService()


@pytest.fixture
def sample_png_b64():
    """A minimal 1x1 transparent PNG as base64."""
    # 1x1 transparent PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_bytes).decode()


class TestImageService:
    def test_save_and_retrieve(
        self, tmp_path, monkeypatch, image_service, sample_png_b64
    ):
        """Save an image and retrieve its path."""
        monkeypatch.setattr(
            "tensortruth.services.image_service.get_session_dir",
            lambda sid: tmp_path / sid,
        )

        session_id = "test-session"
        image_id = image_service.save_image(
            session_id, sample_png_b64, "image/png", "test.png"
        )

        assert image_id  # non-empty string
        path = image_service.get_image_path(session_id, image_id)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_get_image_path_not_found(self, tmp_path, monkeypatch, image_service):
        """Return None for nonexistent image."""
        monkeypatch.setattr(
            "tensortruth.services.image_service.get_session_dir",
            lambda sid: tmp_path / sid,
        )
        path = image_service.get_image_path("test-session", "nonexistent")
        assert path is None

    def test_unsupported_mimetype(self, image_service, sample_png_b64):
        """Reject unsupported MIME types."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            image_service.save_image(
                "sess", sample_png_b64, "image/svg+xml", "test.svg"
            )

    def test_get_mimetype(self, tmp_path, image_service):
        """Return correct MIME type for known extensions."""
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b"fake")
        assert image_service.get_mimetype(png_file) == "image/png"

        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"fake")
        assert image_service.get_mimetype(jpg_file) == "image/jpeg"
