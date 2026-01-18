"""Unit tests for tensortruth.utils.pdf module."""

import pytest


@pytest.mark.unit
class TestClearMarkerConverter:
    """Tests for clear_marker_converter function."""

    def test_clears_marker_converter_when_set(self):
        """Test that clear_marker_converter clears the global MARKER_CONVERTER."""
        from tensortruth.utils import pdf

        # Set a mock converter
        mock_converter = object()
        pdf.MARKER_CONVERTER = mock_converter

        # Clear it
        pdf.clear_marker_converter()

        # Should be None
        assert pdf.MARKER_CONVERTER is None

    def test_handles_none_converter(self):
        """Test that clear_marker_converter handles None gracefully."""
        from tensortruth.utils import pdf

        # Ensure converter is None
        pdf.MARKER_CONVERTER = None

        # Should not raise
        pdf.clear_marker_converter()

        assert pdf.MARKER_CONVERTER is None

    def test_handles_del_exception(self):
        """Test that clear_marker_converter handles del exceptions."""
        from tensortruth.utils import pdf

        # Create a mock that raises on del (unlikely but defensive)
        class BadConverter:
            def __del__(self):
                raise RuntimeError("Del failed")

        pdf.MARKER_CONVERTER = BadConverter()

        # Should not raise - exception is caught
        pdf.clear_marker_converter()

        # Should still be set to None in finally block
        assert pdf.MARKER_CONVERTER is None

    def test_clears_real_converter_reference(self):
        """Test clearing releases the reference for garbage collection."""
        import gc
        import weakref

        from tensortruth.utils import pdf

        # Create an object to track
        class TrackedConverter:
            pass

        converter = TrackedConverter()
        weak_ref = weakref.ref(converter)

        pdf.MARKER_CONVERTER = converter
        del converter  # Remove our reference

        # Object should still exist (held by MARKER_CONVERTER)
        assert weak_ref() is not None

        # Clear the converter
        pdf.clear_marker_converter()
        gc.collect()

        # Object should now be garbage collected
        assert weak_ref() is None
