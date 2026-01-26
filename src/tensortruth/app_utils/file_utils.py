"""File utility functions for TensorTruth."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def atomic_write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON data to a file atomically.

    Uses write to temp file + fsync + os.replace() for crash safety.
    The file is either fully written or not modified at all.

    Args:
        path: Target file path.
        data: Dictionary to serialize as JSON.
        indent: JSON indentation level.

    Raises:
        OSError: If writing fails.
        TypeError: If data is not JSON serializable.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (required for atomic replace)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".json",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace
        os.replace(temp_path, path)

    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
