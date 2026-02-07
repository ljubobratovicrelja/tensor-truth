"""Error types for the user extension system."""

from pathlib import Path
from typing import Optional


class ExtensionLoadError:
    """Records a non-fatal error encountered while loading an extension.

    Attributes:
        file_path: Path to the extension file that failed.
        error: The exception or error message.
        ext_type: Type of extension ("command" or "agent").
    """

    def __init__(self, file_path: Path, error: Exception, ext_type: str = ""):
        self.file_path = file_path
        self.error = error
        self.ext_type = ext_type

    def __repr__(self) -> str:
        return (
            f"ExtensionLoadError({self.file_path.name}, "
            f"{self.ext_type}, {self.error})"
        )


class TemplateResolutionError(Exception):
    """Raised when a {{variable}} in a YAML command cannot be resolved."""

    def __init__(self, variable: str, message: Optional[str] = None):
        self.variable = variable
        super().__init__(
            message or f"Template variable not resolved: {{{{{variable}}}}}"
        )
