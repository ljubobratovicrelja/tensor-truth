"""Code validation for detecting potentially dangerous operations."""

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import List


class CodeValidator:
    """Validate Python code for dangerous operations.

    This validator uses AST parsing to detect potentially dangerous operations
    like dangerous imports, eval/exec calls, etc. It returns warnings rather
    than blocking execution, since the code runs in an isolated Docker container.

    Example:
        validator = CodeValidator()
        warnings = validator.validate("import os\\nos.system('ls')")
        # Returns: ["Imports potentially dangerous module: os",
        #           "Uses dangerous function: system"]
    """

    # Modules that can interact with the system
    DANGEROUS_IMPORTS = {
        "os",
        "subprocess",
        "sys",
        "shutil",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "multiprocessing",
        "threading",
        "ctypes",
        "__import__",
    }

    # Built-in functions that can execute arbitrary code
    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",  # Can read/write files
        "input",  # Requires user interaction (will block)
    }

    def validate(self, code: str) -> List[str]:
        """Validate code and return list of warning messages.

        Args:
            code: Python code string to validate

        Returns:
            List of warning messages (empty if no issues found)
        """
        warnings = []

        # First run flake8 linting
        flake8_warnings = self._run_flake8(code)
        warnings.extend(flake8_warnings)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e.msg} at line {e.lineno}"]

        # Walk the AST and check for dangerous patterns
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.DANGEROUS_IMPORTS:
                        warnings.append(
                            f"Imports potentially dangerous module: {alias.name}"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in self.DANGEROUS_IMPORTS:
                    warnings.append(
                        f"Imports from potentially dangerous module: {node.module}"
                    )

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in self.DANGEROUS_BUILTINS:
                    warnings.append(f"Uses potentially dangerous function: {func_name}")

                # Check for specific dangerous patterns
                if func_name == "input":
                    warnings.append(
                        "Code requires user input - execution will block/timeout"
                    )

        return warnings

    def _run_flake8(self, code: str) -> List[str]:
        """Run flake8 linting on code and return warnings.

        Args:
            code: Python code string to lint

        Returns:
            List of flake8 warning messages
        """
        warnings = []

        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Run flake8 with project settings
            result = subprocess.run(
                [
                    "flake8",
                    temp_file_path,
                    "--max-line-length=88",
                    "--extend-ignore=E203,W503",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Parse flake8 output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        # Format: filename:line:col: code message
                        # Extract just the code and message
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            warnings.append(f"Style: {parts[3].strip()}")

        except subprocess.TimeoutExpired:
            warnings.append("Flake8 linting timed out")
        except FileNotFoundError:
            # flake8 not installed, skip linting
            pass
        except Exception:
            # Don't let linting errors block execution
            pass
        finally:
            # Clean up temp file
            try:
                Path(temp_file_path).unlink(missing_ok=True)
            except Exception:
                pass

        return warnings

    def _get_function_name(self, node) -> str:
        """Extract function name from AST node.

        Args:
            node: AST node representing a function

        Returns:
            Function name as string, or empty string if not identifiable
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For calls like os.system(), return "system"
            return node.attr
        return ""

    def has_critical_issues(self, code: str) -> bool:
        """Check if code has critical issues that should block execution.

        Args:
            code: Python code string

        Returns:
            True if code has critical issues (syntax errors, input() calls)
        """
        warnings = self.validate(code)

        # Block execution if:
        # 1. Syntax errors
        # 2. Uses input() (will block forever)
        critical_patterns = ["Syntax error", "requires user input"]

        for warning in warnings:
            for pattern in critical_patterns:
                if pattern.lower() in warning.lower():
                    return True

        return False
