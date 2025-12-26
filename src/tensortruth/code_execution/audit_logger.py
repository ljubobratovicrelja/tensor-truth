"""Audit logging for code execution security tracking."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .container_manager import ExecutionResult

logger = logging.getLogger(__name__)


class AuditLogger:
    """Log all code executions for security audit trail.

    Maintains a JSONL (JSON Lines) file with all executed code and results.
    This allows post-hoc security analysis and debugging.

    Example:
        audit_logger = AuditLogger()
        audit_logger.log(
            session_id="abc123",
            code="print('hello')",
            result=execution_result
        )
    """

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize audit logger.

        Args:
            log_file: Path to JSONL log file
                (defaults to ~/.tensortruth/execution_audit.jsonl)
        """
        if log_file is None:
            log_file = Path.home() / ".tensortruth" / "execution_audit.jsonl"

        self.log_file = log_file

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()

        logger.info(f"Audit logger initialized: {self.log_file}")

    def log(
        self,
        session_id: str,
        code: str,
        result: ExecutionResult,
        warnings: Optional[list] = None,
    ):
        """Log a code execution to the audit trail.

        Args:
            session_id: Chat session ID
            code: Python code that was executed
            result: ExecutionResult from execution
            warnings: Optional list of validation warnings
        """
        # Compute hash for deduplication and reference
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "code_hash": code_hash,
            "code": code,
            "success": result.success,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr),
            "error_message": result.error_message,
            "warnings": warnings or [],
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    def get_recent_executions(self, limit: int = 100) -> list:
        """Get recent executions from audit log.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of execution entries (most recent first)
        """
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse JSONL and return most recent entries
            entries = []
            for line in reversed(lines[-limit:]):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            return entries

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []

    def get_session_executions(self, session_id: str) -> list:
        """Get all executions for a specific session.

        Args:
            session_id: Chat session ID

        Returns:
            List of execution entries for the session
        """
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                entries = []
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("session_id") == session_id:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            return entries

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []
