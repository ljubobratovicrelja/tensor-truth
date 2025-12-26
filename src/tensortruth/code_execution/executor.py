"""High-level code execution orchestration."""

import logging
from typing import List

from .audit_logger import AuditLogger
from .container_manager import ExecutionResult, SessionContainerManager
from .parser import CodeBlock
from .validator import CodeValidator

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """High-level orchestration of code execution with validation and logging.

    This class coordinates the entire code execution pipeline:
    1. Validates code for dangerous operations
    2. Executes code in isolated Docker containers
    3. Logs all executions to audit trail

    Example:
        orchestrator = ExecutionOrchestrator()
        blocks = [CodeBlock(language="python", code="print('hello')")]
        results = orchestrator.execute_blocks(
            session_id="abc123",
            code_blocks=blocks,
            timeout=30
        )
        for result in results:
            print(result.stdout)
    """

    def __init__(
        self,
        container_manager: SessionContainerManager = None,
        validator: CodeValidator = None,
        audit_logger: AuditLogger = None,
    ):
        """Initialize execution orchestrator.

        Args:
            container_manager: Container manager instance (creates default if None)
            validator: Code validator instance (creates default if None)
            audit_logger: Audit logger instance (creates default if None)
        """
        self.container_manager = container_manager or SessionContainerManager()
        self.validator = validator or CodeValidator()
        self.audit_logger = audit_logger or AuditLogger()

    def execute_blocks(
        self,
        session_id: str,
        code_blocks: List[CodeBlock],
        timeout: int = 30,
        enabled: bool = True,
    ) -> List[ExecutionResult]:
        """Execute multiple code blocks sequentially.

        Args:
            session_id: Chat session ID
            code_blocks: List of CodeBlock objects to execute
            timeout: Maximum execution time per block in seconds
            enabled: Whether code execution is enabled (if False, returns empty results)

        Returns:
            List of ExecutionResult objects (one per code block)
        """
        if not enabled:
            logger.info("Code execution disabled, skipping")
            return []

        if not code_blocks:
            return []

        logger.info(
            f"Executing {len(code_blocks)} code block(s) for session {session_id}"
        )

        results = []

        for i, block in enumerate(code_blocks):
            logger.info(
                f"Executing block {i+1}/{len(code_blocks)} ({len(block.code)} chars)"
            )

            # Validate code
            warnings = self.validator.validate(block.code)
            if warnings:
                logger.warning(f"Validation warnings: {warnings}")

            # Check for critical issues that should block execution
            if self.validator.has_critical_issues(block.code):
                error_msg = "Code has critical issues: " + "; ".join(warnings)
                logger.error(error_msg)
                result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    execution_time=0.0,
                    error_message=error_msg,
                    exit_code=-1,
                )
                results.append(result)
                self.audit_logger.log(session_id, block.code, result, warnings)
                continue

            # Execute code
            try:
                result = self.container_manager.execute_code(
                    session_id=session_id, code=block.code, timeout=timeout
                )
                results.append(result)

                # Log execution
                self.audit_logger.log(session_id, block.code, result, warnings)

                if result.success:
                    logger.info(
                        f"Block {i+1} executed successfully in "
                        f"{result.execution_time:.2f}s"
                    )
                else:
                    logger.warning(
                        f"Block {i+1} failed: {result.error_message or 'Unknown error'}"
                    )

            except Exception as e:
                logger.error(f"Error executing block {i+1}: {e}")
                result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    execution_time=0.0,
                    error_message=f"Execution error: {str(e)}",
                    exit_code=-1,
                )
                results.append(result)
                self.audit_logger.log(session_id, block.code, result, warnings)

        return results

    def cleanup_session(self, session_id: str):
        """Clean up resources for a session.

        Args:
            session_id: Chat session ID
        """
        logger.info(f"Cleaning up execution resources for session {session_id}")
        self.container_manager.cleanup_session(session_id)

    def is_docker_available(self) -> bool:
        """Check if Docker is available for code execution.

        Returns:
            True if Docker daemon is accessible, False otherwise
        """
        return self.container_manager._check_docker_available()
