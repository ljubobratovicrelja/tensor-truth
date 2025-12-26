"""Docker container lifecycle management for code execution."""

import atexit
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str
    stderr: str
    execution_time: float
    error_message: Optional[str]
    exit_code: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "exit_code": self.exit_code,
        }


class SessionContainerManager:
    """Manages Docker containers for code execution sessions.

    Each chat session gets its own persistent Docker container to maintain
    state (variables, imports) across multiple code executions. Containers
    are automatically cleaned up after 1 hour of inactivity.

    Example:
        manager = SessionContainerManager()
        result = manager.execute_code(
            session_id="abc123",
            code="x = 42\nprint(x)",
            timeout=30
        )
        print(result.stdout)  # "42"
    """

    def __init__(
        self,
        image_name: str = "tensortruth-executor:latest",
        idle_timeout: int = 3600,  # 1 hour
        max_containers: int = 10,
    ):
        """Initialize container manager.

        Args:
            image_name: Docker image to use for execution containers
            idle_timeout: Seconds of inactivity before container cleanup
            max_containers: Maximum number of concurrent containers
        """
        self.image_name = image_name
        self.idle_timeout = idle_timeout
        self.max_containers = max_containers

        self._containers: Dict[str, any] = {}  # session_id → Container object
        self._last_used: Dict[str, float] = {}  # session_id → timestamp
        self._lock = threading.RLock()
        self._gc_thread = None
        self._stop_gc = threading.Event()
        self._docker_available = None  # None = not checked, True/False = available

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

        # Start garbage collection thread
        self._start_gc_thread()

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and accessible.

        Returns:
            True if Docker daemon is accessible, False otherwise
        """
        if self._docker_available is not None:
            return self._docker_available

        try:
            import docker

            client = docker.from_env()
            client.ping()
            self._docker_available = True
            logger.info("Docker daemon is available")
            return True
        except Exception as e:
            self._docker_available = False
            logger.warning(f"Docker not available: {e}")
            return False

    def _ensure_image_exists(self):
        """Ensure the executor Docker image exists, build if necessary."""
        import docker

        client = docker.from_env()

        try:
            client.images.get(self.image_name)
            logger.info(f"Docker image {self.image_name} found")
        except docker.errors.ImageNotFound:
            logger.info(f"Building Docker image {self.image_name}...")
            # Find Dockerfile in code_execution directory
            dockerfile_path = Path(__file__).parent / "executor.Dockerfile"

            if not dockerfile_path.exists():
                raise FileNotFoundError(
                    f"Dockerfile not found at {dockerfile_path}. "
                    "Cannot build executor image."
                )

            # Build image
            client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile="executor.Dockerfile",
                tag=self.image_name,
                rm=True,
            )
            logger.info(f"Successfully built {self.image_name}")

    def get_or_create_container(self, session_id: str):
        """Get existing container for session or create new one.

        Args:
            session_id: Chat session ID

        Returns:
            Docker Container object

        Raises:
            RuntimeError: If Docker is unavailable or container creation fails
        """
        if not self._check_docker_available():
            raise RuntimeError(
                "Docker is not available. Please ensure Docker daemon is running."
            )

        with self._lock:
            # Return existing container if alive
            if session_id in self._containers:
                container = self._containers[session_id]
                try:
                    container.reload()
                    if container.status == "running":
                        self._last_used[session_id] = time.time()
                        return container
                    else:
                        # Container stopped, remove and recreate
                        logger.warning(
                            f"Container for session {session_id} is "
                            f"{container.status}, removing"
                        )
                        self._remove_container(session_id)
                except Exception as e:
                    logger.error(f"Error checking container: {e}")
                    self._remove_container(session_id)

            # Check max containers limit
            if len(self._containers) >= self.max_containers:
                # Remove oldest idle container
                self._cleanup_oldest_container()

            # Create new container
            return self._create_container(session_id)

    def _create_container(self, session_id: str):
        """Create a new container for the session.

        Args:
            session_id: Chat session ID

        Returns:
            Docker Container object
        """
        import docker

        client = docker.from_env()

        # Ensure image exists
        self._ensure_image_exists()

        # Create workspace directory for this session
        workspace_path = (
            Path.home() / ".tensortruth" / "sessions" / session_id / "workspace"
        )
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Container configuration
        container_name = f"tensortruth-session-{session_id[:8]}"

        try:
            container = client.containers.run(
                self.image_name,
                name=container_name,
                detach=True,
                stdin_open=True,
                tty=False,
                network_mode="none",  # No network access
                mem_limit="512m",  # 512MB RAM
                cpu_quota=50000,  # 50% of one CPU core (out of 100000)
                pids_limit=100,  # Max 100 processes
                volumes={str(workspace_path): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                user="coderunner",  # Non-root user
                remove=False,  # Don't auto-remove so we can inspect errors
            )

            self._containers[session_id] = container
            self._last_used[session_id] = time.time()
            logger.info(f"Created container {container_name} for session {session_id}")
            return container

        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise RuntimeError(f"Failed to create execution container: {e}")

    def execute_code(
        self, session_id: str, code: str, timeout: int = 30
    ) -> ExecutionResult:
        """Execute Python code in the session's container.

        Args:
            session_id: Chat session ID
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with stdout, stderr, and status
        """
        start_time = time.time()

        try:
            container = self.get_or_create_container(session_id)
        except RuntimeError as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                execution_time=0.0,
                error_message=str(e),
                exit_code=-1,
            )

        # Prepare code for execution with proper output capture
        # We use exec to preserve globals across executions
        wrapped_code = f"""
import sys
import io
import traceback

# Capture stdout and stderr
_stdout = io.StringIO()
_stderr = io.StringIO()
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = _stdout
sys.stderr = _stderr

try:
    exec('''
{code}
''', globals())
except Exception:
    traceback.print_exc(file=sys.stderr)
finally:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
    print(_stdout.getvalue(), end='')
    print(_stderr.getvalue(), file=sys.stderr, end='')
"""

        # Execute code in container
        try:
            exec_result = container.exec_run(
                ["python", "-c", wrapped_code],
                stdin=False,
                tty=False,
                demux=True,  # Separate stdout and stderr
                workdir="/workspace",
                user="coderunner",
            )

            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output

            stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

            execution_time = time.time() - start_time
            success = exit_code == 0

            error_message = None
            if not success:
                error_message = (
                    stderr if stderr else f"Execution failed with exit code {exit_code}"
                )

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                error_message=error_message,
                exit_code=exit_code,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing code in container: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=f"Execution error: {str(e)}",
                exit_code=-1,
            )

    def cleanup_session(self, session_id: str):
        """Remove container for a specific session.

        Args:
            session_id: Chat session ID
        """
        with self._lock:
            self._remove_container(session_id)

    def _remove_container(self, session_id: str):
        """Internal method to remove container (must hold lock).

        Args:
            session_id: Chat session ID
        """
        if session_id in self._containers:
            container = self._containers[session_id]
            try:
                container.stop(timeout=5)
                container.remove()
                logger.info(f"Removed container for session {session_id}")
            except Exception as e:
                logger.error(f"Error removing container: {e}")

            del self._containers[session_id]
            if session_id in self._last_used:
                del self._last_used[session_id]

    def _cleanup_oldest_container(self):
        """Remove the least recently used container (must hold lock)."""
        if not self._last_used:
            return

        oldest_session = min(self._last_used, key=self._last_used.get)
        logger.info(f"Removing oldest container: {oldest_session}")
        self._remove_container(oldest_session)

    def cleanup_all(self):
        """Remove all containers managed by this instance."""
        with self._lock:
            session_ids = list(self._containers.keys())
            for session_id in session_ids:
                self._remove_container(session_id)

    def _start_gc_thread(self):
        """Start background thread for idle container garbage collection."""
        if self._gc_thread is not None and self._gc_thread.is_alive():
            return

        def gc_loop():
            while not self._stop_gc.is_set():
                time.sleep(60)  # Check every minute
                self._gc_idle_containers()

        self._gc_thread = threading.Thread(target=gc_loop, daemon=True)
        self._gc_thread.start()
        logger.info("Started container garbage collection thread")

    def _gc_idle_containers(self):
        """Remove containers idle longer than timeout."""
        with self._lock:
            now = time.time()
            idle_sessions = [
                session_id
                for session_id, last_used in self._last_used.items()
                if (now - last_used) > self.idle_timeout
            ]

            for session_id in idle_sessions:
                logger.info(
                    f"Removing idle container for session {session_id} "
                    f"(idle for {now - self._last_used[session_id]:.0f}s)"
                )
                self._remove_container(session_id)

    def stop(self):
        """Stop the container manager and cleanup all resources."""
        self._stop_gc.set()
        if self._gc_thread:
            self._gc_thread.join(timeout=5)
        self.cleanup_all()
