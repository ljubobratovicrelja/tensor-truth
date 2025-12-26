"""Persistent Python session runner for maintaining state across executions.

This script runs inside the Docker container and maintains a persistent
Python session, executing code blocks while preserving variables and imports
between executions (like Jupyter notebooks).
"""

import json
import sys
import traceback
from pathlib import Path

# Global namespace shared across all executions
_session_globals = {}


def execute_code_block(code: str) -> dict:
    """Execute code in the persistent global namespace.

    Args:
        code: Python code to execute

    Returns:
        Dict with stdout, stderr, success status
    """
    import io

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    success = True
    error_msg = None

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute in persistent global namespace
        exec(code, _session_globals)

    except Exception as e:
        success = False
        error_msg = str(e)
        traceback.print_exc(file=stderr_capture)

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {
        "success": success,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error": error_msg,
    }


def main():
    """Main loop: read code from input file, execute, write results to output file."""
    workspace = Path("/workspace")
    input_file = workspace / ".code_input.json"
    output_file = workspace / ".code_output.json"
    ready_file = workspace / ".session_ready"

    # Signal that session is ready
    ready_file.touch()

    print("Session runner started, waiting for code...", file=sys.stderr, flush=True)

    while True:
        try:
            # Wait for input file to appear
            if not input_file.exists():
                import time

                time.sleep(0.1)
                continue

            # Read code request
            with open(input_file, "r") as f:
                request = json.load(f)

            code = request.get("code", "")
            request_id = request.get("request_id", "unknown")

            # Execute code
            result = execute_code_block(code)
            result["request_id"] = request_id

            # Write result
            with open(output_file, "w") as f:
                json.dump(result, f)

            # Remove input file to signal completion
            input_file.unlink()

            # Exit if shutdown requested
            if request.get("shutdown", False):
                break

        except Exception as e:
            # Write error result
            error_result = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Session runner error: {str(e)}",
                "request_id": (
                    request.get("request_id", "unknown")
                    if "request" in locals()
                    else "unknown"
                ),
            }
            try:
                with open(output_file, "w") as f:
                    json.dump(error_result, f)
                if input_file.exists():
                    input_file.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
