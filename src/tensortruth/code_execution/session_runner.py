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

# Counter for auto-generated plot filenames
_plot_counter = 0


def execute_code_block(code: str) -> dict:
    """Execute code in the persistent global namespace.

    Args:
        code: Python code to execute

    Returns:
        Dict with stdout, stderr, success status
    """
    import io

    global _plot_counter

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

        # Auto-save any matplotlib figures that are open
        try:
            import matplotlib.pyplot as plt

            # Get all figure numbers
            fig_nums = plt.get_fignums()

            if fig_nums:
                # Save each figure
                for fig_num in fig_nums:
                    fig = plt.figure(fig_num)

                    # Generate filename
                    _plot_counter += 1
                    filename = f"plot_{_plot_counter}.png"

                    # Save figure
                    fig.savefig(filename, dpi=150, bbox_inches="tight")
                    print(
                        f"Auto-saved figure {fig_num} to {filename}",
                        file=stdout_capture,
                    )

                # Close all figures to free memory
                plt.close("all")

        except ImportError:
            # matplotlib not imported in this code block, skip
            pass
        except Exception as e:
            # Don't fail the whole execution if auto-save fails
            print(
                f"Warning: Failed to auto-save matplotlib figures: {e}",
                file=stderr_capture,
            )

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
