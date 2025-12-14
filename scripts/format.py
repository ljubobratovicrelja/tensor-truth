"""
Formatting script, also used in pre-commit hooks.
"""

import subprocess
import sys

TARGETS = ["src/tensortruth", "scripts", "tests"]


def run_tool(command):
    try:
        subprocess.run(command, check=True, shell=False)
    except subprocess.CalledProcessError:
        print(f"Error running: {' '.join(command)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Command not found: {command[0]}. Is it installed in your venv?")
        sys.exit(1)


def main():
    print("Starting code cleanup...")

    print("Running isort...")
    run_tool(["isort"] + TARGETS)

    print("Running black...")
    run_tool(["black"] + TARGETS)

    print("Cleanup done.")


if __name__ == "__main__":
    main()
