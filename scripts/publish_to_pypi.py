#!/usr/bin/env python3
"""
Script to rebuild and publish tensor-truth to PyPI.

This script:
1. Checks the latest version on PyPI
2. Compares it with the local version in pyproject.toml
3. Verifies the local version has been incremented
4. Cleans the dist/ directory
5. Builds the project
6. Uploads to PyPI using twine
"""

import logging
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_project_root():
    """Verify that the script is being run from the project root"""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        logger.error(f"pyproject.toml not found at {pyproject_path}")
        logger.error(
            "Please ensure the script is in the scripts/ directory of the project"
        )
        return False

    # Verify it's the tensor-truth project
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            project_name = data.get("project", {}).get("name", "")
            if project_name != "tensor-truth":
                logger.error(
                    f"Found project '{project_name}' but expected 'tensor-truth'"
                )
                return False
    except Exception as e:
        logger.error(f"Failed to read pyproject.toml: {e}")
        return False

    logger.info(f"Project root verified: {project_root}")
    return True


def get_local_version():
    """Get the version from pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    version = data["project"]["version"]
    logger.info(f"Local version: {version}")
    return version


def get_pypi_version(package_name="tensor-truth"):
    """Get the latest version from PyPI"""
    try:
        result = subprocess.run(
            ["pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse output to find LATEST version
        for line in result.stdout.split("\n"):
            if "LATEST:" in line:
                version = line.split("LATEST:")[1].strip()
                logger.info(f"PyPI version: {version}")
                return version

        # If no LATEST found, package might not exist
        logger.warning(
            f"Package '{package_name}' not found on PyPI (this might be the first release)"
        )
        return "0.0.0"

    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking PyPI version: {e}")
        return None


def compare_versions(local_ver, pypi_ver):
    """Compare version strings"""

    def parse_version(v):
        return tuple(map(int, v.split(".")))

    local_parts = parse_version(local_ver)
    pypi_parts = parse_version(pypi_ver)

    if local_parts > pypi_parts:
        logger.info(
            f"Local version ({local_ver}) is greater than PyPI version ({pypi_ver})"
        )
        return True
    elif local_parts == pypi_parts:
        logger.warning(f"Local version ({local_ver}) equals PyPI version ({pypi_ver})")
        return False
    else:
        logger.error(
            f"Local version ({local_ver}) is less than PyPI version ({pypi_ver})"
        )
        return False


def clean_dist():
    """Clean the dist/ directory"""
    dist_path = Path(__file__).parent.parent / "dist"

    if dist_path.exists():
        logger.info(f"Cleaning {dist_path}...")
        shutil.rmtree(dist_path)
        logger.info("dist/ directory cleaned")
    else:
        logger.info("dist/ directory doesn't exist, nothing to clean")


def build_project():
    """Build the project"""
    logger.info("Building project...")

    try:
        subprocess.run(
            [sys.executable, "-m", "build"],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info("Project built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        return False


def upload_to_pypi():
    """Upload to PyPI using twine"""
    logger.info("Uploading to PyPI...")

    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "upload", "dist/*"],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info("Successfully uploaded to PyPI!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Upload failed: {e}")
        logger.info("Make sure you have:")
        logger.info(
            "  1. Configured PyPI credentials (~/.pypirc or environment variables)"
        )
        logger.info("  2. Installed twine: pip install twine")
        return False


def main():
    logger.info("=" * 60)
    logger.info("PyPI Publishing Script for tensor-truth")
    logger.info("=" * 60)

    # Step 0: Verify project root
    logger.info("Step 0: Verifying project root...")
    if not verify_project_root():
        logger.error("Project root verification failed. Aborting.")
        sys.exit(1)

    # Step 1: Get versions
    logger.info("Step 1: Checking versions...")
    local_version = get_local_version()
    pypi_version = get_pypi_version()

    if pypi_version is None:
        logger.error("Failed to retrieve PyPI version. Aborting.")
        sys.exit(1)

    # Step 2: Compare versions
    logger.info("Step 2: Comparing versions...")
    if not compare_versions(local_version, pypi_version):
        logger.error("Version check failed!")
        logger.info("Please increment the version in pyproject.toml before publishing.")
        sys.exit(1)

    # Step 3: Clean dist/
    logger.info("Step 3: Cleaning dist/ directory...")
    clean_dist()

    # Step 4: Build project
    logger.info("Step 4: Building project...")
    if not build_project():
        logger.error("Build failed. Aborting.")
        sys.exit(1)

    # Step 5: Upload to PyPI
    logger.info("Step 5: Uploading to PyPI...")

    # Ask for confirmation before uploading
    response = input("Ready to upload to PyPI. Continue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        logger.warning("Upload cancelled by user.")
        sys.exit(0)

    if not upload_to_pypi():
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("All done! Package published successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
