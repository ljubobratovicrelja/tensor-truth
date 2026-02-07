#!/usr/bin/env python3
"""
Script to rebuild and publish tensor-truth to PyPI.

This script:
1. Verifies no uncommitted changes in the repository
2. Verifies the repository is in sync with HEAD
3. Checks the latest version on PyPI
4. Compares it with the local version in pyproject.toml
5. Verifies the local version has been incremented
6. Checks that no tag exists with the same version
7. Verifies the latest git tag is behind the current version
8. Creates and pushes a git tag with the version
9. Cleans the dist/ directory
10. Builds the project
11. Uploads to PyPI using twine
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


def check_git_status():
    """Check if there are uncommitted changes in the repository"""
    logger.info("Checking for uncommitted changes...")

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.stdout.strip():
            logger.error("Repository has uncommitted changes:")
            logger.error(result.stdout)
            logger.error("Please commit or stash all changes before publishing.")
            return False

        logger.info("Repository is clean (no uncommitted changes)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking git status: {e}")
        return False


def check_git_sync_with_head():
    """Check if the repository is in sync with remote HEAD"""
    logger.info("Checking if repository is in sync with remote...")

    try:
        # Fetch latest from remote
        subprocess.run(
            ["git", "fetch"],
            capture_output=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        current_branch = result.stdout.strip()

        # Check if local is behind remote
        result = subprocess.run(
            ["git", "rev-list", "--count", f"HEAD..origin/{current_branch}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        behind_count = int(result.stdout.strip())

        # Check if local is ahead of remote
        result = subprocess.run(
            ["git", "rev-list", "--count", f"origin/{current_branch}..HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        ahead_count = int(result.stdout.strip())

        if behind_count > 0:
            logger.error(
                f"Local branch is {behind_count} commit(s) behind origin/{current_branch}"
            )
            logger.error("Please pull the latest changes before publishing.")
            return False

        if ahead_count > 0:
            logger.warning(
                f"Local branch is {ahead_count} commit(s) ahead of origin/{current_branch}"
            )
            logger.warning("Make sure you've pushed your changes to the remote.")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                return False

        logger.info(f"Repository is in sync with origin/{current_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking git sync: {e}")
        return False


def check_tag_exists(version):
    """Check if a git tag with the given version already exists"""
    logger.info(f"Checking if tag v{version} already exists...")

    try:
        result = subprocess.run(
            ["git", "tag", "-l", f"v{version}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.stdout.strip():
            logger.error(f"Tag v{version} already exists!")
            logger.error("Please increment the version in pyproject.toml")
            return True

        logger.info(f"Tag v{version} does not exist (good)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking git tags: {e}")
        return True


def get_latest_tag():
    """Get the latest git tag"""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        tag = result.stdout.strip()
        # Remove 'v' prefix if present
        version = tag.lstrip("v")
        logger.info(f"Latest git tag: {tag} (version: {version})")
        return version
    except subprocess.CalledProcessError:
        logger.info("No git tags found in repository")
        return "0.0.0"


def create_and_push_tag(version):
    """Create a git tag with the given version and push it to origin"""
    tag_name = f"v{version}"
    logger.info(f"Creating tag {tag_name}...")

    try:
        # Create the tag
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info(f"Tag {tag_name} created successfully")

        # Push the tag to origin
        logger.info(f"Pushing tag {tag_name} to origin...")
        subprocess.run(
            ["git", "push", "origin", tag_name],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info(f"Tag {tag_name} pushed to origin successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating or pushing tag: {e}")
        return False


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


def build_frontend():
    """Build the React frontend for bundling into the Python package."""
    logger.info("Building frontend...")

    script_path = Path(__file__).parent / "build_frontend.sh"
    if not script_path.exists():
        logger.error(f"Frontend build script not found at {script_path}")
        return False

    try:
        subprocess.run(
            ["bash", str(script_path)],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info("Frontend built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend build failed: {e}")
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

    # Step 1: Check git status (no uncommitted changes)
    logger.info("Step 1: Checking git status...")
    if not check_git_status():
        logger.error("Git status check failed. Aborting.")
        sys.exit(1)

    # Step 2: Check git sync with remote
    logger.info("Step 2: Checking git sync with remote...")
    if not check_git_sync_with_head():
        logger.error("Git sync check failed. Aborting.")
        sys.exit(1)

    # Step 3: Get versions
    logger.info("Step 3: Checking versions...")
    local_version = get_local_version()
    pypi_version = get_pypi_version()

    if pypi_version is None:
        logger.error("Failed to retrieve PyPI version. Aborting.")
        sys.exit(1)

    # Step 4: Compare versions with PyPI
    logger.info("Step 4: Comparing versions with PyPI...")
    if not compare_versions(local_version, pypi_version):
        logger.error("Version check failed!")
        logger.info("Please increment the version in pyproject.toml before publishing.")
        sys.exit(1)

    # Step 5: Check if tag already exists
    logger.info("Step 5: Checking if git tag already exists...")
    if check_tag_exists(local_version):
        logger.error("Tag already exists. Aborting.")
        sys.exit(1)

    # Step 6: Compare with latest git tag
    logger.info("Step 6: Comparing with latest git tag...")
    latest_tag = get_latest_tag()
    if not compare_versions(local_version, latest_tag):
        logger.error("Version is not greater than the latest git tag!")
        logger.info("Please increment the version in pyproject.toml.")
        sys.exit(1)

    # Step 7: Create and push git tag
    logger.info("Step 7: Creating and pushing git tag...")
    if not create_and_push_tag(local_version):
        logger.error("Failed to create or push tag. Aborting.")
        sys.exit(1)

    # Step 8: Clean dist/
    logger.info("Step 8: Cleaning dist/ directory...")
    clean_dist()

    # Step 9: Build frontend
    logger.info("Step 9: Building frontend...")
    if not build_frontend():
        logger.error("Frontend build failed. Aborting.")
        sys.exit(1)

    # Step 10: Build project
    logger.info("Step 10: Building project...")
    if not build_project():
        logger.error("Build failed. Aborting.")
        sys.exit(1)

    # Step 11: Upload to PyPI
    logger.info("Step 11: Uploading to PyPI...")

    # Ask for confirmation before uploading
    response = input("Ready to upload to PyPI. Continue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        logger.warning("Upload cancelled by user.")
        sys.exit(0)

    if not upload_to_pypi():
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"All done! Version {local_version} published successfully!")
    logger.info(f"Tag v{local_version} created and pushed to origin")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
