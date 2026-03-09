"""REST endpoints for extension management."""

import logging

from fastapi import APIRouter, HTTPException

from tensortruth.api.schemas.extension import (
    ExtensionInstallRequest,
    ExtensionInstallResponse,
    ExtensionLibraryResponse,
    ExtensionListResponse,
)
from tensortruth.services.extension_library_service import ExtensionLibraryService

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service() -> ExtensionLibraryService:
    from tensortruth.app_utils.config import load_config

    config = load_config()
    return ExtensionLibraryService(catalog_url=config.extension_catalog_url)


def _raise_for_value_error(e: ValueError) -> None:
    """Map ValueError to appropriate HTTP status code."""
    msg = str(e)
    if "not found" in msg:
        raise HTTPException(status_code=404, detail=msg)
    raise HTTPException(status_code=400, detail=msg)


@router.get("/", response_model=ExtensionListResponse)
async def list_extensions():
    """List installed user extensions."""
    service = _get_service()
    extensions = service.list_installed()
    return {"extensions": extensions}


@router.get("/library", response_model=ExtensionLibraryResponse)
async def list_library():
    """List all extensions available in the library."""
    service = _get_service()
    try:
        extensions = service.list_library()
    except Exception:
        logger.warning("Failed to fetch extension library", exc_info=True)
        extensions = []
    return {"extensions": extensions}


@router.post("/install", response_model=ExtensionInstallResponse)
async def install_extension(request: ExtensionInstallRequest):
    """Install an extension from the library."""
    service = _get_service()
    try:
        filename = service.install(request.type, request.filename)
        return {"installed": [filename], "errors": []}
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        _raise_for_value_error(e)


@router.delete("/{ext_type}/{filename}", status_code=204)
async def uninstall_extension(ext_type: str, filename: str):
    """Uninstall a user extension."""
    service = _get_service()
    try:
        service.uninstall(ext_type, filename)
    except ValueError as e:
        _raise_for_value_error(e)
