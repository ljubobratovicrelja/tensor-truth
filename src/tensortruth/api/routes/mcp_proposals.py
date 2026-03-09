"""REST endpoints for MCP server proposal approve/reject flow."""

import logging

from fastapi import APIRouter, HTTPException

from tensortruth.api.deps import get_mcp_proposal_service, get_tool_service
from tensortruth.services.mcp_server_service import MCPServerService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get proposal status."""
    service = get_mcp_proposal_service()
    proposal = service.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found or expired")
    return {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action,
        "config": proposal.config,
        "target_name": proposal.target_name,
        "status": proposal.status,
        "summary": proposal.summary,
        "session_id": proposal.session_id,
    }


@router.post("/{proposal_id}/approve")
async def approve_proposal(proposal_id: str):
    """Approve a pending proposal and apply the MCP server configuration change."""
    proposal_service = get_mcp_proposal_service()
    mcp_server_service = MCPServerService()
    tool_service = get_tool_service()

    try:
        result = proposal_service.approve_proposal(
            proposal_id, mcp_server_service, tool_service
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Reload tools so the new/changed MCP server is available
    try:
        await tool_service.reload()
    except Exception as e:
        logger.warning("Failed to reload tools after approval: %s", e)

    return {"status": "approved", "result": result}


@router.post("/{proposal_id}/reject")
async def reject_proposal(proposal_id: str):
    """Reject a pending proposal without applying changes."""
    proposal_service = get_mcp_proposal_service()

    try:
        proposal_service.reject_proposal(proposal_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "rejected"}
