"""REST endpoints for tool confirmation approve/reject flow.

Generic endpoints replacing the MCP-specific mcp_proposals routes.
The approve/reject endpoints only signal the asyncio.Event — the tool
wrapper applies the action after being unblocked.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tensortruth.api.deps import get_tool_confirmation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{confirmation_id}")
async def get_confirmation(confirmation_id: str):
    """Get confirmation status."""
    service = get_tool_confirmation_service()
    confirmation = service.get_status(confirmation_id)
    if confirmation is None:
        raise HTTPException(status_code=404, detail="Confirmation not found or expired")
    return {
        "confirmation_id": confirmation.confirmation_id,
        "tool_name": confirmation.tool_name,
        "action_type": confirmation.action_type,
        "title": confirmation.title,
        "summary": confirmation.summary,
        "details": confirmation.details,
        "status": confirmation.status,
        "session_id": confirmation.session_id,
    }


@router.post("/{confirmation_id}/approve")
async def approve_confirmation(confirmation_id: str):
    """Approve a pending confirmation (signals the waiting tool)."""
    service = get_tool_confirmation_service()
    try:
        service.resolve(confirmation_id, "approved")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "approved"}


class ResolveConfirmationRequest(BaseModel):
    """Body for resolving a confirmation with an arbitrary outcome token."""

    outcome: str


@router.post("/{confirmation_id}/resolve")
async def resolve_confirmation(confirmation_id: str, body: ResolveConfirmationRequest):
    """Resolve a confirmation with a specific outcome.

    Used by cards that offer more than approve/reject — e.g. add_arxiv_paper
    offers "add directly to context" as an alternative to indexing.
    """
    outcome = (body.outcome or "").strip()
    if not outcome:
        raise HTTPException(status_code=400, detail="outcome must not be empty")
    service = get_tool_confirmation_service()
    try:
        service.resolve(confirmation_id, outcome)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "approved" if outcome not in ("approved", "rejected") else outcome,
        "outcome": outcome,
    }


@router.post("/{confirmation_id}/reject")
async def reject_confirmation(confirmation_id: str):
    """Reject a pending confirmation (signals the waiting tool)."""
    service = get_tool_confirmation_service()
    try:
        service.resolve(confirmation_id, "rejected")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "rejected"}
