"""General-purpose tool confirmation service.

Implements a blocking confirmation pattern: when a tool needs user approval,
it creates a ToolConfirmationRequest containing an asyncio.Event, emits a
WebSocket event, then awaits the event. The REST approve/reject endpoint
calls event.set(), unblocking the tool. Since asyncio.Event.wait() is
non-blocking to the event loop, the REST handler executes normally on the
same loop.

Replaces the MCP-specific MCPProposalService with a tool-agnostic system
that any tool can use.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Confirmations expire after 5 minutes
CONFIRMATION_TTL_SECONDS = 5 * 60


@dataclass
class ToolConfirmationRequest:
    """A pending tool action awaiting user approval."""

    confirmation_id: str
    tool_name: str  # e.g. "manage_mcp_server"
    action_type: str  # e.g. "mcp_add", "mcp_remove", "bash_execute"
    title: str  # Short: "Add MCP server 'context7'"
    summary: str  # Longer human-readable description
    details: Dict[str, Any]  # Action-specific structured data
    status: str  # "pending" | "approved" | "rejected" | "expired"
    created_at: float
    session_id: str
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _outcome: Optional[str] = field(default=None, repr=False)


class ToolConfirmationService:
    """Manages tool confirmation requests with asyncio.Event-based blocking."""

    def __init__(self) -> None:
        self._confirmations: Dict[str, ToolConfirmationRequest] = {}

    def _expire_old(self) -> None:
        """Remove confirmations older than CONFIRMATION_TTL_SECONDS."""
        now = time.time()
        expired = [
            cid
            for cid, c in self._confirmations.items()
            if now - c.created_at > CONFIRMATION_TTL_SECONDS
        ]
        for cid in expired:
            c = self._confirmations.pop(cid)
            if c.status == "pending":
                c.status = "expired"
                c._outcome = "expired"
                c._event.set()  # Unblock any waiting tool

    def create_confirmation(
        self,
        tool_name: str,
        action_type: str,
        title: str,
        summary: str,
        details: Dict[str, Any],
        session_id: str,
    ) -> ToolConfirmationRequest:
        """Create a new confirmation request (non-blocking).

        Returns:
            The created ToolConfirmationRequest.
        """
        self._expire_old()

        confirmation = ToolConfirmationRequest(
            confirmation_id=str(uuid.uuid4()),
            tool_name=tool_name,
            action_type=action_type,
            title=title,
            summary=summary,
            details=details,
            status="pending",
            created_at=time.time(),
            session_id=session_id,
        )
        self._confirmations[confirmation.confirmation_id] = confirmation
        logger.info(
            "Created confirmation %s: %s/%s '%s'",
            confirmation.confirmation_id,
            tool_name,
            action_type,
            title,
        )
        return confirmation

    async def await_resolution(
        self, confirmation_id: str, timeout: float = 300.0
    ) -> str:
        """Block until the confirmation is resolved or times out.

        Returns:
            "approved", "rejected", or "expired".
        """
        confirmation = self._confirmations.get(confirmation_id)
        if confirmation is None:
            return "expired"

        try:
            await asyncio.wait_for(confirmation._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            confirmation.status = "expired"
            confirmation._outcome = "expired"
            logger.info("Confirmation %s timed out", confirmation_id)

        return confirmation._outcome or "expired"

    def resolve(self, confirmation_id: str, outcome: str) -> None:
        """Resolve a confirmation (called by REST endpoint).

        Args:
            confirmation_id: The confirmation to resolve.
            outcome: "approved" or "rejected".

        Raises:
            ValueError: If confirmation not found or already resolved.
        """
        self._expire_old()
        confirmation = self._confirmations.get(confirmation_id)
        if confirmation is None:
            raise ValueError(f"Confirmation '{confirmation_id}' not found or expired")
        if confirmation.status != "pending":
            raise ValueError(
                f"Confirmation '{confirmation_id}' already {confirmation.status}"
            )

        confirmation.status = outcome
        confirmation._outcome = outcome
        confirmation._event.set()
        logger.info("Resolved confirmation %s: %s", confirmation_id, outcome)

    def get_status(self, confirmation_id: str) -> Optional[ToolConfirmationRequest]:
        """Get a confirmation by ID, or None if not found/expired."""
        self._expire_old()
        return self._confirmations.get(confirmation_id)
