"""Service for managing MCP server configuration proposals.

Implements a two-phase approval pattern: the orchestrator agent creates a
proposal (not yet applied), which renders as an inline approval card in chat.
The user clicks Approve/Reject, which triggers a REST endpoint. This avoids
blocking the agent loop and is robust to disconnections.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Proposals expire after 30 minutes
PROPOSAL_TTL_SECONDS = 30 * 60


@dataclass
class MCPServerProposal:
    """A pending MCP server configuration change awaiting user approval."""

    proposal_id: str
    action: str  # "add" | "update" | "remove"
    config: Dict[str, Any]  # Full MCPServerConfig fields (for add/update) or just name
    target_name: str  # Server name being acted on
    status: str  # "pending" | "approved" | "rejected"
    created_at: float
    session_id: str
    summary: str  # Agent-generated human-readable description


class MCPProposalService:
    """Manages MCP server configuration proposals with TTL-based expiry."""

    def __init__(self) -> None:
        self._proposals: Dict[str, MCPServerProposal] = {}

    def _expire_old(self) -> None:
        """Remove proposals older than PROPOSAL_TTL_SECONDS."""
        now = time.time()
        expired = [
            pid
            for pid, p in self._proposals.items()
            if now - p.created_at > PROPOSAL_TTL_SECONDS
        ]
        for pid in expired:
            del self._proposals[pid]

    def create_proposal(
        self,
        action: str,
        config: Dict[str, Any],
        session_id: str,
        summary: str,
        target_name: str,
    ) -> MCPServerProposal:
        """Create a new proposal for an MCP server configuration change.

        Args:
            action: One of "add", "update", "remove".
            config: Server configuration dict.
            session_id: The chat session that initiated this proposal.
            summary: Human-readable description of the change.
            target_name: The server name being acted on.

        Returns:
            The created MCPServerProposal.
        """
        self._expire_old()

        proposal = MCPServerProposal(
            proposal_id=str(uuid.uuid4()),
            action=action,
            config=config,
            target_name=target_name,
            status="pending",
            created_at=time.time(),
            session_id=session_id,
            summary=summary,
        )
        self._proposals[proposal.proposal_id] = proposal
        logger.info(
            "Created MCP proposal %s: %s server '%s'",
            proposal.proposal_id,
            action,
            target_name,
        )
        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[MCPServerProposal]:
        """Get a proposal by ID, or None if not found/expired."""
        self._expire_old()
        return self._proposals.get(proposal_id)

    def approve_proposal(
        self,
        proposal_id: str,
        mcp_server_service: Any,
        tool_service: Any = None,
    ) -> Dict[str, Any]:
        """Approve a proposal and apply the configuration change.

        Args:
            proposal_id: The proposal to approve.
            mcp_server_service: MCPServerService instance for CRUD operations.
            tool_service: Optional ToolService for reloading tools after changes.

        Returns:
            Dict with result details.

        Raises:
            ValueError: If proposal not found, already resolved, or action fails.
        """
        self._expire_old()
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal '{proposal_id}' not found or expired")
        if proposal.status != "pending":
            raise ValueError(
                f"Proposal '{proposal_id}' already {proposal.status}"
            )

        try:
            if proposal.action == "add":
                result = mcp_server_service.add(proposal.config)
            elif proposal.action == "update":
                result = mcp_server_service.update(
                    proposal.target_name, proposal.config
                )
            elif proposal.action == "remove":
                mcp_server_service.remove(proposal.target_name)
                result = {"name": proposal.target_name, "removed": True}
            else:
                raise ValueError(f"Unknown action: {proposal.action}")
        except Exception as e:
            logger.error("Failed to apply proposal %s: %s", proposal_id, e)
            raise

        proposal.status = "approved"
        logger.info("Approved MCP proposal %s", proposal_id)
        return result

    def reject_proposal(self, proposal_id: str) -> None:
        """Reject a proposal without applying changes.

        Args:
            proposal_id: The proposal to reject.

        Raises:
            ValueError: If proposal not found or already resolved.
        """
        self._expire_old()
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal '{proposal_id}' not found or expired")
        if proposal.status != "pending":
            raise ValueError(
                f"Proposal '{proposal_id}' already {proposal.status}"
            )

        proposal.status = "rejected"
        logger.info("Rejected MCP proposal %s", proposal_id)
