"""Tests for MCPProposalService.

Tests proposal CRUD, approval delegation to MCPServerService,
rejection, TTL expiry, and error handling.
"""

import time
from unittest.mock import MagicMock

import pytest

from tensortruth.services.mcp_proposal_service import (
    PROPOSAL_TTL_SECONDS,
    MCPProposalService,
    MCPServerProposal,
)


@pytest.fixture
def service():
    """Fresh MCPProposalService instance."""
    return MCPProposalService()


@pytest.fixture
def mcp_server_service():
    """Mock MCPServerService."""
    svc = MagicMock()
    svc.add.return_value = {"name": "test-server", "builtin": False}
    svc.update.return_value = {"name": "test-server", "builtin": False}
    svc.remove.return_value = None
    return svc


class TestCreateProposal:
    def test_creates_proposal_with_pending_status(self, service):
        proposal = service.create_proposal(
            action="add",
            config={"name": "ctx7", "command": "npx"},
            session_id="sess-1",
            summary="Add Context7 MCP server",
            target_name="ctx7",
        )
        assert isinstance(proposal, MCPServerProposal)
        assert proposal.status == "pending"
        assert proposal.action == "add"
        assert proposal.target_name == "ctx7"
        assert proposal.session_id == "sess-1"
        assert proposal.summary == "Add Context7 MCP server"

    def test_generates_unique_ids(self, service):
        p1 = service.create_proposal("add", {}, "s1", "s1", "n1")
        p2 = service.create_proposal("add", {}, "s1", "s2", "n2")
        assert p1.proposal_id != p2.proposal_id

    def test_stores_proposal_for_retrieval(self, service):
        proposal = service.create_proposal("add", {"name": "x"}, "s1", "test", "x")
        retrieved = service.get_proposal(proposal.proposal_id)
        assert retrieved is proposal


class TestGetProposal:
    def test_returns_none_for_unknown_id(self, service):
        assert service.get_proposal("nonexistent") is None

    def test_returns_none_for_expired_proposal(self, service):
        proposal = service.create_proposal("add", {}, "s1", "test", "x")
        # Manually expire it
        proposal.created_at = time.time() - PROPOSAL_TTL_SECONDS - 1
        assert service.get_proposal(proposal.proposal_id) is None


class TestApproveProposal:
    def test_delegates_add_to_mcp_server_service(self, service, mcp_server_service):
        config = {"name": "test", "command": "npx", "args": ["-y", "pkg"]}
        proposal = service.create_proposal("add", config, "s1", "Add test", "test")

        result = service.approve_proposal(proposal.proposal_id, mcp_server_service)

        mcp_server_service.add.assert_called_once_with(config)
        assert proposal.status == "approved"
        assert result == {"name": "test-server", "builtin": False}

    def test_delegates_update_to_mcp_server_service(self, service, mcp_server_service):
        config = {"description": "Updated desc"}
        proposal = service.create_proposal("update", config, "s1", "Update", "test")

        service.approve_proposal(proposal.proposal_id, mcp_server_service)

        mcp_server_service.update.assert_called_once_with("test", config)
        assert proposal.status == "approved"

    def test_delegates_remove_to_mcp_server_service(self, service, mcp_server_service):
        proposal = service.create_proposal("remove", {}, "s1", "Remove", "test")

        result = service.approve_proposal(proposal.proposal_id, mcp_server_service)

        mcp_server_service.remove.assert_called_once_with("test")
        assert result == {"name": "test", "removed": True}
        assert proposal.status == "approved"

    def test_raises_for_unknown_proposal(self, service, mcp_server_service):
        with pytest.raises(ValueError, match="not found"):
            service.approve_proposal("nonexistent", mcp_server_service)

    def test_raises_for_already_approved(self, service, mcp_server_service):
        proposal = service.create_proposal("add", {"name": "x"}, "s1", "t", "x")
        service.approve_proposal(proposal.proposal_id, mcp_server_service)

        with pytest.raises(ValueError, match="already approved"):
            service.approve_proposal(proposal.proposal_id, mcp_server_service)

    def test_raises_for_already_rejected(self, service, mcp_server_service):
        proposal = service.create_proposal("add", {"name": "x"}, "s1", "t", "x")
        service.reject_proposal(proposal.proposal_id)

        with pytest.raises(ValueError, match="already rejected"):
            service.approve_proposal(proposal.proposal_id, mcp_server_service)

    def test_propagates_service_error(self, service, mcp_server_service):
        mcp_server_service.add.side_effect = ValueError("Server already exists")
        proposal = service.create_proposal("add", {"name": "x"}, "s1", "t", "x")

        with pytest.raises(ValueError, match="Server already exists"):
            service.approve_proposal(proposal.proposal_id, mcp_server_service)

        # Status should NOT be changed on failure
        assert proposal.status == "pending"


class TestRejectProposal:
    def test_marks_proposal_as_rejected(self, service):
        proposal = service.create_proposal("add", {}, "s1", "test", "x")
        service.reject_proposal(proposal.proposal_id)
        assert proposal.status == "rejected"

    def test_raises_for_unknown_proposal(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.reject_proposal("nonexistent")

    def test_raises_for_already_resolved(self, service):
        proposal = service.create_proposal("add", {}, "s1", "test", "x")
        service.reject_proposal(proposal.proposal_id)

        with pytest.raises(ValueError, match="already rejected"):
            service.reject_proposal(proposal.proposal_id)


class TestTTLExpiry:
    def test_expired_proposals_are_cleaned_up(self, service):
        p1 = service.create_proposal("add", {}, "s1", "old", "x")
        p1.created_at = time.time() - PROPOSAL_TTL_SECONDS - 1

        p2 = service.create_proposal("add", {}, "s1", "new", "y")

        # p1 should be expired, p2 should remain
        assert service.get_proposal(p1.proposal_id) is None
        assert service.get_proposal(p2.proposal_id) is p2
