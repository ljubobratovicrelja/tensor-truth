"""Tests for ToolConfirmationService.

Tests confirmation creation, resolution via asyncio.Event,
TTL expiry, and error handling.
"""

import asyncio
import time

import pytest

from tensortruth.services.tool_confirmation_service import (
    CONFIRMATION_TTL_SECONDS,
    ToolConfirmationService,
)


@pytest.fixture
def service():
    """Fresh ToolConfirmationService instance."""
    return ToolConfirmationService()


class TestCreateConfirmation:
    def test_creates_with_pending_status(self, service):
        c = service.create_confirmation(
            tool_name="manage_mcp_server",
            action_type="mcp_add",
            title="Add MCP server 'ctx7'",
            summary="Add Context7 MCP server",
            details={"config": {"name": "ctx7"}},
            session_id="sess-1",
        )
        assert c.status == "pending"
        assert c.tool_name == "manage_mcp_server"
        assert c.action_type == "mcp_add"
        assert c.session_id == "sess-1"

    def test_generates_unique_ids(self, service):
        c1 = service.create_confirmation("t", "a", "t1", "s", {}, "s1")
        c2 = service.create_confirmation("t", "a", "t2", "s", {}, "s1")
        assert c1.confirmation_id != c2.confirmation_id

    def test_stores_for_retrieval(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")
        retrieved = service.get_status(c.confirmation_id)
        assert retrieved is c


class TestResolve:
    def test_approve_sets_status(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")
        service.resolve(c.confirmation_id, "approved")
        assert c.status == "approved"
        assert c._outcome == "approved"
        assert c._event.is_set()

    def test_reject_sets_status(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")
        service.resolve(c.confirmation_id, "rejected")
        assert c.status == "rejected"
        assert c._outcome == "rejected"

    def test_raises_for_unknown_id(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.resolve("nonexistent", "approved")

    def test_raises_for_already_resolved(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")
        service.resolve(c.confirmation_id, "approved")
        with pytest.raises(ValueError, match="already approved"):
            service.resolve(c.confirmation_id, "rejected")


class TestAwaitResolution:
    @pytest.mark.asyncio
    async def test_unblocks_on_approve(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")

        async def approve_after_delay():
            await asyncio.sleep(0.05)
            service.resolve(c.confirmation_id, "approved")

        asyncio.create_task(approve_after_delay())
        result = await service.await_resolution(c.confirmation_id, timeout=5.0)
        assert result == "approved"

    @pytest.mark.asyncio
    async def test_unblocks_on_reject(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")

        async def reject_after_delay():
            await asyncio.sleep(0.05)
            service.resolve(c.confirmation_id, "rejected")

        asyncio.create_task(reject_after_delay())
        result = await service.await_resolution(c.confirmation_id, timeout=5.0)
        assert result == "rejected"

    @pytest.mark.asyncio
    async def test_returns_expired_on_timeout(self, service):
        c = service.create_confirmation("t", "a", "title", "summary", {}, "s1")
        result = await service.await_resolution(c.confirmation_id, timeout=0.1)
        assert result == "expired"
        assert c.status == "expired"

    @pytest.mark.asyncio
    async def test_returns_expired_for_unknown_id(self, service):
        result = await service.await_resolution("nonexistent", timeout=0.1)
        assert result == "expired"


class TestTTLExpiry:
    def test_expired_confirmations_are_cleaned_up(self, service):
        c1 = service.create_confirmation("t", "a", "old", "s", {}, "s1")
        c1.created_at = time.time() - CONFIRMATION_TTL_SECONDS - 1

        c2 = service.create_confirmation("t", "a", "new", "s", {}, "s1")

        assert service.get_status(c1.confirmation_id) is None
        assert service.get_status(c2.confirmation_id) is c2

    def test_expired_pending_gets_event_set(self, service):
        c = service.create_confirmation("t", "a", "title", "s", {}, "s1")
        c.created_at = time.time() - CONFIRMATION_TTL_SECONDS - 1

        # Trigger expiry
        service.get_status("dummy")

        assert c.status == "expired"
        assert c._event.is_set()
