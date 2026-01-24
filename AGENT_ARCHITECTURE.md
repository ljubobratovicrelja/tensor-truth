# Agent Architecture Guide

This document captures architectural principles and patterns learned from implementing the unified web search streaming system. Use this as a reference when building the agent base API and new agent implementations.

---

## Core Principles

### 1. No Parallel Implementations

**Never create parallel UI components or message types for agent-specific functionality when existing infrastructure can be extended.**

Bad (what we had before):
```
WebSearchSources component  ←  web_sources message type
SourcesList component       ←  sources message type
```

Good (unified approach):
```
SourcesList component       ←  sources message type (with doc_type: "web" in metadata)
```

The frontend already handles streaming, sources, and progress indicators. Agents should emit messages that fit into this existing system rather than creating new parallel paths.

### 2. Extend, Don't Duplicate

When adding agent functionality:

1. **Check what exists** - Look at RAGChunk, SourceNode, StreamStatus patterns
2. **Add to existing types** - Extend metadata fields rather than new message types
3. **Reuse UI components** - Add new `doc_type` cases to existing components
4. **Same streaming flow** - Once LLM generation starts, stream exactly like RAG

### 3. Agent-Specific vs Shared Phases

Agents have two types of phases:

| Phase Type | Examples | How to Handle |
|------------|----------|---------------|
| **Agent-specific** | Searching, fetching, browsing | `agent_progress` messages (OK to be unique) |
| **Shared with LLM** | Generating response | Standard `status` + `token` messages |

The `AgentProgress` component handles agent-specific phases. Once the agent transitions to LLM generation, use the standard streaming pattern.

---

## Message Flow Architecture

### Target Flow (Unified)

```
agent_progress(searching) → agent_progress(fetching) → agent_progress(summarizing) →
status(generating) → token → token → ... → sources → done(title_pending)
```

### Key Messages

| Message Type | Purpose | When to Use |
|--------------|---------|-------------|
| `agent_progress` | Agent-specific phase updates | Search, fetch, browse phases |
| `status` | Pipeline status | "generating" before tokens |
| `token` | Streaming LLM output | During response generation |
| `sources` | Retrieved/fetched sources | After generation, before done |
| `done` | Completion with full response | Always at end |

---

## Data Models

### WebSearchChunk (Agent Streaming Pattern)

```python
@dataclass
class WebSearchChunk:
    """A chunk from agent streaming, compatible with chat consumption."""

    agent_progress: Optional[Dict[str, Any]] = None  # For agent-specific phases
    status: Optional[str] = None                      # "generating" when LLM starts
    token: Optional[str] = None                       # LLM output token
    sources: Optional[List[SourceNode]] = None        # Final sources
    is_complete: bool = False
```

Use this pattern for any new agent. The chunk type unifies:
- Agent progress (unique to each agent)
- Standard pipeline status (shared)
- Token streaming (shared)
- Source delivery (shared)

### Source Conversion

Always convert agent-specific sources to SourceNode format:

```python
def web_source_to_source_node(source: WebSearchSource) -> Dict[str, Any]:
    """Convert agent source to SourceNode format for unified UI."""
    return {
        "text": source.snippet or "",
        "score": 1.0 if source.status == "success" else 0.0,
        "metadata": {
            "source_url": source.url,
            "display_name": source.title,
            "doc_type": "web",          # Identifies source type for UI
            "fetch_status": source.status,
            "fetch_error": source.error,
        },
    }
```

Key metadata fields:
- `doc_type`: Used by SourceCard to select icon ("web", "paper", "library_doc", etc.)
- `source_url`: Clickable link
- `display_name`: Human-readable title
- Agent-specific fields: `fetch_status`, `fetch_error`, etc.

---

## Command Implementation Pattern

### Streaming Generator

```python
async def agent_stream(
    query: str,
    **params
) -> AsyncGenerator[AgentChunk, None]:
    """Streaming agent that yields chunks like RAG does."""

    # Phase 1: Agent-specific work
    yield AgentChunk(agent_progress={"phase": "searching", ...})
    results = await do_agent_work()
    yield AgentChunk(agent_progress={"phase": "complete", ...})

    # Phase 2: LLM generation (standard pattern)
    yield AgentChunk(status="generating")
    async for token in llm.astream_complete(prompt):
        yield AgentChunk(token=token)

    # Phase 3: Deliver sources
    yield AgentChunk(sources=converted_sources, is_complete=True)
```

### Command Execute Method

```python
async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
    full_response = ""
    sources_for_session = []

    async for chunk in agent_stream(...):
        if chunk.agent_progress:
            await websocket.send_json({"type": "agent_progress", **chunk.agent_progress})

        elif chunk.status:
            await websocket.send_json({"type": "status", "status": chunk.status})

        elif chunk.token:
            full_response += chunk.token
            await websocket.send_json({"type": "token", "content": chunk.token})

        elif chunk.sources is not None:
            sources_for_session = chunk.sources
            if chunk.sources:
                source_nodes = [convert_source(s) for s in chunk.sources]
                await websocket.send_json({"type": "sources", "data": source_nodes})

    # Check if title needed (first message)
    is_first = len(session.get("messages", [])) <= 1

    await websocket.send_json({
        "type": "done",
        "content": full_response,
        "confidence_level": "agent_name",
        "title_pending": is_first,
        "sources": [convert_source(s) for s in sources_for_session] if sources_for_session else None,
    })
```

---

## Frontend Integration

### Adding New Agent Types

1. **SourceCard.tsx** - Add icon for new `doc_type`:
```typescript
case "arxiv":
  return <FileSearch className={iconClassName} />;
case "github":
  return <Github className={iconClassName} />;
```

2. **SourceCard.tsx** - Handle agent-specific metadata:
```typescript
// For agents with status tracking
const fetchStatus = metadata.fetch_status as string | undefined;
if (fetchStatus === "failed") {
  return { variant: "destructive", label: "Failed" };
}
```

3. **AgentProgress.tsx** - Already handles any `agent_progress` with phase icons.

### What NOT to Create

- ❌ New source list components (use `SourcesList`)
- ❌ New streaming indicator components (use `StreamingIndicator`)
- ❌ New message types for sources (use `sources` type)
- ❌ Separate state slices per agent (use existing `streamingSources`)

---

## Session Integration

### Title Generation

Commands should trigger title generation for first messages:

```python
# In chat.py command wrapper
if needs_title and full_response:
    title = await generate_smart_title_async(full_response)
    session_service.update_title(session_id, title)
    await websocket.send_json({"type": "title", "title": title})
```

### Source Persistence

Sources are saved with the assistant message:

```python
assistant_message = {
    "role": "assistant",
    "content": full_response,
}
if captured_sources:
    assistant_message["sources"] = captured_sources

session_service.add_message(session_id, assistant_message)
```

---

## Testing Pattern

### Mock Streaming Generator

```python
def make_mock_stream(tokens=None, sources=None):
    """Create mock async generator for testing."""
    async def mock_generator(**kwargs):
        yield AgentChunk(agent_progress={"phase": "searching"})
        yield AgentChunk(status="generating")
        for token in (tokens or ["Test"]):
            yield AgentChunk(token=token)
        yield AgentChunk(sources=sources or [], is_complete=True)
    return mock_generator

# Usage in tests
with patch("module.agent_stream", side_effect=make_mock_stream()):
    await command.execute(...)
```

### Key Test Cases

1. `test_sends_token_messages` - Verify streaming works
2. `test_sends_sources_as_source_nodes` - Verify format conversion
3. `test_sends_agent_progress` - Verify phase updates
4. `test_sends_status_generating` - Verify transition to LLM
5. `test_title_pending_for_first_message` - Verify title trigger
6. `test_handles_errors` - Verify graceful error handling

---

## Checklist for New Agents

When implementing a new agent:

- [ ] Create `AgentChunk` dataclass (or reuse `WebSearchChunk` pattern)
- [ ] Implement `agent_stream()` async generator
- [ ] Convert sources to `SourceNode` format with appropriate metadata
- [ ] Add `doc_type` case to `SourceCard.tsx` for icon
- [ ] Handle agent-specific metadata in `SourceCard.tsx` badge logic
- [ ] Set `title_pending` in done message for first message
- [ ] Include sources in done message for session saving
- [ ] Write tests using mock streaming generator pattern
- [ ] DO NOT create new source list components
- [ ] DO NOT create new message types when existing ones work

---

## Example: Future ArXiv Agent

```python
# sources would have:
{
    "text": paper.abstract,
    "score": relevance_score,
    "metadata": {
        "source_url": f"https://arxiv.org/abs/{paper.id}",
        "display_name": paper.title,
        "doc_type": "arxiv",
        "authors": paper.authors,
        "published": paper.published_date,
    }
}

# SourceCard.tsx would add:
case "arxiv":
  return <GraduationCap className={iconClassName} />;
```

The existing `SourcesList` handles display, expansion, and metrics automatically.

---

## Summary

The key insight is that **agents are just specialized data gathering + LLM prompting**. Once data is gathered, the flow should match exactly what RAG does:

1. Gather context (agent-specific, use `agent_progress`)
2. Generate response (shared, use `status` + `token`)
3. Deliver sources (shared, use `sources` with converted format)
4. Complete (shared, use `done` with `title_pending`)

This ensures consistent UX, code reuse, and maintainability.
