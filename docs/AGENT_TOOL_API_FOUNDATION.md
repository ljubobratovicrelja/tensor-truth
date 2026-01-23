# Agent & Tool API Foundation

This document describes the Agent & Tool API foundation implemented for TensorTruth, and outlines what remains to be done.

## Status: Framework Complete, Agents Not Yet Registered

The **framework** is complete. The services, API endpoints, schemas, and frontend types are all in place. However, **no concrete agents are registered** yet - the `/api/agents` endpoint returns an empty list until agents are registered via `AgentService.register_agent()`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Frontend                             │
├─────────────────────────────────────────────────────────────────┤
│  GET /api/tools, GET /api/agents, WebSocket progress events     │
├─────────────────────────────────────────────────────────────────┤
│                      FastAPI Routes                             │
│  - List tools/agents                                            │
│  - Execute commands (/browse, /toolname)                        │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                              │
│  ToolService              AgentService                          │
│  - Manages ToolSpecs      - Creates LlamaIndex agents           │
│  - Returns FunctionTools  - Loads agent configs (YAML/built-in) │
│  - Executes single tools  - Manages execution callbacks         │
├─────────────────────────────────────────────────────────────────┤
│                   LlamaIndex (THE Foundation)                   │
│  FunctionTool    ToolSpec       FunctionAgent    ReActAgent     │
│  McpToolSpec     QueryEngineTool ChatMemoryBuffer               │
└─────────────────────────────────────────────────────────────────┘
```

### Core Principle

**DO NOT** create custom base classes for Tools or Agents. LlamaIndex provides:
- `FunctionTool` - the tool interface
- `ToolSpec` - bundles of related tools (e.g., `McpToolSpec` for MCP)
- `FunctionAgent` - agent using LLM function calling
- `ReActAgent` - agent using ReAct prompting

Our services **configure and manage** LlamaIndex classes, not wrap or replace them.

---

## What Was Implemented

### 1. ToolService (`/src/tensortruth/services/tool_service.py`)

Manages tool sources (MCP servers) and provides `FunctionTool` lists to agents.

```python
class ToolService:
    async def load_tools() -> None          # Load from MCP registry at startup
    @property tools -> List[FunctionTool]   # All loaded tools
    def get_tools_by_names(names) -> List   # Filter for agent construction
    def list_tools() -> List[Dict]          # Metadata for API
    async def execute_tool(name, params)    # Direct tool execution
```

**Key points:**
- Uses existing `MCPServerRegistry` from `/src/tensortruth/agents/server_registry.py`
- Loaded at startup in `main.py` lifespan handler
- Singleton via `get_tool_service()` in deps.py

### 2. AgentService (`/src/tensortruth/services/agent_service.py`)

Creates and executes LlamaIndex agents from configuration.

```python
@dataclass
class AgentCallbacks:
    on_progress: Optional[Callable[[str], None]]      # Status messages
    on_tool_call: Optional[Callable[[str, Dict], None]]  # Tool invocations
    on_token: Optional[Callable[[str], None]]         # Streaming tokens

class AgentService:
    def register_agent(config: AgentConfig) -> None   # Add agent config
    def list_agents() -> List[Dict]                   # Metadata for API
    async def run(agent_name, goal, callbacks, session_params) -> AgentResult
```

**Key points:**
- `_load_builtin_agents()` is a no-op - agents registered in subsequent phase
- Creates `FunctionAgent` or `ReActAgent` based on `AgentConfig.agent_type`
- Wraps tools with callbacks for progress tracking

### 3. AgentConfig (`/src/tensortruth/agents/config.py`)

Configuration dataclass for defining agents:

```python
@dataclass
class AgentConfig:
    name: str                          # Unique ID (e.g., "browse")
    description: str                   # Human-readable description
    tools: List[str]                   # Required tool names
    system_prompt: str                 # Agent behavior prompt
    agent_type: Literal["function", "react"] = "function"
    model: Optional[str] = None        # Override session model
    max_iterations: int = 10
```

### 4. API Routes (`/src/tensortruth/api/routes/tools.py`)

```
GET /api/tools   -> {"tools": [...]}   # List MCP tools
GET /api/agents  -> {"agents": [...]}  # List registered agents (empty for now)
```

### 5. WebSocket Progress Types (`/src/tensortruth/api/schemas/chat.py`)

```python
class StreamToolProgress(BaseModel):
    type: Literal["tool_progress"] = "tool_progress"
    tool: str
    action: Literal["calling", "completed", "failed"]
    params: Dict[str, Any] = {}

class StreamAgentProgress(BaseModel):
    type: Literal["agent_progress"] = "agent_progress"
    agent: str
    status: str  # "starting", "searching", "synthesizing", "complete"
```

### 6. Frontend Types (`/frontend/src/api/types.ts`)

```typescript
export interface StreamToolProgress {
  type: "tool_progress";
  tool: string;
  action: "calling" | "completed" | "failed";
  params: Record<string, unknown>;
}

export interface StreamAgentProgress {
  type: "agent_progress";
  agent: string;
  status: string;
}
```

### 7. Frontend State (`/frontend/src/stores/chatStore.ts`)

Added to the Zustand store:
```typescript
toolProgress: StreamToolProgress | null;
agentProgress: StreamAgentProgress | null;
setToolProgress: (progress: StreamToolProgress | null) => void;
setAgentProgress: (progress: StreamAgentProgress | null) => void;
```

### 8. WebSocket Handler (`/frontend/src/hooks/useWebSocket.ts`)

Added cases for `tool_progress` and `agent_progress` message types.

---

## Test Coverage

- `/tests/unit/test_tool_service.py` - 17 tests
- `/tests/unit/test_agent_service.py` - 22 tests

All tests pass. Run with:
```bash
source venv/bin/activate
python -m pytest tests/unit/test_tool_service.py tests/unit/test_agent_service.py -v
```

---

## What Remains To Be Done

### Phase 1: Register Built-in Agents

The framework is ready, but no agents are registered. Implement `_load_builtin_agents()` in `AgentService`:

```python
def _load_builtin_agents(self) -> None:
    """Register built-in agent configurations."""
    # Browse agent
    self.register_agent(AgentConfig(
        name="browse",
        description="Research topics on the web",
        tools=["search_web", "fetch_page"],
        system_prompt=BROWSE_AGENT_PROMPT,  # Define this
        agent_type="function",
        max_iterations=15,
    ))

    # Add more agents as needed
```

**Suggested agents:**
1. `browse` - Web research with search + fetch
2. `research` - Deep research with multiple sources
3. `summarize` - Synthesize information from fetched pages

### Phase 2: Wire Agent Execution to Chat Flow

Currently, agents are not invoked from the chat WebSocket. To complete the integration:

1. **In `/src/tensortruth/api/routes/chat.py`**:
   - Detect agent commands (e.g., `/browse query`)
   - Call `agent_service.run()` with appropriate callbacks
   - Stream `tool_progress` and `agent_progress` events via WebSocket

2. **Create agent callbacks that emit WebSocket messages**:
```python
callbacks = AgentCallbacks(
    on_progress=lambda msg: await websocket.send_json(
        StreamAgentProgress(agent=agent_name, status=msg).model_dump()
    ),
    on_tool_call=lambda tool, params: await websocket.send_json(
        StreamToolProgress(tool=tool, action="calling", params=params).model_dump()
    ),
)
```

### Phase 3: Two-Model Architecture (Optional Enhancement)

The existing `MCPBrowseAgent` in `/src/tensortruth/agents/mcp_agent.py` uses a two-model pattern:
- Reasoning model for agent execution
- Synthesis model for final answer quality

To add this to `AgentService`:

1. Add `synthesis_model` field to `AgentConfig`
2. After `agent.run()`, optionally synthesize the response with a larger model
3. This improves answer quality for web research tasks

### Phase 4: Natural Language Agent Routing

Enable natural language triggers (not just `/command` syntax):

1. **IntentService integration**: Already exists at `/src/tensortruth/services/intent_service.py`
2. **In chat flow**: If `intent.intent == "browse"`, route to browse agent
3. **Config flag**: `agent.enable_natural_language_agents` in config

### Phase 5: User-Defined Agents (Future)

Allow users to define agents via YAML:

```yaml
# ~/.tensortruth/agents/my_agent.yaml
name: my_agent
description: My custom agent
tools:
  - search_web
  - fetch_page
system_prompt: |
  You are a specialized research agent...
agent_type: function
max_iterations: 10
```

Load in `_load_builtin_agents()`:
```python
user_agents_dir = Path.home() / ".tensortruth" / "agents"
for yaml_file in user_agents_dir.glob("*.yaml"):
    config = AgentConfig(**yaml.safe_load(yaml_file.read_text()))
    self.register_agent(config)
```

### Phase 6: User-Defined Tools (Future)

Allow users to add custom tools:
- Load Python modules from `~/.tensortruth/tools/*.py`
- Each module exports a `ToolSpec` or `FunctionTool`
- Register in `ToolService.load_tools()`

---

## File Reference

### New Files Created
| File | Purpose |
|------|---------|
| `/src/tensortruth/services/tool_service.py` | ToolService class |
| `/src/tensortruth/services/agent_service.py` | AgentService + AgentCallbacks |
| `/src/tensortruth/api/routes/tools.py` | API endpoints |
| `/tests/unit/test_tool_service.py` | ToolService tests |
| `/tests/unit/test_agent_service.py` | AgentService tests |

### Modified Files
| File | Changes |
|------|---------|
| `/src/tensortruth/agents/config.py` | Added `AgentConfig` dataclass |
| `/src/tensortruth/services/__init__.py` | Added exports |
| `/src/tensortruth/api/deps.py` | Added `get_tool_service()`, `get_agent_service()` |
| `/src/tensortruth/api/main.py` | Added tool loading, router |
| `/src/tensortruth/api/schemas/chat.py` | Added progress schemas |
| `/frontend/src/api/types.ts` | Added TS interfaces |
| `/frontend/src/stores/chatStore.ts` | Added progress state |
| `/frontend/src/hooks/useWebSocket.ts` | Added message handlers |

### Untouched (Backward Compat)
| File | Status |
|------|--------|
| `/src/tensortruth/agents/mcp_agent.py` | Stays for Streamlit compat |
| `/src/tensortruth/app_utils/commands.py` | Streamlit code, deprecated |

---

## Verification Commands

```bash
# Run all new tests
source venv/bin/activate
python -m pytest tests/unit/test_tool_service.py tests/unit/test_agent_service.py -v

# Run linting
./scripts/lint.sh

# Verify API endpoints (requires server running)
curl localhost:8000/api/tools
curl localhost:8000/api/agents  # Returns empty list until agents registered
```

---

## Next Steps for Continuing Agent

1. **Read this document** to understand the architecture
2. **Implement `_load_builtin_agents()`** in `AgentService` with at least a `browse` agent
3. **Wire agent execution** into the chat WebSocket flow
4. **Test end-to-end**: `/browse latest AI news` should execute the agent and stream progress
5. **Update this document** with any changes to the plan

---

## Sources

- [LlamaIndex Agents](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/)
- [LlamaIndex Tools](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools/)
- [llama-index-tools-mcp](https://pypi.org/project/llama-index-tools-mcp/)
