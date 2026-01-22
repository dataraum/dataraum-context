# UI/API Consolidation Plan

## Overview

This plan consolidates the UI/API layer for dataraum-context, integrating:
- **Python backend** (`dataraum`) with FastAPI
- **TypeScript prototype patterns** (analytics-agents-ts)
- **Web visualizer patterns** (calculation-graphs/web_visualizer)
- **Entropy system** already built
- **MCP Server** (after UI learnings collected)

**Key decisions:**
- Package renamed to `dataraum` (completed)
- Build UI first, MCP Server last (learn from UI iteration)
- API grows organically as UI needs emerge
- Configuration Hub as separate phase after core UI

---

## Current Status (2026-01-22)

### Completed

- [x] **Restructure** - Moved to `packages/dataraum-api/src/dataraum/`
- [x] **FastAPI app skeleton** - `api/main.py` with lifespan management
- [x] **Core Pydantic schemas** - `api/schemas.py`
- [x] **Dependency injection** - `api/deps.py` using core ConnectionManager
- [x] **Source endpoints** - GET /sources, GET /sources/{id}, POST /sources
- [x] **Pipeline endpoints** - POST /sources/{id}/run, GET /sources/{id}/status
- [x] **SSE progress streaming** - GET /runs/{run_id}/stream (real-time updates)
- [x] **Table endpoints** - GET /tables, GET /tables/{id}, GET /columns/{id}
- [x] **Query endpoint** - POST /query (read-only SQL execution)
- [x] **Singleton execution** - Only one pipeline at a time
- [x] **Startup cleanup** - Marks interrupted runs on server start
- [x] **Free-threading support** - Python 3.14t with `-Xgil=0`
- [x] **API test suite** - 31 tests covering all implemented endpoints

### In Progress

- [ ] **Context endpoint** - GET /context/{source_id} (needs GraphExecutionContext integration)
- [ ] **Entropy endpoint** - GET /entropy/{source_id} (needs to_dashboard_dict() integration)
- [ ] **Graphs endpoint** - GET /graphs, POST /graphs/{id}/execute (partial)

### Not Started

- [ ] Frontend (Phase 3-5)
- [ ] Configuration Hub (Phase 6)
- [ ] MCP Server (Phase 7)

---

## Part A: Remaining API Work

### Context Endpoint

```python
# api/routers/context.py
@router.get("/context/{source_id}")
def get_context(source_id: str, session: SessionDep) -> ContextResponse:
    """Return full context document for LLM consumption."""
    # Uses GraphExecutionContext from graphs/context.py
    pass
```

### Entropy Endpoint

```python
# api/routers/entropy.py
@router.get("/entropy/{source_id}")
def get_entropy_dashboard(source_id: str, session: SessionDep) -> EntropyDashboardResponse:
    """Return entropy dashboard data."""
    # Uses EntropyContext.to_dashboard_dict() from entropy/models.py
    pass
```

### Graphs Endpoint

```python
# api/routers/graphs.py
@router.get("/graphs")
def list_graphs(source_id: str, session: SessionDep) -> GraphListResponse:
    """List available transformation graphs."""
    pass

@router.post("/graphs/{graph_id}/execute")
def execute_graph(graph_id: str, request: GraphExecuteRequest) -> GraphExecuteResponse:
    """Execute a graph and return results."""
    pass
```

---

## Part B: MCP Server (Phase 7 - after UI)

**Rationale for building last:**
- UI iteration validates API contracts first
- Same data structures serve both REST and MCP
- Easier to debug HTTP than stdio-based tools
- Learnings from UI inform better tool design

### 4 Tools (per CLAUDE.md)

| Tool | Purpose | Wraps API |
|------|---------|-----------|
| `get_context` | Primary context retrieval | `GET /api/v1/context/{source_id}` |
| `query` | Execute read-only SQL | `POST /api/v1/query` |
| `get_metrics` | Available metrics for ontology | `GET /api/v1/graphs` + metadata |
| `annotate` | Update semantic annotation | `PUT /api/v1/columns/{id}/annotation` |

### File Structure

```
src/dataraum/mcp/
├── __init__.py
├── server.py         # Main MCP server
├── tools/
│   ├── context.py    # Wraps context router
│   ├── query.py      # Wraps query router
│   ├── metrics.py    # Wraps graphs + ontology
│   └── annotate.py   # Wraps annotation endpoint
└── formatting.py     # Output formatters (markdown, json)
```

### Entry Point

```python
# mcp/server.py
from mcp.server import Server

class DataRaumMCPServer:
    def __init__(self, output_dir: Path):
        self.server = Server("dataraum")
        self.manager = ConnectionManager(ConnectionConfig.for_directory(output_dir))
        self._register_tools()

# pyproject.toml already has:
# dataraum-mcp = "dataraum.mcp.server:main"  # TODO: implement
```

---

## Part C: Frontend Architecture

### Technology Stack

**Frontend:**
- **React 19** + TypeScript
- **XY Flow** for graphs
- **TanStack** suite (Query, Router, Form, Table, Virtual)
- **Zod** - schema validation
- **Tailwind CSS** + shadcn/ui
- **Vite** for build

**Backend (completed):**
- **FastAPI** with Pydantic V2
- **SQLAlchemy 2.0** sync (works well with free-threading)
- **uvicorn** with `-Xgil=0`

### Component Structure

**From web_visualizer patterns:**
1. **Expand/Collapse** - Chevron toggle for calculation breakdown
2. **Drill-Down Modals** - Full-screen overlay with sortable tables
3. **View Modes** - Sidebar navigation between different layouts
4. **Node Types** - TableNode, ColumnNode, FieldNode, MetricNode

**New Components:**
- EntropyDashboard (from `to_dashboard_dict()`)
- PipelineMonitor (SSE for real-time progress)
- ContextViewer (hierarchical tree + graph)
- QualityView (issues list)

### Data Layer

```typescript
// TanStack Query hooks
const queryKeys = {
  sources: ['sources'],
  source: (id: string) => ['sources', id],
  tables: (sourceId: string) => ['tables', { sourceId }],
  context: (sourceId: string) => ['context', sourceId],
  entropy: (sourceId: string) => ['entropy', sourceId],
  pipelineStatus: (sourceId: string) => ['pipeline', sourceId, 'status'],
};

// SSE for pipeline progress
const usePipelineStream = (runId: string) => {
  // Connect to GET /api/v1/runs/{runId}/stream
  // Parse SSE events: start, phase_complete, phase_failed, complete, error
};
```

---

## Part D: Configuration Hub (Phase 6)

**Purpose:** Enable human-in-the-loop refinement of automated analysis.

### Configuration Areas

| Area | UI Component | Backend |
|------|--------------|---------|
| **Ontology Selection** | Dropdown selector | `config/ontologies/*.yaml` |
| **Semantic Overrides** | Column → concept mapping editor | `PUT /api/v1/columns/{id}/annotation` |
| **Entropy Thresholds** | Slider controls | `config/entropy/thresholds.yaml` |
| **Quality Rules** | Toggle switches + severity dropdowns | `config/rules/*.yaml` |

---

## Part E: Project Structure (Current)

```
dataraum-context/
├── packages/
│   └── dataraum-api/                # Python backend
│       ├── src/dataraum/
│       │   ├── api/                 # FastAPI (DONE)
│       │   │   ├── main.py
│       │   │   ├── schemas.py
│       │   │   ├── deps.py
│       │   │   ├── server.py
│       │   │   └── routers/
│       │   │       ├── sources.py   # DONE
│       │   │       ├── pipeline.py  # DONE
│       │   │       ├── tables.py    # DONE
│       │   │       ├── query.py     # DONE
│       │   │       ├── context.py   # TODO: integrate
│       │   │       ├── entropy.py   # TODO: integrate
│       │   │       └── graphs.py    # TODO: integrate
│       │   ├── mcp/                 # TODO: MCP Server
│       │   ├── pipeline/
│       │   ├── entropy/
│       │   ├── graphs/
│       │   └── ...
│       ├── tests/
│       │   └── api/                 # 31 tests (DONE)
│       └── pyproject.toml
│
├── packages/web/                    # TODO: Frontend
│
├── config/
├── docs/
│   └── plans/
│       └── ui-api-consolidation.md  # This file
└── README.md
```

---

## Implementation Phases (Revised)

### Phase 1: API Foundation ✅ COMPLETE
- [x] FastAPI app skeleton with dependency injection
- [x] Core Pydantic schemas
- [x] Source endpoints (CRUD)
- [x] Pipeline endpoints (trigger, status, SSE stream)
- [x] Table/Column endpoints
- [x] Query endpoint
- [x] Singleton execution + startup cleanup
- [x] Free-threading support
- [x] API test suite (31 tests)

### Phase 2: Metadata API (Next)
- [ ] Context endpoint integration (GraphExecutionContext)
- [ ] Entropy endpoint integration (to_dashboard_dict)
- [ ] Graphs endpoint completion
- [ ] Tests for remaining endpoints

### Phase 3: Frontend Foundation
- [ ] Vite + React 19 + TypeScript scaffold
- [ ] API client generation from OpenAPI + Zod schemas
- [ ] TanStack setup (Query, Router, Form, Table, Virtual)
- [ ] Tailwind + shadcn/ui components

### Phase 4: Core UI
- [ ] Layout (Sidebar + MainContent)
- [ ] Source selector
- [ ] Pipeline monitor with SSE
- [ ] Table browser (schema view)

### Phase 5: Advanced UI
- [ ] Entropy dashboard (heatmap, readiness indicators)
- [ ] Graph visualization (React Flow)
- [ ] Drill-down modals (from web_visualizer)
- [ ] Quality view

### Phase 6: Configuration Hub
- [ ] Ontology selector
- [ ] Semantic overrides editor
- [ ] Entropy threshold configuration
- [ ] Quality rule toggles

### Phase 7: MCP Server
- [ ] MCP server skeleton
- [ ] `get_context` tool
- [ ] `query` tool
- [ ] `get_metrics` tool
- [ ] `annotate` tool
- [ ] Claude Desktop configuration docs

### Phase 8: Polish & Integration
- [ ] End-to-end testing (Playwright)
- [ ] Error boundaries
- [ ] Export functionality (CSV/JSON)
- [ ] Documentation

---

## API Reference (Implemented)

### Sources
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/sources` | List sources with pagination |
| GET | `/api/v1/sources/{id}` | Get source details |
| POST | `/api/v1/sources` | Create new source |

### Pipeline
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/sources/{id}/run` | Trigger pipeline (returns run_id) |
| GET | `/api/v1/sources/{id}/status` | Get pipeline status |
| GET | `/api/v1/runs/{run_id}/stream` | SSE progress stream |

### Tables
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tables` | List tables (filter by source_id) |
| GET | `/api/v1/tables/{id}` | Get table with columns |
| GET | `/api/v1/columns/{id}` | Get column details |

### Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Execute read-only SQL |

---

## Running the API

```bash
# With free-threading (recommended)
uv run python -Xgil=0 -m dataraum.api.server

# Or via script (warns if GIL enabled)
uv run dataraum-api

# Environment variables
DATARAUM_OUTPUT_DIR=./pipeline_output  # Required
DATARAUM_API_HOST=127.0.0.1            # Default
DATARAUM_API_PORT=8000                 # Default
DATARAUM_API_RELOAD=false              # Set true for dev
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/dataraum/api/main.py` | FastAPI app factory with lifespan |
| `src/dataraum/api/server.py` | Entry point with free-threading |
| `src/dataraum/api/routers/pipeline.py` | SSE streaming, singleton execution |
| `src/dataraum/graphs/context.py` | GraphExecutionContext - for context API |
| `src/dataraum/entropy/models.py` | EntropyContext.to_dashboard_dict() |
| `src/dataraum/pipeline/orchestrator.py` | Pipeline execution with run_id |
| `tests/api/conftest.py` | Test fixtures (test_client, seeded_source, seeded_tables) |
