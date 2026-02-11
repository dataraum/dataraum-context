# Next Phase Evaluation: API, MCP, Instrumentation, and Code Quality

This document evaluates five areas for the next development phase:
1. API + UI with Python/TypeScript monorepo structure
2. MCP Server for Claude Desktop
3. Instrumentation and logging improvements
4. Code cleanup opportunities
5. Storage/core module consolidation

---

## 1. API + UI Implementation

### Current State
- **FastAPI** is in dependencies but not implemented
- **No API routes** exist in the codebase
- **External web-app** has a mature TypeScript implementation with:
  - SvelteKit frontend
  - Hono API on Cloudflare Workers
  - Proven agent patterns for data/business analysis

### Recommended Monorepo Structure

```
dataraum-context/
├── pyproject.toml              # Python package config
├── package.json                # Root workspace config (pnpm/bun)
├── turbo.json                  # Turborepo for monorepo orchestration
│
├── src/                        # Python source (unchanged)
│   └── dataraum_context/
│       ├── api/                # NEW: FastAPI implementation
│       │   ├── __init__.py
│       │   ├── app.py          # FastAPI application
│       │   ├── routes/         # Route modules
│       │   │   ├── sources.py
│       │   │   ├── tables.py
│       │   │   ├── analysis.py
│       │   │   ├── pipeline.py
│       │   │   └── query.py
│       │   ├── schemas/        # Pydantic API schemas
│       │   └── middleware/     # Auth, CORS, logging
│       └── ... (existing modules)
│
├── packages/                   # NEW: TypeScript packages
│   ├── ui/                     # Frontend application
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── routes/         # SvelteKit pages
│   │   │   ├── lib/            # Components, utilities
│   │   │   └── app.html
│   │   └── vite.config.ts
│   │
│   ├── api-client/             # Generated TypeScript client
│   │   ├── package.json
│   │   └── src/
│   │       └── client.ts       # Generated from OpenAPI spec
│   │
│   └── shared/                 # Shared types/utilities
│       ├── package.json
│       └── src/
│           └── types.ts
│
├── tests/                      # Python tests (unchanged)
├── e2e/                        # NEW: Playwright E2E tests
└── docs/
```

### API Design Principles

Based on the pipeline phases and existing data model:

```python
# api/routes/sources.py
@router.post("/sources")
async def create_source(source: SourceCreate) -> SourceResponse:
    """Import a new data source (CSV directory)."""

@router.get("/sources/{source_id}")
async def get_source(source_id: str) -> SourceResponse:
    """Get source with all tables and metadata."""

# api/routes/pipeline.py
@router.post("/sources/{source_id}/run")
async def run_pipeline(
    source_id: str,
    target_phase: str | None = None,
) -> PipelineRunResponse:
    """Execute pipeline on source."""

@router.get("/sources/{source_id}/status")
async def get_pipeline_status(source_id: str) -> PipelineStatusResponse:
    """Get current pipeline status and phase results."""

# api/routes/analysis.py
@router.get("/tables/{table_id}/correlations")
async def get_correlations(table_id: str) -> CorrelationsResponse:
    """Get numeric and categorical correlations."""

@router.get("/tables/{table_id}/relationships")
async def get_relationships(table_id: str) -> RelationshipsResponse:
    """Get detected join relationships."""

@router.get("/tables/{table_id}/quality")
async def get_quality_summary(table_id: str) -> QualitySummaryResponse:
    """Get quality assessment and issues."""

# api/routes/query.py
@router.post("/query")
async def execute_query(query: QueryRequest) -> QueryResponse:
    """Execute SQL against DuckDB."""
```

### TypeScript Tech Stack Recommendation

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Frontend | **SvelteKit 5** | Proven in web-app prototype, excellent DX |
| UI Components | **Tailwind + DaisyUI** | Rapid prototyping, consistent design |
| SQL Editor | **CodeMirror** | Already used in prototype |
| Charts | **Chart.js** or **Observable Plot** | Lightweight, flexible |
| API Client | **openapi-typescript** | Generated from FastAPI OpenAPI spec |
| Build | **Turborepo + Bun** | Fast monorepo builds |

### Implementation Priority

1. **Phase 1**: FastAPI core routes (sources, pipeline, basic queries)
2. **Phase 2**: OpenAPI spec generation, TypeScript client generation
3. **Phase 3**: SvelteKit scaffold with basic views
4. **Phase 4**: Interactive UI components (SQL editor, charts)

---

## 2. MCP Server for Claude Desktop

### Current State
- `mcp>=0.1.0` is in dependencies
- **No MCP implementation exists**
- CLAUDE.md mentions 4 planned tools: `get_context`, `query`, `get_metrics`, `annotate`

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Desktop                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  User: "What are the data quality issues in the     │   │
│  │         customer table?"                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Claude calls: get_context(table="customer")        │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┼────────────────────────────────┘
                             │ stdio
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP Server (Python)                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ get_context │  │    query    │  │    get_metrics      │ │
│  │             │  │             │  │                     │ │
│  │ Returns:    │  │ Executes    │  │ Returns available   │ │
│  │ - Schema    │  │ DuckDB SQL  │  │ metrics for         │ │
│  │ - Quality   │  │ with safety │  │ selected ontology   │ │
│  │ - Relations │  │ checks      │  │                     │ │
│  │ - Entropy   │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ConnectionManager                       │   │
│  │   ┌─────────────┐        ┌─────────────────────┐    │   │
│  │   │ metadata.db │        │    data.duckdb      │    │   │
│  │   │  (SQLite)   │        │                     │    │   │
│  │   └─────────────┘        └─────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Implementation

```python
# src/dataraum_context/mcp/server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import Tool, Resource

class DataRaumMCPServer:
    """MCP Server for Claude Desktop integration."""

    def __init__(self, output_dir: Path):
        self.server = Server("dataraum-context")
        self.manager = ConnectionManager(
            ConnectionConfig.for_directory(output_dir)
        )
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        @self.server.tool()
        async def get_context(
            table: str | None = None,
            include_quality: bool = True,
            include_relationships: bool = True,
        ) -> str:
            """Get comprehensive context for data analysis.

            Returns schema, quality issues, relationships, and entropy
            information for the specified table or all tables.
            """
            # Build ExecutionContext with rich_context
            ctx = ExecutionContext.with_rich_context(
                session=self.manager.get_session(),
                duckdb_conn=self.manager.get_duckdb_connection(),
            )
            return ctx.to_markdown(table_filter=table)

        @self.server.tool()
        async def query(sql: str) -> str:
            """Execute a read-only SQL query against the data.

            Only SELECT statements are allowed. Returns results
            as formatted markdown table.
            """
            # Validate read-only
            if not sql.strip().upper().startswith("SELECT"):
                raise ValueError("Only SELECT queries allowed")

            result = self.manager.duckdb_conn.execute(sql).fetchdf()
            return result.to_markdown()

        @self.server.tool()
        async def get_metrics(ontology: str = "default") -> str:
            """Get available metrics for the selected ontology.

            Returns metric definitions with formulas that can be
            computed from the available data.
            """
            # Load ontology and filter applicable metrics
            ...

        @self.server.tool()
        async def annotate(
            table: str,
            column: str,
            annotation_type: str,
            value: str,
        ) -> str:
            """Add semantic annotation to a column.

            Allows human-in-loop correction of detected semantics.
            annotation_type: 'role', 'entity_type', 'business_term'
            """
            # Update SemanticAnnotation in database
            ...

    def _register_resources(self):
        @self.server.resource("schema://tables")
        async def list_tables() -> str:
            """List all available tables with row counts."""
            ...

        @self.server.resource("schema://tables/{table_name}")
        async def get_table_schema(table_name: str) -> str:
            """Get detailed schema for a specific table."""
            ...

# Entry point for Claude Desktop
def main():
    import sys
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./pipeline_output")
    server = DataRaumMCPServer(output_dir)
    asyncio.run(stdio_server(server.server))
```

### Claude Desktop Configuration

```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "dataraum": {
      "command": "uv",
      "args": ["run", "dataraum-mcp", "/path/to/pipeline_output"],
      "env": {}
    }
  }
}
```

### MCP Tools Summary

| Tool | Purpose | Inputs | Output |
|------|---------|--------|--------|
| `get_context` | Primary context retrieval | table?, include_quality?, include_relationships? | Markdown with schema, quality, relationships |
| `query` | Execute SQL | sql (SELECT only) | Markdown table |
| `get_metrics` | Available metrics | ontology | Metric definitions with formulas |
| `annotate` | Human-in-loop update | table, column, type, value | Confirmation |

---

## 3. Instrumentation and Logging

### Current State
- Uses `logging.getLogger(__name__)` across 20+ modules
- CLI uses `RichHandler` for pretty console output
- `structlog` is in dependencies but **unused**
- No metrics, tracing, or structured logging
- Config has `log_format: "json"` but not implemented

### Recommended Improvements

#### 3.1 Structured Logging with structlog

```python
# src/dataraum_context/core/logging.py
import structlog
from structlog.typing import FilteringBoundLogger

def configure_logging(
    log_level: str = "INFO",
    log_format: str = "console",  # "console" or "json"
    show_timestamps: bool = True,
) -> FilteringBoundLogger:
    """Configure structured logging for the application."""

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()

# Usage in phases:
logger = structlog.get_logger()

async def _run(self, ...):
    logger.info("phase_started", phase=self.name, source_id=source_id)

    with logger.contextvars(table=table.table_name):
        logger.debug("processing_table", row_count=table.row_count)
        # ... work ...

    logger.info("phase_completed",
                phase=self.name,
                duration_ms=duration * 1000,
                tables_processed=count)
```

#### 3.2 Phase Timing Metrics

```python
# src/dataraum_context/pipeline/metrics.py
from dataclasses import dataclass, field
from typing import Dict, List
import time

@dataclass
class PhaseMetrics:
    """Metrics collected during phase execution."""
    phase_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Counters
    tables_processed: int = 0
    rows_processed: int = 0
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    db_queries: int = 0

    # Sub-timings
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def time_operation(self, name: str):
        """Context manager for timing sub-operations."""
        class Timer:
            def __init__(self, metrics, op_name):
                self.metrics = metrics
                self.op_name = op_name
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.metrics.timings[self.op_name] = time.time() - self.start

        return Timer(self, name)


@dataclass
class PipelineMetrics:
    """Aggregate metrics for entire pipeline run."""
    source_id: str
    phases: List[PhaseMetrics] = field(default_factory=list)

    def summary(self) -> dict:
        """Generate metrics summary for CLI output."""
        return {
            "total_duration_s": sum(p.duration_seconds for p in self.phases),
            "phases": {
                p.phase_name: {
                    "duration_s": p.duration_seconds,
                    "tables": p.tables_processed,
                    "rows": p.rows_processed,
                    "llm_calls": p.llm_calls,
                    "bottleneck": max(p.timings.items(), key=lambda x: x[1])[0]
                                  if p.timings else None,
                }
                for p in self.phases
            },
            "totals": {
                "llm_calls": sum(p.llm_calls for p in self.phases),
                "llm_tokens": sum(p.llm_tokens_in + p.llm_tokens_out for p in self.phases),
                "rows_processed": sum(p.rows_processed for p in self.phases),
            }
        }
```

#### 3.3 Enhanced CLI Output

```python
# Enhanced verbose output in cli.py
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

def display_metrics_summary(metrics: PipelineMetrics):
    """Display formatted metrics summary."""
    console = Console()

    table = Table(title="Phase Execution Summary")
    table.add_column("Phase", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Tables", justify="right")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Bottleneck", style="yellow")

    for phase in metrics.phases:
        summary = metrics.summary()["phases"][phase.phase_name]
        table.add_row(
            phase.phase_name,
            f"{summary['duration_s']:.2f}s",
            str(summary['tables']),
            str(summary['llm_calls']),
            summary['bottleneck'] or "-",
        )

    console.print(table)

    # Bottleneck analysis
    slowest = max(metrics.phases, key=lambda p: p.duration_seconds)
    console.print(f"\n[yellow]Slowest phase:[/yellow] {slowest.phase_name} "
                  f"({slowest.duration_seconds:.2f}s)")
```

#### 3.4 Verbose Flags Enhancement

```
dataraum run /data --verbose        # Current: basic logging
dataraum run /data --verbose -v     # Detailed: per-table progress
dataraum run /data --verbose -vv    # Debug: SQL queries, LLM prompts
dataraum run /data --metrics        # Output metrics JSON to file
dataraum run /data --profile        # Enable Python profiler
```

---

## 4. Code Cleanup

### 4.1 Findings Summary

| Category | Finding | Action |
|----------|---------|--------|
| Dead code | `_profile_column_stats` (line 373) | **Remove** - Never called, superseded by parallel version |
| Private functions | 29+ other `_` prefixed functions | **Keep** - All legitimate helpers/parallel processors |
| TODOs | 4 found | **Keep** - All Phase 2.5+ or valid refactoring notes |
| Test shadowing | None found | N/A |
| Duplicate code | None found | N/A |

### 4.2 Specific Items Reviewed

**`_profile_column_stats` in statistics/profiler.py (DEAD CODE):**

There are TWO functions with similar names:
1. `_profile_column_stats_parallel` (line 49) - **Active**, used by ThreadPoolExecutor
2. `_profile_column_stats` (line 373) - **DEAD CODE**, never called anywhere

```python
# Line 373 - DEAD CODE (defined but never called)
def _profile_column_stats(
    table: Table,
    column: Column,
    duckdb_conn: duckdb.DuckDBPyConnection,
    profiled_at: datetime,
    settings: Settings,
) -> Result[ColumnProfile]:
    """Profile a single column for all row-based statistics."""
    # ~100 lines of implementation that is NEVER USED
```

The parallel version (`_profile_column_stats_parallel`) is the only one called:
```python
# Line 314 - This is what's actually used
pool.submit(
    _profile_column_stats_parallel,  # <-- The parallel version
    duckdb_conn,
    table.table_name,
    ...
)
```

**Action:** Remove `_profile_column_stats` (lines 373-530) - it's dead code.

**Other `_` prefixed functions:** All legitimate:
1. Parallel processing workers (ThreadPoolExecutor pattern)
2. Internal helpers that shouldn't be in public API
3. Following Python conventions correctly

### 4.3 Maintenance Coupling Point

**`core/connections.py:_import_all_models()`** must stay in sync with 16 db_models.py files:

```python
def _import_all_models() -> None:
    """Import all SQLAlchemy models to ensure they're registered."""
    from dataraum_context.analysis.correlation import db_models as _  # noqa: F401
    from dataraum_context.analysis.cycles import db_models as _  # noqa: F401
    # ... 14 more imports
```

**Recommendation:** Add auto-discovery or documentation:

```python
# Option A: Auto-discovery (more complex but less maintenance)
def _import_all_models() -> None:
    """Auto-discover and import all db_models.py files."""
    import importlib
    import pkgutil

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=["src/dataraum_context"],
        prefix="dataraum_context.",
    ):
        if modname.endswith(".db_models"):
            importlib.import_module(modname)

# Option B: Documentation (simpler, keep current approach)
# Add comment documenting all 16 modules that must be imported
```

### 4.4 TODOs to Address

| Location | TODO | Priority |
|----------|------|----------|
| `llm/privacy.py` | Implement SDV synthetic data | Low (optional feature) |
| `temporal/patterns.py` | Revisit change point sampling | Medium |
| `temporal/processor.py` | Revisit sampling strategy | Medium |
| `entropy/semantic/business_meaning.py` | LLM semantic quality eval | Medium (Phase 2.5) |

---

## 5. Storage/Core Module Consolidation

### 5.1 Current Structure

```
core/
├── config.py          (139 lines) - Pydantic Settings
├── connections.py     (507 lines) - Thread-safe connection management
└── models/
    ├── base.py        (171 lines) - Result[T], enums, refs
    └── __init__.py    (32 lines)  - Re-exports with docs

storage/
├── base.py            (108 lines) - SQLAlchemy Base, init/reset
├── models.py          (157 lines) - Source, Table, Column
└── __init__.py        (6 lines)   - Re-exports
```

### 5.2 Analysis

**Reasons to merge:**
- `storage/` only contains 271 lines of code
- Both deal with "foundational" concerns
- `core/connections.py` already imports from `storage/`

**Reasons to keep separate:**
- Clean separation: `core/` = runtime, `storage/` = persistence
- `storage/models.py` contains domain entities (Source, Table, Column)
- `core/models/base.py` explicitly says "NO DOMAIN RE-EXPORTS"
- Adding domain models to `core/` violates this principle

### 5.3 Recommendation: Keep Separate, Improve Organization

Instead of merging, improve the structure:

```
core/
├── config.py              # Settings
├── connections.py         # Connection management
├── types.py               # Move from models/base.py: Result, enums
└── models.py              # Move from models/base.py: ColumnRef, TableRef

storage/
├── base.py                # SQLAlchemy Base
├── entities.py            # Rename from models.py: Source, Table, Column
├── registry.py            # NEW: Model discovery for _import_all_models()
└── migrations/            # Future: Alembic migrations
```

**Benefits:**
- `types.py` clearly indicates non-SQLAlchemy types
- `entities.py` clearer than `models.py` (avoids Pydantic confusion)
- `registry.py` solves the maintenance coupling problem
- Keeps architectural separation intact

### 5.4 Tooling for Large-Scale Refactoring

For safe file moves and renames:

```bash
# Use rope for Python refactoring
pip install rope

# Or use IDE refactoring (PyCharm/VS Code)
# - "Move" refactoring updates all imports
# - "Rename" refactoring updates all references

# For verification after refactoring:
uv run pytest tests/ -v           # All tests pass
uv run pyright                    # Type checking passes
uv run ruff check .               # Linting passes
```

**Recommended approach:**
1. Use IDE "Move Symbol" feature
2. Run full test suite after each move
3. Commit after each successful refactoring
4. Use git blame to verify import updates

---

## Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)
- [ ] Implement structlog integration
- [ ] Add phase metrics collection
- [ ] Enhance CLI verbose output
- [ ] Create MCP server skeleton

### Phase 2: API Core (2-3 weeks)
- [ ] FastAPI application setup
- [ ] Core routes (sources, pipeline, query)
- [ ] OpenAPI spec generation
- [ ] TypeScript client generation

### Phase 3: MCP Completion (1 week)
- [ ] Implement all 4 MCP tools
- [ ] Resource providers for schema
- [ ] Claude Desktop configuration docs
- [ ] Integration testing

### Phase 4: UI Foundation (2-3 weeks)
- [ ] Monorepo setup (Turborepo + packages/)
- [ ] SvelteKit scaffold
- [ ] Basic views (sources, tables, quality)
- [ ] SQL editor integration

### Phase 5: Polish (1-2 weeks)
- [ ] Code cleanup (registry.py, entity renames)
- [ ] Documentation updates
- [ ] E2E testing setup
- [ ] Performance profiling integration

---

## Decision Summary

| Topic | Recommendation |
|-------|----------------|
| Monorepo structure | `packages/` for TypeScript, keep `src/` for Python |
| Frontend framework | SvelteKit 5 (proven in prototype) |
| MCP implementation | 4 tools as specified in CLAUDE.md |
| Logging | Migrate to structlog with JSON support |
| Metrics | Add PhaseMetrics with timing breakdowns |
| Code cleanup | No immediate cleanup needed, add registry.py |
| Storage/core merge | Keep separate, rename for clarity |
| Refactoring tooling | IDE refactoring + full test suite verification |
