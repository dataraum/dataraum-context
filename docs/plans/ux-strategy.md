# UX Strategy: CLI, TUI, Web, MCP

## Overview

This document captures the UX strategy for dataraum's multiple interfaces. The goal is to provide a consistent experience across different user personas while leveraging each interface's strengths.

**Decision Date:** 2026-02-04

---

## User Personas & Primary Interfaces

| Persona | Primary Interface | Use Case |
|---------|-------------------|----------|
| **Data Engineer** | CLI + TUI | Pipeline ops, debugging, config iteration |
| **Data Analyst** | Web UI + MCP | Explore data, ask questions, fix issues |
| **AI Agent** | MCP | Automated analysis, query generation |
| **Business User** | Web UI | Dashboards, quality monitoring |
| **Jupyter User** | Python API | Interactive exploration, notebooks |

---

## Interface Characteristics

```
┌─────────────────────────────────────────────────────────────────┐
│                     Interface Trade-offs                         │
├─────────────┬─────────────┬─────────────┬──────────────┬────────┤
│ Dimension   │    CLI      │    TUI      │   Web UI     │  MCP   │
├─────────────┼─────────────┼─────────────┼──────────────┼────────┤
│ Automation  │ ★★★★★       │ ★★☆☆☆       │ ★☆☆☆☆        │ ★★★★★  │
│ Exploration │ ★★☆☆☆       │ ★★★★☆       │ ★★★★★        │ ★★★☆☆  │
│ Visualization│ ★☆☆☆☆      │ ★★☆☆☆       │ ★★★★★        │ ☆☆☆☆☆  │
│ Fix Actions │ ★★★☆☆       │ ★★★☆☆       │ ★★★★★        │ ★★★★☆  │
│ No Server   │ ★★★★★       │ ★★★★★       │ ☆☆☆☆☆        │ ★★★★★  │
│ Remote/SSH  │ ★★★★★       │ ★★★★★       │ ☆☆☆☆☆        │ ★★★★★  │
│ Large Data  │ ★★★☆☆       │ ★★★★☆       │ ★★★★★        │ ★★★☆☆  │
└─────────────┴─────────────┴─────────────┴──────────────┴────────┘
```

---

## Key Decisions

### 1. CLI vs TUI Balance

**Decision:** CLI shows summaries by default, `--tui` flag for interactive mode.

```bash
# CLI: Summary output (default)
dataraum status ./output              # Rich tables
dataraum entropy ./output             # Summary table
dataraum contracts ./output           # Traffic light table
dataraum query ./output "revenue?"    # Answer + SQL

# CLI: Machine-readable
dataraum status ./output --json       # For scripts/CI

# TUI: Interactive drill-down
dataraum status ./output --tui        # → Home screen
dataraum entropy ./output --tui       # → Entropy dashboard
dataraum contracts ./output --tui     # → Contracts screen
dataraum query ./output --tui         # → Query screen

# Shortcut: Launch TUI directly
dataraum tui ./output                 # Home screen
```

### 2. Package Structure

**Decision:** Single package with optional extras (Option A).

```toml
[project]
dependencies = [
    # Core - always installed
    "duckdb", "pyarrow", "sqlalchemy", "pydantic", "pyyaml",
    "anthropic",  # LLM
]

[project.optional-dependencies]
cli = ["typer", "rich", "textual"]      # CLI + TUI
api = ["fastapi", "uvicorn", "jinja2"]  # Web server
mcp = ["mcp"]                           # MCP server
all = ["dataraum[cli,api,mcp]"]
```

**Usage:**
```bash
pip install dataraum           # Core only - for Jupyter/Python API
pip install dataraum[cli]      # + CLI/TUI
pip install dataraum[api]      # + Web server
pip install dataraum[all]      # Everything
```

**Rationale:**
- Single version number, simple OSS distribution
- Jupyter users get lightweight install
- CLI deps only loaded when needed (lazy imports)
- CI can use path filters to optimize

### 3. Session Synchronization

**Decision:** Defer for now. Event-based approach can be added later.

The UI docs describe shared sessions across CLI/Web, which enables powerful workflows (terminal + browser side-by-side). This can be implemented later via:
- Event-based approach in web server pushing to UI
- Shared session store (Redis/SQLite)
- SSE for real-time updates

### 4. Fix Actions Scope

**Decision:** Start read-only + recommendations. Stabilize first.

**Phase 1 (Current):** Read-only analysis with recommendations
- Show entropy issues
- Suggest resolutions
- Evaluate contracts

**Phase 2 (Future):** Preview-and-apply with undo
- Aggregation fixes (main trajectory)
- Source data fixes (limited scope)
- Full audit trail

### 5. MCP Tool Design

**Decision:** High-level tools, let LLM reason to combine attributes.

```python
# High-level tools (implemented)
get_context(output_dir) -> str           # Full context document
get_entropy(output_dir, table?) -> str   # Entropy summary
evaluate_contract(output_dir, contract) -> str
query(output_dir, question) -> str       # NL query

# NOT: granular tools like get_column_entropy, list_resolution_hints
# Let the LLM use get_entropy and reason about the response
```

---

## Command/Feature Mapping

### Pipeline Operations

| Command | CLI | TUI | Web | MCP | Notes |
|---------|-----|-----|-----|-----|-------|
| `run` | ✅ Primary | Progress display | Progress + live | ❌ | Batch, CI/CD |
| `status` | ✅ Quick check | ✅ Drill-down | Dashboard widget | `get_status` | |
| `phases` | ✅ Reference | ❌ | ❌ | `list_phases` | Dev reference |
| `reset` | ✅ Primary | Confirm dialog | ⚠️ Destructive | ❌ | Limited |

### Data Exploration

| Command | CLI | TUI | Web | MCP | Notes |
|---------|-----|-----|-----|-----|-------|
| Table list | `dataraum tables` | Tree view | Schema graph | `list_tables` | |
| Column details | `dataraum columns <t>` | Detail panel | + charts | `get_column` | |
| Sample data | `dataraum sample <t>` | Preview | Arrow table | `sample_data` | |
| Schema graph | ❌ | Limited | Cytoscape | JSON | Vis-heavy |

### Entropy & Quality

| Command | CLI | TUI | Web | MCP | Notes |
|---------|-----|-----|-----|-----|-------|
| Entropy summary | `dataraum entropy` | Dashboard | Radar + table | `get_entropy` | Core |
| Column entropy | `--column` flag | Drill-down | Detail view | Part of above | |
| Contracts | `dataraum contracts` | Traffic lights | Dashboard | `evaluate_contract` | |
| Resolution hints | In output | Panel | Action buttons | Part of entropy | |

### Query & Analysis

| Command | CLI | TUI | Web | MCP | Notes |
|---------|-----|-----|-----|-----|-------|
| NL Query | `dataraum query "..."` | Query screen | Conversational | `query` | Core |
| SQL execution | Raw result | Table view | Arrow + viz | ❌ (unsafe) | |
| Query library | `dataraum library` | Search | Search + ctx | `search_queries` | |
| Context | `dataraum context` | ❌ (large) | In response | `get_context` | For LLMs |

---

## Module Structure

```
src/dataraum/
├── __init__.py              # Public API exports
├── core/                    # Shared infrastructure
│   ├── connections.py       # ConnectionManager
│   ├── config.py            # Settings
│   └── models/              # Result type, etc.
│
├── analysis/                # Pipeline analysis (unchanged)
├── entropy/                 # Entropy detection (unchanged)
├── graphs/                  # Calculation graphs (unchanged)
├── query/                   # Query agent + library (unchanged)
├── llm/                     # LLM providers (unchanged)
├── pipeline/                # Orchestrator (unchanged)
├── storage/                 # SQLAlchemy base (unchanged)
│
├── cli/                     # CLI module (refactored from cli.py)
│   ├── __init__.py          # app = typer.Typer()
│   ├── commands/            # Command implementations
│   │   ├── run.py
│   │   ├── status.py
│   │   ├── entropy.py
│   │   ├── contracts.py
│   │   ├── query.py
│   │   └── inspect.py
│   ├── tui/                 # Textual app
│   │   ├── app.py           # DataraumApp
│   │   ├── screens/
│   │   │   ├── home.py
│   │   │   ├── entropy.py
│   │   │   ├── contracts.py
│   │   │   ├── query.py
│   │   │   └── table.py
│   │   ├── widgets/
│   │   │   ├── entropy_summary.py
│   │   │   └── table_tree.py
│   │   └── styles.tcss
│   └── formatters.py        # Rich output formatting
│
├── api/                     # FastAPI server
│   ├── main.py
│   ├── deps.py
│   ├── routers/
│   └── templates/           # Jinja2 for HTMX (future)
│
└── mcp/                     # MCP server
    ├── __init__.py
    ├── server.py            # Tool definitions
    └── formatters.py        # LLM-optimized output
```

---

## Python API for Jupyter

The public API (usable without CLI deps):

```python
from dataraum import Context

# Load a pipeline output
ctx = Context("./pipeline_output")

# Explore
ctx.tables                    # List of tables
ctx.entropy.summary()         # DataFrame with entropy scores
ctx.entropy.table("orders")   # Table-level detail
ctx.contracts.evaluate("aggregation_safe")  # ContractEvaluation

# Query
result = ctx.query("What's the total revenue by month?")
result.answer                 # Natural language answer
result.sql                    # Generated SQL
result.data                   # pandas DataFrame

# Get context for LLM
ctx.context_document()        # Formatted for prompts
```

This is the same interface whether you're in Jupyter, a script, or behind the API.

---

## TUI Design

### Navigation

Once in TUI, user can navigate between screens:
- Sidebar or `Tab` to switch: Home → Entropy → Contracts → Query
- `q` to quit
- `?` for help
- `Esc` to go back

### Screens

1. **HomeScreen** - Overview
   - Table tree (sidebar)
   - Pipeline status
   - Entropy summary widget
   - Recent queries

2. **EntropyScreen** - Entropy Dashboard
   - Table selector
   - Entropy scores table
   - Compound risks
   - Resolution hints panel

3. **ContractsScreen** - Contract Evaluation
   - Contract list with traffic lights
   - Drill-down to violations
   - Confidence levels

4. **QueryScreen** - Natural Language Query
   - Input box
   - Query history
   - Results display
   - Generated SQL preview

5. **TableScreen** - Table Detail
   - Column list with types
   - Statistics summary
   - Entropy per column
   - Sample data preview

---

## Implementation Phases

### Phase 1: CLI Refactor + Basic TUI ✅ COMPLETE
1. ✅ Refactor `cli.py` → `cli/` module with commands
2. ✅ Add `--tui` and `--json` flags
3. ✅ Create basic Textual app structure
4. ✅ Implement EntropyScreen (most visual)
5. ✅ Test with existing pipeline output

### Phase 2: Complete TUI ✅ COMPLETE
1. ✅ HomeScreen with table tree (clickable rows)
2. ✅ ContractsScreen with traffic lights
3. ✅ QueryScreen with history
4. ✅ TableScreen for drill-down
5. ✅ Navigation between screens (h/e/c///?/q)

### Phase 3: MCP Server ✅ COMPLETE
1. ✅ Implement high-level tools (get_context, get_entropy, evaluate_contract, query)
2. ✅ Create LLM-optimized formatters
3. ✅ Add `dataraum-mcp` entry point
4. ⏳ Document Claude Desktop config (see Phase 6)

### Phase 4: Python API ✅ COMPLETE
1. ✅ Create `Context` class
2. ✅ Entropy accessor
3. ✅ Contracts accessor
4. ✅ Query accessor
5. ✅ Jupyter-friendly display methods (`_repr_html_`, `to_dataframe`)

### Phase 5: Web UI (HTMX) ⏳ FUTURE
1. Add Jinja2 templates to FastAPI
2. Content negotiation (HTML vs JSON)
3. Basic dashboard
4. Conversational interface
5. Arrow data tables

### Phase 6: Documentation 📝 TODO
1. **User Documentation** - Consolidate into publishable docs
   - Getting Started guide
   - CLI reference
   - Python API reference
   - MCP integration guide (Claude Desktop config)
   - TUI usage guide
2. **Spec Cleanup** - Move completed specs to `docs/specs/`
   - Entropy specs → `docs/specs/entropy/`
   - Query specs → `docs/specs/query/`
   - Archive obsolete plans
3. **Web Publishing** - Set up docs site (MkDocs/Sphinx)

---

## Related Documents

- [docs/ui/01-architecture-overview.md](../ui/01-architecture-overview.md) - HTMX/HATEOAS vision
- [docs/ui/05-summary.md](../ui/05-summary.md) - UI summary
- [docs/plans/cli-tui-plan.md](./cli-tui-plan.md) - Previous CLI-first plan (superseded)
- [docs/ENTROPY_IMPLEMENTATION_PLAN.md](../ENTROPY_IMPLEMENTATION_PLAN.md) - Entropy system

---

## Cleanup Tasks

These should be done once the respective phases are complete:

1. ~~**Remove old CLI** (`src/dataraum/cli.py`)~~ ✅ DONE (2026-02-04)
2. **Clean up API routes** - After MCP and Web UI work; evaluate which routes are still needed
3. ~~**Update entry points**~~ ✅ DONE - `pyproject.toml` points to `cli/` module, verified working

---

## Open Questions (Deferred)

1. **Session expiry policy** - How long do sessions live?
2. **Undo history depth** - How many mutations to track?
3. **Semantic layer editing** - In v1 or later?
4. **Multi-user collaboration** - Shared sessions?
