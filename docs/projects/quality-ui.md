# Project: Quality UI

*Web-based quality dashboard for visualization, reports, and team sharing.*

---

## Problem

The MCP plugin outputs text/Markdown — fine for interactive LLM sessions, but not for:
- Visual dashboards (quality scores, trends, dimension breakdowns)
- Shareable reports (send to a manager or auditor)
- Team workflows (multiple people reviewing the same data quality)
- Persistent views (bookmark and revisit without re-running tools)

## Scope

This project lives in `../dataraum-ui` — a separate web application that reads from the same `pipeline_output/` database the plugin writes to.

## Architecture

```
dataraum-context (this repo)          dataraum-ui (separate repo)
┌─────────────────────┐               ┌─────────────────────┐
│  Pipeline           │               │  Web Service         │
│  MCP Server         │               │  (FastAPI / HTMX)    │
│  Plugin/Skills      │               │                      │
│         │           │               │  Quality Dashboard   │
│         ▼           │               │  Report Export       │
│  pipeline_output/   │──── reads ───▶│  Trend Charts        │
│    SQLite DB        │               │  Fix Review UI       │
│    DuckDB files     │               │                      │
└─────────────────────┘               └─────────────────────┘
```

The UI is a **read layer** on top of the existing data. It doesn't run pipelines or apply fixes — it visualizes what the pipeline and plugin have produced.

## Features

### Quality Dashboard

- Overall quality grade per source (A–F scale)
- Dimension breakdown (structural, semantic, value, computational) as gauges or bars
- Column-level drill-down: entropy scores, top issues, resolution status
- Action list with priority, effort, status (from fixes ledger)
- Contract compliance summary (pass/fail per contract)

### Quality Report Export

Generate a self-contained HTML report (single file, inline CSS/JS):
- Overall quality grade per dimension
- Resolved vs open actions
- Quality score trend over runs
- Cleaned dataset summary
- Exportable as PDF via print/weasyprint

### Quality Score Trend

Once `QualitySnapshot` has multiple runs:
- Score per dimension over time (line chart)
- Annotated with what was fixed between runs
- Regression detection (quality dropped — what changed?)

### Fix Review UI

Visual interface for reviewing and approving fixes from the ledger:
- List of pending fixes with before/after preview
- Approve / reject / modify workflow
- Batch operations (approve all semantic fixes, etc.)

## Technology

- **Backend**: FastAPI (already partially built in `src/dataraum/api/`)
- **Frontend**: HTMX + minimal JS for interactivity
- **Charts**: Chart.js or similar lightweight library
- **Report export**: Jinja2 HTML templates + optional PDF
- **No SPA framework**: keep it simple, server-rendered

## Data Access

The UI reads from the same SQLite database in `pipeline_output/`:
- `EntropySnapshotRecord` — quality scores
- `EntropyInterpretationRecord` — column-level analysis
- `PipelineRun` / `PhaseCheckpoint` — run history
- `ResolutionRecord` — fix status (from fixes project)
- `QualitySnapshot` — trend data (from persistent state)

DuckDB files for actual data queries (show example rows, run ad-hoc queries).

## How Local Data Gets There

For now: the UI runs locally and reads from the local `pipeline_output/` directory. The path is configured via environment variable or CLI argument.

Future (Phase B/C from BACKLOG.md):
- Remote MCP server + hosted pipeline = data lives server-side
- UI becomes the web frontend for the hosted service
- Same read layer, different data location

## Dependencies

- Persistent state layer (for resolution/quality data to display)
- Fixes ledger (for fix review UI)
- Pipeline output DB (already exists)

## Open Questions

- Should the UI be able to trigger pipeline runs, or is it strictly read-only?
- How do we handle concurrent access (plugin writing while UI reads)?
- Should the UI have its own auth layer for team scenarios?
- Chart rendering: server-side (matplotlib/plotly images) vs client-side (Chart.js)?
