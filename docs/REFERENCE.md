# DataRaum Context Engine — Project Reference

This document contains architecture, technology stack, module structure, design decisions,
prototype references, and file locations. Read this when you need project context.
For behavioral rules and workflows, see `CLAUDE.md` in the project root.

## What Is This?

A Python library for extracting rich metadata context from data sources to power
AI-driven data analytics. The core idea: instead of giving AI tools to discover
metadata at runtime, we pre-compute comprehensive metadata and serve it as
structured context documents interpreted through domain ontologies.

## Core Principles

1. **DuckDB for compute** — All data operations through DuckDB/PyArrow
2. **SQLAlchemy for metadata** — SQLite for dev, PostgreSQL for production
3. **Configuration as YAML** — Ontologies, patterns, rules, null values
4. **OSS-ready** — Clean interfaces, pip-installable, Apache-licensed core
5. **CLI-first** — Textual TUI for interactive use, MCP for LLM integration

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data compute | DuckDB | Load, query, transform data |
| Data interchange | PyArrow | Efficient data transfer |
| Pattern detection | PyArrow + Pint | Type inference, unit detection |
| Topology | TDA (existing prototype) | Relationship detection |
| Metadata storage | SQLAlchemy | SQLite (dev) / PostgreSQL (prod) |
| CLI | Typer + Rich | Command-line interface |
| TUI | Textual | Interactive terminal UI |
| AI interface | MCP SDK | Tool definitions for AI |
| Python Runtime | Python 3.14t | Free-threading for true parallelism |

## Free-Threading (NO_GIL)

This project uses **Python 3.14 free-threaded build** for true CPU parallelism. The GIL is disabled, enabling:
- ~3.5x speedup on 4-core CPU-bound work via `ThreadPoolExecutor`
- Pipeline phases run in parallel without GIL contention

```bash
# Run pipeline with GIL disabled
uv run python -Xgil=0 -m dataraum run /path/to/data

# Verify free-threading is enabled
python -c "import sys; print('Free-threading:', not sys._is_gil_enabled())"
```

### Architecture Notes
- **Single ConnectionManager** (`core/connections.py`) shared across modules
- **Pipeline in ThreadPoolExecutor**: Gets true parallelism from free-threading
- **DuckDB**: Read cursors are thread-safe; writes serialized via mutex

## Module Structure

```
src/dataraum/
├── analysis/       # Data analysis modules
│   ├── typing/         # Type inference, pattern detection
│   ├── statistics/     # Column profiling, distributions
│   ├── correlation/    # Numeric/categorical correlations
│   ├── relationships/  # Join detection, FK candidates
│   ├── semantic/       # LLM-powered semantic analysis
│   ├── temporal/       # Time series analysis
│   ├── slicing/        # Data slicing recommendations
│   ├── cycles/         # Business cycle detection
│   ├── validation/     # Data validation rules
│   └── quality_summary/# Quality report synthesis
├── entropy/        # Uncertainty quantification (core innovation)
│   ├── detectors/      # Entropy measurement per dimension
│   ├── context.py      # Entropy context builder
│   └── interpretation.py # LLM entropy interpretation
├── graphs/         # Calculation graphs, context assembly
├── pipeline/       # Pipeline orchestrator (18 phases)
├── sources/        # Data source loaders (CSV, Parquet)
├── storage/        # SQLAlchemy models
├── llm/            # LLM providers and prompts
├── core/           # Config, connections, utilities
├── cli/            # Textual TUI
│   ├── main.py         # Typer app, command definitions
│   ├── app.py          # Textual DataraumApp
│   ├── screens/        # TUI screens
│   └── widgets/        # TUI widgets
└── mcp/            # MCP server
    ├── server.py       # MCP tool definitions
    └── formatters.py   # LLM-optimized output
```

**Note:** SQLAlchemy DB models are co-located with business logic in `db_models.py` files within each module.

## Data Flow

```
Source (CSV/DB/API)
    ↓
[staging] Load as VARCHAR → raw_{table}
    ↓
[profiling] Statistical analysis → profiles, type_candidates
    ↓
[profiling] Type resolution → typed_{table}, quarantine_{table}
    ↓
[enrichment] LLM semantic analysis → roles, entities, relationships
    ↓
[enrichment] Topology (TDA) + temporal → additional metadata
    ↓
[quality] LLM rule generation + assessment → rules, scores
    ↓
[context] Assembly + LLM summary → ContextDocument (for AI)
         + LLM suggested queries
```

## Key Design Decisions

### VARCHAR-First Staging
Load all data as VARCHAR to preserve raw values. Type inference happens in
profiling, not during load. This prevents silent data loss.

### Quarantine Pattern
Failed type casts don't fail the pipeline. They go to quarantine tables for
human review. The workflow can checkpoint and wait for approval.

### Pre-computed Context
AI doesn't discover metadata. It receives a pre-assembled `ContextDocument`
with all relevant metadata already computed and interpreted through the
selected ontology.

### Minimal AI Tools
Only 6 MCP tools:
- `analyze` — Run pipeline on CSV/Parquet data
- `get_context` — Primary context retrieval
- `get_entropy` — Entropy analysis for tables/columns
- `evaluate_contract` — Data readiness evaluation
- `query` — Execute SQL with entropy awareness
- `get_actions` — Prioritized resolution actions

### Ontologies as Configuration
Domain ontologies (financial_reporting, marketing, etc.) are YAML configs that:
- Map column patterns to business terms
- Define computable metrics with formulas
- Provide domain-specific quality rules
- Guide semantic interpretation

### Core Concept — Entropy
The key innovation is quantifying **uncertainty (entropy)** in data so LLMs can make deterministic decisions. See `docs/ENTROPY_IMPLEMENTATION_PLAN.md` for the full spec.

## LLM-Powered Features

| Feature | Description | Fallback |
|---------|-------------|----------|
| Semantic Analysis | Column roles, entity types, relationships | `config/semantic_overrides.yaml` |
| Quality Rules | Domain-specific rule generation | `config/rules/*.yaml` |
| Suggested Queries | Context-aware SQL queries | Skip |
| Context Summary | Natural language data overview | Skip |

When LLM is disabled, manual YAML configuration is required for semantic analysis.
Prompts are customizable in `config/prompts/`.

## Existing Prototypes

### Pattern Detection (`prototypes/pattern_detection/helper`)
**REIMPLEMENT** — take inspiration from the code. Use the yaml configurations for patterns and null values inside config for a reimplementation. Don't use the stats part.

```python
from helper.convert_csv import ChunkedArrayAnalyzer
csv_analyzer = ChunkedArrayAnalyzer(header)
converted_row = csv_analyzer.process_chunked_array(struct_array)
# Returns: PatternResult with type_candidates, detected_patterns, detected_unit, stats
```

Features: regex-based pattern matching, Pint unit detection, type candidate scoring.

### Topology Analysis (`prototypes/topology/`)
**REIMPLEMENT** — take inspiration from the code, remove string comparisons for ranking, focus on extracting the topology and related metrics as additional context for the semantic analysis.

```python
from core.topology_extractor import TableTopologyExtractor
from core.relationship_finder import TableRelationshipFinder
extractor = TableTopologyExtractor()
finder = TableRelationshipFinder()
topology = extractor.extract_topology(df)
relationships = finder.find_relationships(tables)
# Returns: TopologyGraph with relationships, confidence scores
```

Features: TDA-based structural analysis, semantic similarity, FK candidate detection.

### Analytics Agents (`prototypes/analytics-agents-ts/`)
**REIMPLEMENT** — check the data-analysis prompts inside the prompts folder. Follow the same pattern of system prompt, user prompt, JSON Schema as a tool. Only focus on the semantic meaning, skip the data quality part.

Features: semantic analysis of data schema, business purpose summarization, dynamic prompt creation.

## File Locations

| What | Where |
|------|-------|
| Architecture docs | `docs/` |
| Existing prototypes | `prototypes/` |
| Source code | `src/dataraum/` |
| Tests | `tests/` |
| Ontology configs | `config/ontologies/` |
| Pattern configs | `config/patterns/` |
| Quality rule configs | `config/rules/` |
| LLM prompts | `config/prompts/` |
| LLM config | `config/llm.yaml` |
| Null value lists | `config/null_values.yaml` |
| Semantic overrides | `config/semantic_overrides.yaml` |
| Example data | `examples/data/` |
| Plans | `docs/plans/` |
| Specs | `docs/specs/` |
| Archive | `docs/archive/` |

## Dependencies

```toml
# pyproject.toml [project.dependencies]
dependencies = [
    "duckdb>=1.0.0",
    "pyarrow>=15.0.0",
    "pint>=0.23",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
    "pydantic>=2.0.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "textual>=0.50.0",
    "mcp>=1.0.0",
    "pyyaml>=6.0.0",
]

# Optional: LLM providers
# pip install dataraum[anthropic]  # Claude
# pip install dataraum[openai]     # OpenAI
# pip install dataraum[llm]        # Both

# Optional: Everything
# pip install dataraum[all]
```

## Implementation Order (Historical Reference)

> For current status, see `docs/BACKLOG.md`.
> Phases 1-4 are complete.

| Phase | Status |
|-------|--------|
| Phase 1: Foundation (Storage, Core/Config) | Complete |
| Phase 2A: LLM Infrastructure | Complete |
| Phase 2B: Data Pipeline | Complete |
| Phase 3: Intelligent Enrichment | Complete |
| Phase 4: Quality & Context (incl. Entropy) | Complete |
| Phase 5: Interfaces & Orchestration | In Progress |
