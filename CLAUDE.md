# DataRaum Context Engine

## What Is This?

A Python library for extracting rich metadata context from data sources to power 
AI-driven data analytics. The core idea: instead of giving AI tools to discover 
metadata at runtime, we pre-compute comprehensive metadata and serve it as 
structured context documents interpreted through domain ontologies.

## Quick Start for Claude Code

```bash
# Read these first
cat docs/ARCHITECTURE.md    # High-level design
cat docs/DATA_MODEL.md      # PostgreSQL schema
cat docs/INTERFACES.md      # Module interfaces

# Check existing prototypes (DO NOT REWRITE)
ls prototypes/pattern_detection/helper/ # PyArrow + Pint pattern detection, check _convertCsvFile in ../main.py
ls prototypes/topology/                 # TDA-based relationship detection
```

## Core Principles

1. **DuckDB for compute** - All data operations through DuckDB/PyArrow
2. **SQLAlchemy for metadata** - SQLite for dev, PostgreSQL for production
3. **Hamilton for dataflows** - Functional DAG with automatic lineage
4. **Configuration as YAML** - Ontologies, patterns, rules, null values
5. **OSS-ready** - Clean interfaces, pip-installable, Apache-licensed core

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data compute | DuckDB | Load, query, transform data |
| Data interchange | PyArrow | Efficient data transfer |
| Pattern detection | PyArrow + Pint | Type inference, unit detection |
| Topology | TDA (existing prototype) | Relationship detection |
| Metadata storage | SQLAlchemy | SQLite (dev) / PostgreSQL (prod) |
| Dataflows | Apache Hamilton | DAG orchestration with lineage |
| API | FastAPI | HTTP interface |
| AI interface | MCP SDK | Tool definitions for AI |

## Module Structure

```
src/dataraum_context/
├── core/           # Config, connections, shared models
├── staging/        # Raw data loading (VARCHAR-first)
├── profiling/      # Statistical metadata, type inference
├── enrichment/     # Semantic, topological, temporal metadata
├── quality/        # Rule generation, scoring, anomalies
├── context/        # Context assembly, ontology application
├── storage/        # PostgreSQL metadata repository
├── dataflows/      # Hamilton dataflow definitions
├── api/            # FastAPI routes
└── mcp/            # MCP server and tools
```

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
[enrichment] Semantic, topology, temporal → metadata tables
    ↓
[quality] Rule generation + assessment → rules, scores
    ↓
[context] Assembly with ontology → ContextDocument (for AI)
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
Only 4 MCP tools:
- `get_context` - Primary context retrieval
- `query` - Execute SQL
- `get_metrics` - Available metrics for ontology  
- `annotate` - Human-in-loop semantic updates

### Ontologies as Configuration
Domain ontologies (financial_reporting, marketing, etc.) are YAML configs that:
- Map column patterns to business terms
- Define computable metrics with formulas
- Provide domain-specific quality rules
- Guide semantic interpretation

## Existing Prototypes

### Pattern Detection (`prototypes/pattern_detection/helper`)
**REIMPLEMENT** take inspiration from the code. Use the yaml configurations for patterns and null values inside config for a reimplementation.

```python
# Entry point
from prototypes.pattern_detection import detect_patterns

result = detect_patterns(arrow_table, column_name)
# Returns: PatternResult with type_candidates, detected_patterns, detected_unit
```

Features:
- Regex-based pattern matching (dates, emails, UUIDs, etc.)
- Pint integration for unit detection (kg, m/s, USD, etc.)
- Type candidate scoring with confidence

### Topology Analysis (`prototypes/topology/`)
**REIMPLEMENT** - take inspiration from the code, remove string comparisons for ranking, focus on extracting the topology and related metrics as additional context forr the semantic analysis.

```python
# Entry point
from prototypes.topology import analyze_topology

result = analyze_topology(tables_dict)  # {name: ArrowTable}
# Returns: TopologyGraph with relationships, confidence scores
```

Features:
- TDA-based structural analysis
- Semantic similarity integration
- FK candidate detection with evidence

## Implementation Guidelines

### Error Handling
```python
# Use Result type, not exceptions
from dataraum_context.core.models import Result

async def some_operation() -> Result[SomeOutput]:
    try:
        # ... do work ...
        return Result.ok(output, warnings=["minor issue"])
    except SomeExpectedError as e:
        return Result.fail(str(e))
```

### Database Connections
```python
# Use async context managers
async with get_duckdb_connection() as duckdb_conn:
    async with get_metadata_connection() as pg_conn:
        result = await some_operation(duckdb_conn, pg_conn)
```

### Testing
- Unit tests for each module
- Integration tests with DuckDB in-memory + test PostgreSQL
- Property-based tests for pattern detection
- Use pytest-asyncio for async tests

### Code Style
- Type hints on all functions
- Pydantic models for data classes
- No classes where functions suffice
- Docstrings with Args/Returns
- Max function length: ~50 lines

## Suggested Implementation Order

- *Storage* - SQLAlchemy migrations from DATA_MODEL.md
- *Staging* - VARCHAR-first loaders
- *Profiling* - Integrate pattern detection based on the prototype and new configurations
- *Enrichment* - Integrate and extend TDA prototype, removing ranking attempt
- *Quality* - Rule generation and scoring
- *Context* - Assembly and ontology application
- *API/MCP* - FastAPI routes and MCP tools
- *Workflows* - Hamilton orchestration

## File Locations

| What | Where |
|------|-------|
| Architecture docs | `docs/` |
| Existing prototypes | `prototypes/` |
| Source code | `src/dataraum_context/` |
| Tests | `tests/` |
| Ontology configs | `config/ontologies/` |
| Pattern configs | `config/patterns/` |
| Quality rule configs | `config/rules/` |
| Null value lists | `config/null_values.yaml` |
| Example data | `examples/data/` |

## Dependencies

```toml
# pyproject.toml [project.dependencies]
dependencies = [
    "duckdb>=1.0.0",
    "pyarrow>=15.0.0",
    "pint>=0.23",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
    "asyncpg>=0.29.0",  # Optional PostgreSQL support
    "pydantic>=2.0.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "sf-hamilton>=1.82.0",
    "mcp>=0.1.0",
    "pyyaml>=6.0.0",
]
```

## Quick Reference

### Run tests
```bash
pytest tests/ -v
```

### Start API server
```bash
uvicorn dataraum_context.api.fastapi_app:app --reload
```

### Start MCP server
```bash
python -m dataraum_context.mcp.server
```

### Run migration
```bash
python -m dataraum_context.storage.migrations up
```
