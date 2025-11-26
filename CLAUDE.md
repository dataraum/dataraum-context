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
cat docs/DATA_MODEL.md      # SQLAlchemy schema
cat docs/INTERFACES.md      # Module interfaces

# Check configuration
cat config/llm.yaml         # LLM provider settings
cat config/prompts/         # Customizable prompts

# Check existing prototypes (USE AS INSPIRATION)
ls prototypes/pattern_detection/helper/ # PyArrow + Pint pattern detection, check _convertCsvFile in ../main.py
ls prototypes/topology/                 # TDA-based relationship detection
ls prototypes/analytics-agents-ts       # Agents and prompts (inside prompts folder) for data-engineering and business-analysis
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
├── storage/        # SQLAlchemy models and repository
├── llm/            # LLM providers, prompts, features
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
[enrichment] LLM semantic analysis → roles, entities, relationships
    ↓
[enrichment] Topology (TDA) + temporal → additional metadata
    ↓
[quality] LLM rule generation + assessment → rules, scores
    ↓
[context] Assembly + LLM summary → ContextDocument (for AI)
         + LLM suggested queries
```

## LLM-Powered Features

The following features use LLM (configurable via `config/llm.yaml`):

| Feature | Description | Fallback |
|---------|-------------|----------|
| Semantic Analysis | Column roles, entity types, relationships | `config/semantic_overrides.yaml` |
| Quality Rules | Domain-specific rule generation | `config/rules/*.yaml` |
| Suggested Queries | Context-aware SQL queries | Skip |
| Context Summary | Natural language data overview | Skip |

When LLM is disabled, manual YAML configuration is required for semantic analysis.
Prompts are customizable in `config/prompts/`.

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
**REIMPLEMENT** take inspiration from the code. Use the yaml configurations for patterns and null values inside config for a reimplementation. Don't use the stats part.

```python
# Entry point
from helper.convert_csv import ChunkedArrayAnalyzer

csv_analyzer = ChunkedArrayAnalyzer(header)
converted_row = csv_analyzer.process_chunked_array(struct_array)
# Returns: PatternResult with type_candidates, detected_patterns, detected_unit, stats
```

Features:
- Regex-based pattern matching (dates, emails, UUIDs, etc.)
- Pint integration for unit detection (kg, m/s, USD, etc.)
- Type candidate scoring with confidence

### Topology Analysis (`prototypes/topology/`)
**REIMPLEMENT** - take inspiration from the code, remove string comparisons for ranking, focus on extracting the topology and related metrics as additional context forr the semantic analysis.

```python
# Entry point
from core.topology_extractor import TableTopologyExtractor
from core.relationship_finder import TableRelationshipFinder

extractor = TableTopologyExtractor()
finder = TableRelationshipFinder()


topology = extractor.extract_topology(df)
relationships = finder.find_relationships(tables)

# Returns: TopologyGraph with relationships, confidence scores
```

Features:
- TDA-based structural analysis
- Semantic similarity integration
- FK candidate detection with evidence

### Topology Analysis (`prototypes/analytics-agents-ts/`)
**REIMPLEMENT** - take inspiration from the code, check the data-analysis prompts inside the prompts folder. Check the data-analysis schema inside the prommpts/schemas folder. Follow the same pattern of system prompt, user prompt, JSON Schema as a tool. Only focus on the semantic meaning, skip the data quality part

Features:
- Analyse the semantics of the provided data schema.
- Summarize the business purpose.
- Dynamically create the prompts, with context injected. 

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
    async with get_metadata_session() as session:
        result = await some_operation(duckdb_conn, session)
```

### Testing
- Unit tests for each module
- Integration tests with DuckDB in-memory + SQLite in-memory
- Property-based tests for pattern detection
- Mock LLM responses for deterministic testing
- Use pytest-asyncio for async tests

### Code Style
- Type hints on all functions
- Pydantic models for data classes
- No classes where functions suffice
- Docstrings with Args/Returns
- Max function length: ~50 lines

## Implementation Order

### Phase 1: Foundation
These must come first - everything else depends on them.

| Step | Module | Deliverables |
|------|--------|--------------|
| 1 | Storage | SQLAlchemy models, Alembic migrations, `llm_cache` table |
| 2 | Core/Config | Settings, YAML loaders, connection managers |

### Phase 2A: LLM Infrastructure (parallel track)

| Step | Module | Deliverables |
|------|--------|--------------|
| 3 | LLM Providers | `LLMProvider` ABC, Anthropic, OpenAI, Mock implementations |
| 4 | LLM Prompts + Privacy | Template renderer, SDV sample generator, cache logic |

### Phase 2B: Data Pipeline (parallel track)

| Step | Module | Deliverables |
|------|--------|--------------|
| 5 | Staging | VARCHAR-first loaders for CSV/Parquet/DB |
| 6 | Profiling | Pattern detection, type candidates, Pint units |

### Phase 3: Intelligent Enrichment
Requires: Storage, LLM, Profiling results

| Step | Module | Deliverables |
|------|--------|--------------|
| 7 | Enrichment/Semantic | `llm_analyze_semantics()`, manual override loading |
| 8 | Enrichment/Topology | TDA prototype wrapper, relationship detection |
| 9 | Enrichment/Temporal | Granularity, completeness, gap analysis |

### Phase 4: Quality & Context
Requires: All enrichment metadata

| Step | Module | Deliverables |
|------|--------|--------------|
| 10 | Quality | `llm_generate_rules()`, rule engine, scoring |
| 11 | Context | Assembly, `llm_generate_queries()`, `llm_generate_summary()` |

### Phase 5: Interfaces & Orchestration
Requires: Full pipeline working

| Step | Module | Deliverables |
|------|--------|--------------|
| 12 | API/MCP | FastAPI routes, MCP server, 4 tools |
| 13 | Dataflows | Hamilton DAG, checkpoints, resume logic |

### Testing Strategy

| Phase | Approach |
|-------|----------|
| 1-2 | Unit tests with SQLite in-memory |
| 3-4 | Mock LLM provider (no API calls) |
| 5-6 | Property-based tests, sample datasets |
| 7-9 | Integration tests: mock LLM + real DuckDB |
| 10-11 | Golden file tests for context documents |
| 12-13 | API contract tests, end-to-end tests |

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
| LLM prompts | `config/prompts/` |
| LLM config | `config/llm.yaml` |
| Null value lists | `config/null_values.yaml` |
| Semantic overrides | `config/semantic_overrides.yaml` |
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
    "pydantic>=2.0.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "sf-hamilton>=1.82.0",
    "mcp>=0.1.0",
    "pyyaml>=6.0.0",
]

# Optional: LLM providers
# pip install dataraum-context[anthropic]  # Claude
# pip install dataraum-context[openai]     # OpenAI
# pip install dataraum-context[llm]        # Both

# Optional: Synthetic data for privacy
# pip install dataraum-context[privacy]    # SDV

# Optional: Everything
# pip install dataraum-context[all]
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
