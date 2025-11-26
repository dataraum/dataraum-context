# DataRaum Context Engine

A rich metadata context engine for AI-driven data analytics.

## Overview

Traditional semantic layers tell BI tools "what things are called." This system tells AI "what the data means, how it behaves, how it relates, and what you can compute from it."

The core insight: AI agents don't need tools to discover metadata at runtime. They need **rich, pre-computed context** delivered in a format optimized for LLM consumption.

## Features

- **Statistical Metadata**: Distributions, cardinality, null rates, patterns
- **Semantic Metadata**: Column roles, entity types, business terms
- **Topological Metadata**: Relationships via TDA, join paths, hierarchies
- **Temporal Metadata**: Granularity, gaps, seasonality, trends
- **Quality Metadata**: Generated rules, scores, anomalies
- **Ontological Context**: Domain-specific interpretation (financial, marketing, etc.)

## Architecture

```
Data Sources → Staging → Profiling → Enrichment → Quality → Context
                  ↓           ↓           ↓           ↓         ↓
              DuckDB     PostgreSQL metadata storage      ContextDocument
                                                              ↓
                                                         AI (via MCP)
```

## Quick Start

```bash
# Install
pip install dataraum-context

# Run migrations (SQLite by default, no config needed)
dataraum-context migrate up

# Start API server
dataraum-context serve

# For production with PostgreSQL:
export DATARAUM_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dataraum"
dataraum-context migrate up
dataraum-context serve
```

## Usage

### Ingest Data

```python
from dataraum_context import ingest_source, SourceConfig

config = SourceConfig(
    name="sales_data",
    source_type="csv",
    path="data/sales.csv",
)

result = await ingest_source(config)
print(f"Loaded {result.total_rows} rows into {len(result.tables)} tables")
```

### Get Context

```python
from dataraum_context import get_context, ContextRequest

request = ContextRequest(
    tables=["sales", "customers"],
    ontology="financial_reporting",
)

context = await get_context(request)

# AI-consumable context document
print(context.tables)
print(context.relevant_metrics)
print(context.suggested_queries)
```

### MCP Integration

The context engine exposes 4 tools via MCP:

- `get_context` - Primary context retrieval
- `query` - Execute SQL against the data
- `get_metrics` - Available metrics for an ontology
- `annotate` - Update semantic annotations

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and component details
- [Data Model](docs/DATA_MODEL.md) - PostgreSQL metadata schema
- [Interfaces](docs/INTERFACES.md) - Module interfaces and data structures

## Technology Stack

| Component | Technology |
|-----------|------------|
| Data Compute | DuckDB |
| Data Interchange | PyArrow |
| Pattern Detection | PyArrow + Pint |
| Relationship Detection | TDA |
| Metadata Storage | SQLAlchemy (SQLite / PostgreSQL) |
| Dataflows | Apache Hamilton |
| API | FastAPI |
| AI Interface | MCP SDK |

## Development

```bash
# Clone
git clone https://github.com/dataraum/dataraum-context
cd dataraum-context

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy src/

# Lint
ruff check src/
```

## License

Apache 2.0
