# Architecture Overview

## Vision

Traditional semantic layers tell BI tools "what things are called." This system tells AI 
"what the data means, how it behaves, how it relates, and what you can compute from it."

The core insight: AI agents don't need tools to discover metadata at runtime. They need 
**rich, pre-computed context** delivered in a format optimized for LLM consumption.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI CONSUMERS                                       │
│                                                                             │
│   Claude / GPT / Local LLM  ←──  MCP Server  ←──  ContextDocument           │
│                                   (4 tools)       (pre-assembled)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTEXT LAYER                                      │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Ontologies    │    │    Assembly     │    │   Suggested     │        │
│   │                 │    │                 │    │   Queries       │        │
│   │ • financial     │───▶│ Combine all     │───▶│                 │        │
│   │ • marketing     │    │ metadata into   │    │ Generate useful │        │
│   │ • operations    │    │ ContextDocument │    │ starting points │        │
│   │ • custom        │    │                 │    │                 │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METADATA MODULES                                   │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │ Statistical │  │  Semantic   │  │ Topological │  │  Temporal   │      │
│   │             │  │             │  │             │  │             │      │
│   │ • profiles  │  │ • roles     │  │ • FKs (TDA) │  │ • granular- │      │
│   │ • distribs  │  │ • entities  │  │ • hierarchs │  │   ity       │      │
│   │ • patterns  │  │ • terms     │  │ • join paths│  │ • gaps      │      │
│   │ • units     │  │ • domains   │  │ • graph     │  │ • trends    │      │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│          │                │                │                │              │
│          └────────────────┴────────────────┴────────────────┘              │
│                                    │                                        │
│                           ┌────────▼────────┐                              │
│                           │     Quality     │                              │
│                           │                 │                              │
│                           │ • rules (gen'd) │                              │
│                           │ • scores        │                              │
│                           │ • anomalies     │                              │
│                           └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                         DuckDB                                   │      │
│   │                                                                  │      │
│   │   raw_{table}  ──▶  typed_{table}  ──▶  (queries)               │      │
│   │        │                  │                                      │      │
│   │        │                  ▼                                      │      │
│   │        │         quarantine_{table}                              │      │
│   │        │                                                         │      │
│   │        ▼                                                         │      │
│   │   Source files (CSV, Parquet, etc.)                             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION                                     │
│                                                                             │
│   Apache Hamilton Dataflows                                                 │
│   ┌────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐       │
│   │ Stage  │─▶│ Profile │─▶│ Resolve  │─▶│ Enrich  │─▶│ Quality │       │
│   │        │   │         │   │ Types    │   │         │   │         │       │
│   └────────┘   └────┬────┘   └────┬─────┘   └─────────┘   └─────────┘       │
│                     │             │                                         │
│                     ▼             ▼                                         │
│              [CHECKPOINT]  [CHECKPOINT]                                     │
│              Human review  Quarantine                                       │
│              of types      review                                           │
│              (SQL DB)     (SQL DB)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METADATA STORAGE                                  │
│                                                                             │
│   SQLAlchemy (SQLite dev / PostgreSQL prod)                                 │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ metadata.sources          metadata.column_profiles              │       │
│   │ metadata.tables           metadata.type_candidates              │       │
│   │ metadata.columns          metadata.semantic_annotations         │       │
│   │ metadata.relationships    metadata.temporal_profiles            │       │
│   │ metadata.quality_rules    metadata.quality_scores               │       │
│   │ metadata.ontologies       metadata.checkpoints                  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Deep Dives

### Staging Layer

**Purpose**: Get data into DuckDB without losing information.

**Key Decision**: VARCHAR-first loading.

Why? CSV readers like `read_csv_auto` make irreversible type decisions:
- "N/A" in an integer column → NULL (data loss)
- "1,234.56" vs "1.234,56" (locale ambiguity)
- "2024-01-15" vs "01/15/2024" (format ambiguity)

By loading everything as VARCHAR:
- Original values preserved exactly
- Type inference happens explicitly in profiling
- Failed casts go to quarantine, not oblivion

```sql
-- What we create
CREATE TABLE raw_sales (
    _source_file VARCHAR,      -- provenance
    _source_row BIGINT,        -- for quarantine linkage
    _loaded_at TIMESTAMP,      -- audit
    date VARCHAR,              -- all actual columns as VARCHAR
    amount VARCHAR,
    customer_id VARCHAR,
    ...
);
```

### Profiling Layer

**Purpose**: Extract statistical metadata and infer types.

**Components**:

1. **Statistical Profiler** - Computes distributions, cardinality, null rates
2. **Pattern Detector** (existing prototype) - Identifies patterns, detects units
3. **Type Inferencer** - Produces type candidates with confidence scores

**Type Inference Logic**:

```python
# For each column, try casting to each type
type_candidates = []
for dtype in [BIGINT, DOUBLE, DATE, TIMESTAMP, BOOLEAN]:
    success_rate = try_cast_success_rate(column, dtype)
    if success_rate > 0.5:  # At least half parse
        type_candidates.append(TypeCandidate(
            dtype=dtype,
            confidence=success_rate,
            failed_examples=get_failures(column, dtype)
        ))

# Pattern detection enriches this
pattern_result = detect_patterns(arrow_table, column)
if pattern_result.detected_unit:
    # "kg", "USD", "m/s" → suggests numeric type + semantic info
```

**Output**: `type_candidates` table with multiple options per column, ranked by confidence.

### Type Resolution Layer

**Purpose**: Apply type decisions and handle failures gracefully.

**Key Pattern**: Quarantine

```sql
-- Create typed table with cast attempts
CREATE TABLE typed_sales AS
SELECT 
    _source_row,
    TRY_CAST(date AS DATE) as date,
    TRY_CAST(amount AS DOUBLE) as amount,
    TRY_CAST(customer_id AS BIGINT) as customer_id,
    -- Track what failed
    CASE WHEN date IS NOT NULL AND TRY_CAST(date AS DATE) IS NULL 
         THEN TRUE ELSE FALSE END as _date_failed,
    ...
FROM raw_sales;

-- Extract failures for review
CREATE TABLE quarantine_sales AS
SELECT raw.*, 'date_cast_failed' as _reason
FROM raw_sales raw
JOIN typed_sales typed ON raw._source_row = typed._source_row
WHERE typed._date_failed;
```

**Workflow Checkpoint**: If quarantine has rows, workflow pauses for human review.

### Enrichment Layer

**Purpose**: Extract semantic, topological, and temporal metadata.

#### Semantic Enrichment

- **Column Role Detection**: Is this a measure, dimension, key, timestamp?
- **Entity Detection**: What does this table represent? (customers, transactions, products)
- **Business Term Mapping**: Match columns to ontology concepts

```python
# Role detection heuristics
if column.cardinality_ratio > 0.95 and column.null_ratio == 0:
    likely_role = "key"
elif column.dtype in NUMERIC_TYPES and column.semantic_patterns.is_additive:
    likely_role = "measure"
elif column.dtype in TIME_TYPES:
    likely_role = "timestamp"
else:
    likely_role = "dimension" or "attribute"
```

#### Topological Enrichment (TDA Prototype)

The existing prototype uses Topological Data Analysis to detect relationships:

- Value overlap analysis (traditional FK detection)
- Structural similarity (TDA features)
- Semantic similarity (column name embeddings)

Output: Relationship graph with confidence scores and evidence.

#### Temporal Enrichment

For identified time columns:

```python
# Granularity detection
gaps = compute_consecutive_gaps(time_column)
dominant_gap = mode(gaps)

granularity = match_granularity(dominant_gap)  # day, week, month, etc.

# Completeness analysis
expected = date_range / granularity_interval
actual = distinct_periods
completeness = actual / expected

# Gap detection
gaps = find_gaps(time_column, granularity)
```

### Quality Layer

**Purpose**: Generate and execute quality rules, compute scores.

#### Rule Generation

Rules are generated from:
1. **Metadata signals** (null rates → not_null rules, cardinality → unique rules)
2. **Ontology definitions** (financial data should have non-negative revenue)
3. **Statistical baselines** (values outside 3σ are anomalies)

```python
# Generated rules
if column.null_ratio == 0:
    generate_rule(NotNullRule(column))

if column.cardinality_ratio > 0.99:
    generate_rule(UniqueRule(column))

if relationship.type == "foreign_key" and relationship.is_confirmed:
    generate_rule(ReferentialIntegrityRule(relationship))
```

#### Quality Scoring

Five dimensions (DQ standard):
- **Completeness**: Null rates
- **Validity**: Values within expected ranges/patterns
- **Consistency**: Referential integrity, cross-column rules
- **Uniqueness**: Duplicate detection
- **Timeliness**: Data freshness, gap analysis

### Context Layer

**Purpose**: Assemble all metadata into AI-consumable documents.

#### Ontology Application

Ontologies are YAML configs:

```yaml
name: financial_reporting
concepts:
  - name: revenue
    indicators: ["revenue", "sales", "income"]
    temporal_behavior: additive
metrics:
  - name: gross_margin
    formula: "(revenue - cogs) / revenue"
    required_concepts: [revenue, cogs]
quality_rules:
  - revenue >= 0
  - date should have no gaps
```

When context is requested with `ontology=financial_reporting`:
1. Columns are mapped to concepts
2. Applicable metrics are identified
3. Domain-specific quality rules are applied
4. Context document is annotated with financial semantics

#### Context Document Structure

```python
ContextDocument(
    tables=[
        TableContext(
            name="sales",
            description="Daily sales transactions",  # inferred
            row_count=1_000_000,
            entity_type="transaction",
            grain_columns=["sale_id"],
            time_columns=["sale_date"],
            granularity="daily",
            quality_score=0.94,
            columns=[
                ColumnContext(
                    name="amount",
                    data_type="DOUBLE",
                    semantic_role="measure",
                    business_name="Sale Amount",
                    detected_unit="USD",
                    quality_score=0.98,
                ),
                ...
            ]
        )
    ],
    relationships=[...],
    ontology="financial_reporting",
    relevant_metrics=[
        MetricDefinition(
            name="Total Revenue",
            formula="SUM(amount)",
            applicable_tables=["sales"],
        ),
        ...
    ],
    quality_summary=QualitySummary(
        overall_score=0.92,
        critical_issues=["Missing data in Q3 2024"],
    ),
    suggested_queries=[
        SuggestedQuery(
            description="Revenue by month",
            sql="SELECT DATE_TRUNC('month', sale_date), SUM(amount) FROM sales GROUP BY 1",
        ),
        ...
    ]
)
```

### API Layer

**FastAPI** for HTTP access:

```
POST /sources              # Add data source
GET  /sources/{id}/tables  # List tables
POST /profile              # Trigger profiling
POST /context              # Get context document
PUT  /columns/{id}/annotation  # Human-in-loop updates
GET  /metrics/{ontology}   # Available metrics
POST /query                # Execute SQL
```

**MCP Server** for AI tool access:

Only 4 tools - minimal surface area:

| Tool | Purpose |
|------|---------|
| `get_context` | Primary context retrieval |
| `query` | Execute SQL (read-only) |
| `get_metrics` | List metrics for ontology |
| `annotate` | Update semantic annotations |

### Dataflow Layer

**Apache Hamilton** provides:
- Functional DAG definition via Python functions
- Automatic lineage tracking
- Built-in data validation (`@check_output`)
- Parallel execution where possible
- Visualization of dataflows

**PostgreSQL checkpoints** provide:
- Human-in-loop review state persistence
- Workflow resume after interruption
- Audit trail of decisions

Key dataflows:
- `ingest_dataflow` - Full pipeline from source to context
- `profile_dataflow` - Profiling only (for re-profiling)
- `enrich_dataflow` - Enrichment only (after manual type fixes)

## Key Algorithms

### TDA for Relationship Detection

The topology prototype uses persistent homology to identify structural patterns 
in data distributions that suggest relationships:

1. Build point cloud from column value distributions
2. Compute persistence diagrams
3. Compare diagrams across tables for similarity
4. Combine with value overlap for FK confidence

### Type Inference Pipeline

```
Raw values
    ↓
Pattern matching (regex) → semantic hints
    ↓
Unit detection (Pint) → numeric + unit type
    ↓
Cast testing → success rates per type
    ↓
Confidence scoring → ranked candidates
    ↓
Human review (if ambiguous) → final decision
```

### Quality Rule Generation

```
Metadata signals
    ↓
Statistical baselines (mean, stddev, ranges)
    ↓
Cardinality analysis (unique, low-cardinality)
    ↓
Relationship constraints (FK integrity)
    ↓
Ontology rules (domain-specific)
    ↓
Rule deduplication and prioritization
```

## Deployment Considerations

### Compute vs Storage Separation

- **DuckDB**: Ephemeral compute. Can be in-memory or file-based.
- **SQLite/PostgreSQL**: Persistent metadata. SQLite for dev, PostgreSQL for production.
- **Data files**: Parquet preferred for DuckDB performance.

### Scaling

- Profiling is embarrassingly parallel (per-column)
- TDA is expensive - consider sampling for large tables
- Quality assessment parallelizes well
- Context assembly is cheap (metadata queries)
- For production scale, switch to PostgreSQL for concurrent access

### Caching

- Profile results are versioned (re-run on data change)
- Context documents can be cached with TTL
- Configuration files (ontologies, patterns, rules) cached in memory

## Extension Points

### Custom Ontologies

Add YAML to `config/ontologies/` directory:

```yaml
name: my_domain
concepts: [...]
metrics: [...]
quality_rules: [...]
```

### Custom Pattern Detectors

Extend pattern detection prototype:

```python
CUSTOM_PATTERNS = {
    'my_id_format': r'^[A-Z]{3}-\d{6}$',
}
```

### Custom Quality Rules

```python
@quality_rule
def my_custom_rule(table: str, column: str) -> QualityRule:
    return QualityRule(
        rule_type="custom",
        rule_expression="...",
    )
```
