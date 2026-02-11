# Calculation Engine Design

## Problem Statement

Financial analysis requires:
1. **Reproducibility** - Same inputs must produce identical outputs every time
2. **Transparency** - Users must see exactly how a number was calculated
3. **Traceability** - Full lineage from raw data to final metrics
4. **Explorability** - Visual representation of calculation flow

**Why CTEs Don't Work:**
- Embedded in SQL, hard to visualize as separate steps
- LLM-generated queries vary slightly each time
- No persistent intermediate results for inspection
- Cannot answer "how was this number calculated?" after the fact

## Design Principles

### 1. Materialized Intermediate Steps (Not CTEs)

```
WRONG: Single SQL with CTEs
┌─────────────────────────────────────────────────────────┐
│ WITH step1 AS (...), step2 AS (...), step3 AS (...)    │
│ SELECT final FROM step3                                 │
└─────────────────────────────────────────────────────────┘
Problem: Steps exist only during query execution

RIGHT: Persisted calculation steps
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Level 1  │───▶│ Level 2  │───▶│ Level 3  │───▶│ Output   │
│ (saved)  │    │ (saved)  │    │ (saved)  │    │ (saved)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
Benefit: Each step is queryable, traceable, visualizable
```

### 2. Unified Agent-Based Execution (ALL Graphs)

```
ANY Graph (filter OR metric): Specifications, hints, accounting context
          ↓
Graph Agent: LLM interprets graph + data schema → generates executable SQL
          ↓
Generated SQL: Cached, deterministic, auditable
          ↓
Code Executor: Runs SQL, captures results
          ↓
Visualization: Show exactly what happened
```

**Key Insight**: ALL graphs (filters AND metrics) are specifications, not executable code.

Even filter graphs have the same mapping problem:
```yaml
# Filter graph might say:
condition: "revenue > 0"  # But what column IS "revenue"?
```

The actual data might have `Umsatz` column, or `Betrag` with account codes 4000-4999.
Only an LLM can interpret this mapping.

**Unified Agent Approach**:
- ONE agent handles both filter and metric graphs
- ONE code path for all graph types
- ONE caching strategy for generated SQL
- Simpler architecture, easier to maintain

The **Graph Agent** uses LLM to:
1. Read ANY graph specification (filter or metric)
2. Analyze the actual data schema (column names, data types, samples)
3. Generate concrete SQL that maps abstract fields to real columns
4. Cache generated SQL by (graph_id, version, schema_mapping_id)
5. Execute cached SQL deterministically (no LLM in execution path)

**Why This Matters**:
- Raw data may have `Konto` column with values 1200, 1400
- Graph says `standard_field: "accounts_receivable"` or `condition: "revenue > 0"`
- Agent understands: "In DATEV SKR03, accounts 1200, 1400 = AR, accounts 4000-4999 = revenue"
- Direct execution CANNOT do this mapping - there is no "executor" that works without LLM

### 3. Named, Versioned Calculation Graphs

Each calculation (DSO, Cash Runway, OCF) is defined in YAML:
- Graph ID and version for reproducibility
- Dependencies organized by execution level
- Validation rules (pre and post)
- Interpretation guides

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ Q&A Agent   │  │ Dashboard   │  │ Visual Graph Explorer           │  │
│  │ "What is    │  │ KPI Cards   │  │ (React Flow)                    │  │
│  │  our DSO?"  │  │             │  │ - Expand/collapse steps         │  │
│  └──────┬──────┘  └──────┬──────┘  │ - Drill into transactions       │  │
│         │                │         │ - See calculation breakdown     │  │
└─────────┼────────────────┼─────────┴─────────────────────────────────┴──┘
          │                │                        │
          ▼                ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GRAPH AGENT LAYER (Unified)                         │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               Graph Agent (LLM-Powered)                          │    │
│  │  - Handles ALL graph types (filters AND metrics)                 │    │
│  │  - Loads graph YAML (specification with accounting context)      │    │
│  │  - Analyzes actual data schema (columns, types, samples)         │    │
│  │  - Generates executable SQL for THIS specific dataset            │    │
│  │  - Caches generated SQL for deterministic re-execution           │    │
│  │  - Returns execution trace with full lineage                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ Graph Loader    │  │ SQL Generator   │  │ SQL Executor    │          │
│  │ - Parse YAML    │  │ - LLM prompt    │  │ - Run cached    │          │
│  │ - Rich context  │  │   construction  │  │   SQL in DuckDB │          │
│  │   for agent     │  │ - SQL parsing   │  │ - Capture       │          │
│  │                 │  │ - Validation    │  │   step results  │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               Generated SQL Cache                                │    │
│  │  - Key: (graph_id, graph_version, schema_mapping_id)             │    │
│  │  - Value: Validated SQL ready for execution                      │    │
│  │  - Cache hit → No LLM call, deterministic execution              │    │
│  │  - Cache miss → LLM generates SQL, validates, caches             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │                                             │
          ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Filtered Views                                │    │
│  │  clean_transactions  │  clean_customers  │  clean_accounts       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ▲                                           │
│                              │ (Filters applied)                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Raw Tables                                    │    │
│  │  raw_transactions  │  raw_customers  │  raw_chart_of_accounts    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Calculation Graph Definition (YAML)

```yaml
graph_id: "dso"
version: "1.0"
category: "working_capital"

output:
  metric_id: "dso"
  description: "Days Sales Outstanding"
  unit: "days"

dependencies:
  # Level 1: Raw data extraction
  accounts_receivable:
    level: 1
    type: "direct"
    source:
      statement: "balance_sheet"
      standard_field: "accounts_receivable"
      aggregation: "end_of_period"

  revenue:
    level: 1
    type: "direct"
    source:
      statement: "income_statement"
      standard_field: "revenue"
      aggregation: "sum"

  days_in_period:
    level: 1
    type: "constant"
    default_value: 30

  # Level 2: Derived calculation
  dso:
    level: 2
    type: "derived"
    calculation:
      formula: "(accounts_receivable / revenue) * days_in_period"
      depends_on: ["accounts_receivable", "revenue", "days_in_period"]
```

### 2. The Calculation Agent

The Calculation Agent is an LLM-powered component that bridges the gap between
abstract graph specifications and concrete data schemas.

**Why an Agent is Required:**

```
Graph Specification (abstract):
┌─────────────────────────────────────────────────────────────┐
│ dependencies:                                               │
│   accounts_receivable:                                      │
│     source:                                                 │
│       standard_field: "accounts_receivable"                 │
│       statement: "balance_sheet"                            │
│     datev_mapping:                                          │
│       skr03: [1200, 1400, 1576]                            │
│     accounting_notes: |                                     │
│       Should include trade receivables...                   │
│       Should EXCLUDE employee advances...                   │
└─────────────────────────────────────────────────────────────┘
          ↓
          ↓ Agent interprets + generates code
          ↓
Actual Data Schema:
┌─────────────────────────────────────────────────────────────┐
│ clean_buchungen (filtered transactions):                    │
│   - Konto: INTEGER (account number)                         │
│   - Betrag: DECIMAL (amount)                                │
│   - Buchungsdatum: DATE                                     │
│   - Kontenart: TEXT (SKR03)                                │
└─────────────────────────────────────────────────────────────┘
          ↓
          ↓ Agent generates
          ↓
Generated SQL:
┌─────────────────────────────────────────────────────────────┐
│ SELECT SUM(Betrag) as accounts_receivable                   │
│ FROM clean_buchungen                                        │
│ WHERE Konto IN (1200, 1400, 1576)                          │
│   AND Buchungsdatum = (SELECT MAX(Buchungsdatum) ...)      │
└─────────────────────────────────────────────────────────────┘
```

**Agent Workflow:**

1. **Receive Request**: "Calculate DSO for March 2025"
2. **Load Graph**: Parse `dso_calculation_graph.yaml` with full context
3. **Analyze Schema**: Query actual table structure, sample data, column meanings
4. **Generate Code**: Create SQL/Python that maps abstract fields to concrete columns
5. **Validate Code**: Check generated SQL syntax, column references
6. **Execute Code**: Run deterministically, capture results
7. **Persist**: Store generated code + results for reproducibility

**Generated Code Storage:**

```python
@dataclass
class GeneratedCalculationCode:
    """LLM-generated code for a specific graph + schema combination."""

    code_id: str                    # Unique ID
    graph_id: str                   # Which graph
    graph_version: str              # Graph version
    schema_mapping_id: str          # Which schema mapping

    # Generated code
    generated_sql: str              # The actual SQL
    generated_python: str | None    # Optional Python for complex logic

    # Generation metadata
    llm_model: str                  # Which model generated this
    llm_prompt_hash: str            # Hash of prompt for reproducibility
    generation_timestamp: datetime

    # Validation
    is_validated: bool              # Has code been validated
    validation_errors: list[str]
```

**Caching Strategy:**

Once code is generated for a (graph_id, schema_mapping_id) pair:
- Subsequent executions reuse cached code (no LLM call)
- Cache invalidates when graph version or schema changes
- User can force regeneration if needed

### 3. Execution Result Model

```python
@dataclass
class CalculationExecution:
    """A single execution of a calculation graph."""

    execution_id: str           # Unique ID for this execution
    graph_id: str               # Which calculation graph
    graph_version: str          # Version of the graph

    # Input parameters
    parameters: dict            # User-provided parameters
    schema_mapping_id: str      # Which schema mapping was used
    filter_execution_id: str    # Which filter execution provided data

    # NEW: Link to generated code
    generated_code_id: str      # Which generated code was executed

    # Results
    output_value: float         # Final metric value
    output_unit: str            # Unit of the metric
    interpretation: str | None  # e.g., "GOOD", "CONCERNING"

    # Reproducibility
    calculation_hash: str       # Hash of (graph_id, version, params, mapping, filter)
    executed_at: datetime

    # Traceability
    step_results: list[StepResult]  # All intermediate results


@dataclass
class StepResult:
    """Result of a single calculation step."""

    step_id: str                # e.g., "accounts_receivable"
    level: int                  # Execution level (1, 2, 3...)

    # Value
    value: Any                  # Computed value (float, list, dict)
    value_type: str             # "scalar", "series", "table"

    # Source tracing
    source_query: str | None    # SQL that produced this value
    source_rows: int | None     # How many rows contributed

    # For derived steps
    formula_used: str | None    # e.g., "(ar / revenue) * days"
    inputs_used: dict | None    # {"ar": 100000, "revenue": 60000}
```

### 3. Persistence Schema

```sql
-- Generated calculation code (LLM-generated, cached)
CREATE TABLE generated_calculation_code (
    code_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    graph_version TEXT NOT NULL,
    schema_mapping_id TEXT NOT NULL,

    -- Generated code
    generated_sql TEXT NOT NULL,           -- The SQL to execute
    generated_python TEXT,                  -- Optional Python for complex logic

    -- Generation metadata
    llm_model TEXT NOT NULL,                -- e.g., "claude-3-opus"
    llm_prompt_hash TEXT NOT NULL,          -- Hash of prompt for reproducibility
    generation_timestamp TIMESTAMP NOT NULL,

    -- Validation
    is_validated BOOLEAN DEFAULT FALSE,
    validation_errors JSONB,

    -- Unique constraint: one code per (graph, version, schema)
    UNIQUE (graph_id, graph_version, schema_mapping_id)
);

-- Calculation executions
CREATE TABLE calculation_executions (
    execution_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    graph_version TEXT NOT NULL,

    -- Links to other entities
    schema_mapping_id TEXT REFERENCES schema_mappings(mapping_id),
    filter_execution_id TEXT REFERENCES filtering_executions(execution_id),
    generated_code_id TEXT REFERENCES generated_calculation_code(code_id),  -- NEW

    -- Parameters
    parameters JSONB NOT NULL,

    -- Results
    output_value DOUBLE,
    output_unit TEXT,
    interpretation TEXT,

    -- Reproducibility
    calculation_hash TEXT NOT NULL,
    executed_at TIMESTAMP NOT NULL,

    -- Full trace
    step_results JSONB NOT NULL
);

-- Index for finding executions by hash (reproducibility check)
CREATE INDEX idx_calc_hash ON calculation_executions(calculation_hash);

-- Index for finding latest execution of a graph
CREATE INDEX idx_calc_graph_time ON calculation_executions(graph_id, executed_at DESC);

-- Index for finding generated code by graph + schema
CREATE INDEX idx_gen_code_lookup ON generated_calculation_code(graph_id, graph_version, schema_mapping_id);
```

---

## Key Flows

### Flow 1: Execute Calculation (First Time - Agent-Based)

```
User: "Calculate DSO for March 2025"
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Calculation Graph (Rich Context for Agent)               │
│    - Parse dso_calculation_graph.yaml (714 lines!)              │
│    - Includes: accounting notes, DATEV mappings, validation     │
│    - Includes: interpretation guides, industry benchmarks       │
│    - This is CONTEXT for the agent, not executable code         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Check Generated Code Cache                                    │
│    - Lookup: (graph_id="dso", graph_version="1.0",              │
│              schema_mapping_id="mapping-xyz")                    │
│    - If found AND valid → Skip to step 5 (execute cached code)  │
│    - If not found → Continue to step 3 (generate code)          │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼ (Cache miss - first time for this schema)
┌─────────────────────────────────────────────────────────────────┐
│ 3. Calculation Agent: Analyze Schema + Generate Code             │
│    LLM receives:                                                 │
│    - Graph specification (accounting context, field definitions) │
│    - Actual data schema (table: clean_buchungen,                │
│      columns: Konto, Betrag, Buchungsdatum, Kontenart)          │
│    - Sample data rows                                            │
│    - Schema mapping context                                      │
│                                                                  │
│    LLM generates:                                                │
│    - SQL to extract accounts_receivable:                         │
│      SELECT SUM(Betrag) FROM clean_buchungen                    │
│      WHERE Konto IN (1200, 1400, 1576)                          │
│      AND Buchungsdatum = (SELECT MAX(Buchungsdatum)...)         │
│    - SQL to extract revenue (similar)                           │
│    - Final DSO calculation SQL                                   │
│                                                                  │
│    Agent validates:                                              │
│    - SQL syntax is correct                                       │
│    - Referenced columns exist                                    │
│    - Types are compatible                                        │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Persist Generated Code                                        │
│    - Store in generated_calculation_code table                   │
│    - Link to graph_id, graph_version, schema_mapping_id         │
│    - Record LLM model and prompt hash for reproducibility       │
│    - Code is now cached for future executions                   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Execute Generated Code (Deterministic)                        │
│    - Run the SQL queries in order (level by level)              │
│    - Capture intermediate results                                │
│    - Validate results against graph constraints                  │
│    - This step is fully deterministic - no LLM involved         │
│                                                                  │
│    Results:                                                      │
│    ├── accounts_receivable = 100,000                            │
│    ├── revenue = 60,000                                          │
│    ├── days_in_period = 30                                       │
│    └── dso = (100,000 / 60,000) * 30 = 50.0 days               │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Persist Execution Results                                     │
│    - Save CalculationExecution with all StepResults             │
│    - Link to generated_code_id for full traceability            │
│    - Generate calculation_hash for reproducibility               │
│    - Return execution_id                                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
Result: DSO = 50.0 days (CONCERNING)
        execution_id: "exec-abc123"
        generated_code_id: "code-def456"  ← NEW: link to generated code
        calculation_hash: "7f3a9c..."
```

**Key Difference from Old Design:**

The old design assumed graphs could be directly executed. This was WRONG because:
- `standard_field: "accounts_receivable"` is abstract
- Actual data might have `Konto` column with account codes
- Only an LLM can interpret "accounts 1200, 1400 in SKR03 = AR"

The new design separates:
1. **Graph = Specification**: What to calculate, with accounting context
2. **Agent = Interpreter**: LLM that generates concrete code
3. **Code = Execution**: Deterministic execution of generated SQL/Python

### Flow 2: Answer Question (Using Cached Execution)

```
User: "What is our DSO?"
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Q&A Agent Interprets Question                                 │
│    - Identifies metric: DSO                                      │
│    - Identifies period: current (latest)                         │
│    - No need to re-calculate if recent execution exists          │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Check for Recent Execution                                    │
│    SELECT * FROM calculation_executions                          │
│    WHERE graph_id = 'dso'                                        │
│    AND executed_at > NOW() - INTERVAL '1 day'                   │
│    ORDER BY executed_at DESC LIMIT 1                            │
│                                                                  │
│    → Found: execution_id = "exec-abc123"                        │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Return Cached Result                                          │
│    - Output: 50.0 days                                          │
│    - Interpretation: "CONCERNING"                                │
│    - Hash: "7f3a9c..." (for verification)                       │
│    - Link to visual explorer for breakdown                       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
Agent: "Your DSO is 50.0 days (CONCERNING).
        This means on average it takes 50 days to collect payment.

        Breakdown:
        - Accounts Receivable: €100,000
        - Monthly Revenue: €60,000
        - Period: 30 days

        [View calculation details →]"
```

### Flow 3: Visual Exploration

```
User clicks "View calculation details"
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ React Flow Visualizer                                            │
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
│  │ AR Balance │────▶│            │────▶│    DSO     │          │
│  │ €100,000   │     │  Formula   │     │  50 days   │          │
│  │ [▼ expand] │     │ (AR/Rev)×D │     │ CONCERNING │          │
│  └────────────┘     └────────────┘     └────────────┘          │
│        │                  ▲                                      │
│        │            ┌─────┴─────┐                               │
│  ┌─────┴─────┐     │           │                                │
│  │ Revenue   │─────┘     ┌─────┴─────┐                         │
│  │ €60,000   │           │ Days: 30  │                         │
│  │ [▼ expand]│           │ (constant)│                         │
│  └───────────┘           └───────────┘                          │
│                                                                  │
│  [Expand AR] → Shows:                                           │
│    - Source table: clean_transactions                           │
│    - Account codes: 1200, 1400                                  │
│    - Transaction count: 1,247                                   │
│    - Top customers by AR...                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration with Filtering

### Filtering Influences Calculations

```
              Filtering                          Calculation
              ────────                          ───────────

┌─────────────────────────────────────────────────────────────────┐
│ FilteringRecommendation                                          │
│                                                                  │
│ scope_filters:                                                   │
│   - column: "transaction_date"                                   │
│     condition: "BETWEEN '2025-01-01' AND '2025-03-31'"          │
│     reason: "Q1 2025 analysis"                                   │
│     impacts: ["dso", "cash_runway"]  ◀── Links to calculations  │
│                                                                  │
│ quality_filters:                                                 │
│   - column: "amount"                                             │
│     condition: "amount > 0"                                      │
│     reason: "Invalid negative amounts"                           │
│     impacts: ["all"]                                             │
│                                                                  │
│ calculation_impacts:   ◀── Tells users which calcs are affected │
│   - calc_id: "dso"                                              │
│     affected_steps: ["accounts_receivable", "revenue"]          │
│     severity: "high"                                             │
│     reason: "Date filter changes period scope"                   │
└─────────────────────────────────────────────────────────────────┘
          │
          │ Applied to create
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ FilteringExecution                                               │
│   execution_id: "filter-xyz"                                     │
│   clean_view: "clean_transactions"                               │
│   quarantine_table: "quarantine_transactions"                    │
│   clean_rows: 45,000                                            │
│   quarantined_rows: 523                                          │
└─────────────────────────────────────────────────────────────────┘
          │
          │ Used by
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ CalculationExecution                                             │
│   execution_id: "calc-abc"                                       │
│   filter_execution_id: "filter-xyz"  ◀── Traceable link         │
│   graph_id: "dso"                                                │
│   output_value: 50.0                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Bi-directional Relationship

1. **Filtering → Calculation**: Filter execution provides clean data
2. **Calculation → Filtering**: Calculation graph defines which columns need filtering

```yaml
# In dso_calculation_graph.yaml
dependencies:
  accounts_receivable:
    source:
      standard_field: "accounts_receivable"

    # NEW: Filter hints for the filtering system
    filter_hints:
      scope_relevant: true  # Date filters affect this
      quality_checks:
        - "value >= 0"      # Should not be negative
        - "value IS NOT NULL"
```

---

## Reproducibility Guarantees

### Calculation Hash Components

```python
def generate_calculation_hash(
    graph_id: str,
    graph_version: str,
    parameters: dict,
    schema_mapping_id: str,
    filter_execution_id: str,
    data_snapshot_time: datetime,
) -> str:
    """Generate hash that uniquely identifies a calculation."""

    hash_input = {
        "graph": f"{graph_id}@{graph_version}",
        "params": sorted(parameters.items()),
        "mapping": schema_mapping_id,
        "filters": filter_execution_id,
        "data_as_of": data_snapshot_time.isoformat(),
    }

    return hashlib.sha256(
        json.dumps(hash_input, sort_keys=True).encode()
    ).hexdigest()[:16]
```

### Same Hash = Same Result

```
If calculation_hash matches:
  - Same calculation graph version
  - Same parameters
  - Same schema mapping
  - Same filter execution
  - Same data snapshot

  → GUARANTEED identical output

User can verify: "This DSO of 50 days was calculated using
                  hash 7f3a9c... - you can reproduce it."
```

---

## Implementation Phases

### Phase B.1: Graph Loader and Models (Foundation) ✅
- [x] `TransformationGraph` loader from YAML
- [x] `GraphExecution` persistence model
- [x] Hash generation for reproducibility
- [x] Basic execution trace
- [x] React Flow export for visualization
- [x] Delete incorrect `executor.py` (direct execution was wrong)

### Phase B.2: Unified Graph Agent (Core - LLM-Powered)
- [ ] `GraphAgent` class that handles ALL graph types (filters + metrics)
- [ ] LLM prompt construction with:
  - Graph specification (accounting context, field definitions)
  - Actual data schema (columns, types, samples)
  - DATEV/industry-specific context from graph
- [ ] SQL generation from LLM response
- [ ] SQL validation (syntax, column existence)
- [ ] SQL execution in DuckDB
- [ ] Result capture and step tracing
- [ ] `GeneratedCode` persistence model

### Phase B.3: Generated SQL Caching
- [ ] Cache lookup by (graph_id, version, schema_mapping_id)
- [ ] Cache hit → skip LLM, execute cached SQL
- [ ] Cache miss → generate, validate, cache, execute
- [ ] Cache invalidation on graph/schema changes
- [ ] Force regeneration option

### Phase B.4: Filter Integration
- [ ] Execute on filtered views (clean_*)
- [ ] Track filter_execution_id in results
- [ ] calculation_impacts in FilteringRecommendation

### Phase B.5: Visualization Export
- [x] Export execution trace as JSON for React Flow
- [ ] Include drill-down data (transactions, accounts)
- [ ] Link from Q&A responses to visualizer
- [ ] Show generated SQL in visualization

### Phase B.6: Q&A Integration
- [ ] Query recent executions instead of re-calculating
- [ ] Answer "how was this calculated?" from trace
- [ ] Support "recalculate with different parameters"
- [ ] Show generated SQL when explaining calculations

---

## Design Decisions

### 1. Historical vs Current Values (No General Cache)

**Key Insight**: Financial data has two distinct states:
- **Historical (Closed)**: Immutable - "DSO for March 2025" once March is closed
- **Current (Open)**: Always fresh - "DSO for current period" always recalculates

```
┌─────────────────────────────────────────────────────────────────┐
│                    Period State Model                            │
│                                                                  │
│   Jan 2025    Feb 2025    Mar 2025    Apr 2025 (current)        │
│   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────────────┐          │
│   │CLOSED│    │CLOSED│    │CLOSED│    │    OPEN      │          │
│   │──────│    │──────│    │──────│    │──────────────│          │
│   │DSO=45│    │DSO=48│    │DSO=50│    │ DSO=? (calc) │          │
│   │FINAL │    │FINAL │    │FINAL │    │ on demand    │          │
│   └──────┘    └──────┘    └──────┘    └──────────────┘          │
│                                                                  │
│   Stored once,              │           Always recalculated      │
│   never changes             │           from current data        │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
@dataclass
class CalculationExecution:
    period: str                    # "2025-03"
    period_status: str             # "closed" | "open"
    is_final: bool                 # True if period is closed

    # For closed periods: stored result is authoritative
    # For open periods: result is point-in-time snapshot
```

**Query Behavior**:
- "What was DSO in March?" → Return stored final value (no recalc)
- "What is current DSO?" → Always recalculate from live data
- "What is DSO for April?" (open period) → Recalculate, store as non-final

### 2. Multi-Period = Separate Executions

Trend analysis (e.g., "DSO over last 12 months") = 12 separate executions.

```
User: "Show me DSO trend for 2024"

System executes:
  ├── execute_graph("dso", period="2024-01") → 45 days (final)
  ├── execute_graph("dso", period="2024-02") → 48 days (final)
  ├── execute_graph("dso", period="2024-03") → 50 days (final)
  │   ... (9 more)
  └── execute_graph("dso", period="2024-12") → 52 days (final)

Returns: [
  {"period": "2024-01", "value": 45, "final": true},
  {"period": "2024-02", "value": 48, "final": true},
  ...
]
```

**Benefits**:
- Simple mental model
- Each period independently traceable
- Closed periods never recalculated
- Open period always fresh

### 3. Prototype Integration Strategy

The `prototypes/calculation-graphs/` code provides:
- YAML structure for calculation graphs ✓ (keep format)
- Graph loader and validator ✓ (adapt to our architecture)
- Step executor pattern ✓ (reimplement in our style)
- Web visualizer ✓ (keep as-is, update data export)

**Approach**: Generate new code that fits `src/dataraum_context/` architecture.
Don't preserve prototype code - use it as reference only.

```
prototypes/calculation-graphs/     →    src/dataraum_context/calculations/
├── graph_loader.py               →    ├── graph_loader.py (new)
├── graph_executor.py             →    ├── executor.py (new)
├── dso_calculation_graph.yaml    →    config/calculations/dso.yaml
└── web_visualizer/               →    (keep separate, update JSON export)
```

---

## Success Criteria

- [ ] **Agent generates code**: LLM interprets graph + schema → executable SQL
- [ ] **Code is cached**: Same (graph, schema) → reuse generated code
- [ ] **Execution is deterministic**: Generated SQL runs without LLM
- [ ] Closed periods: stored once, never recalculated
- [ ] Open periods: always fresh from live data
- [ ] Same parameters → Same hash → Same result (reproducibility)
- [ ] Every number traceable to source transactions AND generated code
- [ ] Visual graph shows all calculation steps AND generated SQL
- [ ] Multi-period trends = separate executions per period
- [ ] Filter changes clearly show impact on calculations

---

## Next Steps

1. ~~**Create module structure**: `src/dataraum_context/graphs/`~~ ✅ Done
2. ~~**Implement graph loader**: Parse YAML, validate structure~~ ✅ Done
3. ~~**Export for visualization**: JSON format for React Flow~~ ✅ Done
4. ~~**Delete executor.py**: Direct execution approach was wrong~~ ✅ Done
5. **Implement Graph Agent**: Unified LLM-powered SQL generator
   - Create `src/dataraum_context/graphs/agent.py`
   - Build prompt with graph context + data schema
   - Parse LLM response into SQL
   - Validate generated SQL
   - Execute SQL and capture results
   - Cache generated SQL for reuse
6. **Add generated code persistence**: `GeneratedCode` SQLAlchemy model
7. **Integrate with filtering**: Link to `filter_execution_id`

---

## Current State Notes

**What exists** (`src/dataraum_context/graphs/`):
- `models.py` - Graph and execution data models ✅
- `loader.py` - YAML graph loader ✅
- `export.py` - React Flow visualization export ✅
- `persistence.py` - SQLAlchemy persistence ✅
- `agent.py` - **NEW**: Unified Graph Agent (LLM-powered SQL generation)

**Architecture**:
- NO separate executor - the agent handles everything
- Agent generates SQL, caches it, executes it
- Same approach for ALL graph types (filters and metrics)

**Two types of graphs exist**:
1. `config/calculation_graphs/` - Rich, detailed (714 lines for DSO)
   - Accounting notes, DATEV mappings, industry benchmarks
   - This is CONTEXT for the LLM agent
2. `config/graphs/` - Simplified (for quick prototyping)
   - Should be consolidated with detailed graphs
