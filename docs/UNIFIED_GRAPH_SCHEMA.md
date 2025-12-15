# Unified Transformation Graph Schema

## Vision

One schema to define both **filters** and **calculations**. Same YAML format, same executor, same visualizer.

```
┌─────────────────────────────────────────────────────────────────┐
│                 UNIFIED TRANSFORMATION GRAPHS                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Filter Graph │  │ Filter Graph │  │  Calc Graph  │          │
│  │ (technical)  │  │ (business)   │  │    (DSO)     │          │
│  │              │  │              │  │              │          │
│  │ source:      │  │ source:      │  │ source:      │          │
│  │ "system"     │  │ "user"       │  │ "system"     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────┬────────┘                 │                   │
│                  │                          │                   │
│                  ▼                          ▼                   │
│         ┌────────────────┐         ┌────────────────┐          │
│         │ Filtered Data  │────────▶│ Metric Result  │          │
│         └────────────────┘         └────────────────┘          │
│                  │                          │                   │
│                  └────────────┬─────────────┘                   │
│                               ▼                                 │
│                    ┌─────────────────────┐                     │
│                    │  Unified Visualizer │                     │
│                    │  - Step-by-step     │                     │
│                    │  - Drill-down       │                     │
│                    │  - Lineage          │                     │
│                    └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Schema

### Graph Header

```yaml
# Every transformation graph starts with this header
graph_id: "unique_identifier"
graph_type: "filter" | "metric"
version: "1.0"

metadata:
  name: "Human Readable Name"
  description: "What this graph does"
  category: "quality" | "scope" | "working_capital" | "profitability" | "liquidity"
  source: "system" | "user" | "llm"
  created_by: "system" | "user@email.com" | "claude-3"
  created_at: "2025-01-15"
  tags: ["finance", "ar", "quality"]
```

### Output Definition

```yaml
# For filter graphs
output:
  type: "classification"
  categories:
    clean: "Row passes all checks, include in analysis"
    exclude: "Row outside scope, skip silently"
    quarantine: "Row has quality issues, save for review"
    flag: "Row included but marked for attention"

# For metric graphs
output:
  type: "scalar" | "series" | "table"
  metric_id: "dso"
  unit: "days" | "currency" | "ratio" | "count"
  decimal_places: 1
```

### Dependencies (Unified)

```yaml
dependencies:

  # ═══════════════════════════════════════════════════════════
  # STEP TYPES (work for both filter and metric graphs)
  # ═══════════════════════════════════════════════════════════

  # Type 1: DATA EXTRACTION - Pull values from source
  step_name:
    level: 1
    type: "extract"
    source:
      table: "transactions"           # Or use standard_field for abstraction
      column: "amount"
      standard_field: "revenue"       # Abstract field, resolved by schema mapping
    aggregation: "sum" | "avg" | "min" | "max" | "count" | "end_of_period"

  # Type 2: CONSTANT - Fixed or parameterized value
  step_name:
    level: 1
    type: "constant"
    value: 30
    parameter: "days_in_period"       # User can override

  # Type 3: PREDICATE - Boolean condition (for filters)
  step_name:
    level: 1
    type: "predicate"
    condition: "amount > 0"
    on_false: "quarantine"            # What to do when condition fails
    reason: "Amount must be positive"

  # Type 4: FORMULA - Calculate derived value (for metrics)
  step_name:
    level: 2
    type: "formula"
    expression: "(accounts_receivable / revenue) * days_in_period"
    depends_on: ["accounts_receivable", "revenue", "days_in_period"]

  # Type 5: COMPOSITE - Combine multiple steps
  step_name:
    level: 3
    type: "composite"
    logic: "step_a AND step_b AND step_c"   # For filters
    # Or for complex metric logic:
    expression: "CASE WHEN x > 0 THEN a ELSE b END"
```

---

## Complete Examples

### Example 1: Technical Quality Filter (System)

```yaml
graph_id: "technical_quality"
graph_type: "filter"
version: "1.0"

metadata:
  name: "Technical Data Quality"
  description: "System-defined technical guardrails for data quality"
  category: "quality"
  source: "system"
  tags: ["technical", "quality", "guardrail"]

output:
  type: "classification"
  categories:
    clean: "Passes all technical checks"
    quarantine: "Technical quality issue"

dependencies:

  # Level 1: Atomic technical checks
  amount_not_null:
    level: 1
    type: "predicate"
    condition: "amount IS NOT NULL"
    on_false: "quarantine"
    reason: "Missing amount value"
    severity: "critical"

  amount_numeric:
    level: 1
    type: "predicate"
    condition: "TRY_CAST(amount AS DOUBLE) IS NOT NULL"
    on_false: "quarantine"
    reason: "Amount is not a valid number"
    severity: "critical"

  date_valid:
    level: 1
    type: "predicate"
    condition: "transaction_date IS NOT NULL AND transaction_date <= CURRENT_DATE"
    on_false: "quarantine"
    reason: "Invalid or future date"
    severity: "critical"

  account_reference:
    level: 1
    type: "predicate"
    condition: "account_id IS NOT NULL"
    on_false: "quarantine"
    reason: "Missing account reference"
    severity: "high"

  # Level 2: Composite classification
  technical_clean:
    level: 2
    type: "composite"
    logic: "amount_not_null AND amount_numeric AND date_valid AND account_reference"
    output_step: true  # This is the final classification
```

### Example 2: Business Scope Filter (User)

```yaml
graph_id: "q1_2025_external_ar"
graph_type: "filter"
version: "1.0"

metadata:
  name: "Q1 2025 External AR Scope"
  description: "User-defined scope for Q1 2025 external accounts receivable analysis"
  category: "scope"
  source: "user"
  created_by: "analyst@company.com"
  tags: ["ar", "q1-2025", "external"]

output:
  type: "classification"
  categories:
    clean: "In scope for analysis"
    exclude: "Out of scope"
    flag: "In scope but needs attention"

parameters:
  period_start:
    type: "date"
    default: "2025-01-01"
  period_end:
    type: "date"
    default: "2025-03-31"
  exclude_intercompany:
    type: "boolean"
    default: true

dependencies:

  # Level 1: Scope boundaries
  in_period:
    level: 1
    type: "predicate"
    condition: "transaction_date BETWEEN {period_start} AND {period_end}"
    on_false: "exclude"
    reason: "Outside Q1 2025 period"

  is_ar_account:
    level: 1
    type: "predicate"
    condition: "account_type IN ('accounts_receivable', 'trade_receivable')"
    on_false: "exclude"
    reason: "Not an AR account"

  not_intercompany:
    level: 1
    type: "predicate"
    condition: "intercompany_flag = FALSE OR intercompany_flag IS NULL"
    on_false: "exclude"
    reason: "Intercompany transaction excluded"
    enabled: "{exclude_intercompany}"  # Conditional on parameter

  # Level 2: Business flags
  large_balance:
    level: 2
    type: "predicate"
    condition: "amount > 100000"
    on_false: "clean"  # Pass through
    on_true: "flag"    # Flag for attention
    reason: "Large balance - verify with customer"

  # Level 3: Final scope
  in_scope:
    level: 3
    type: "composite"
    logic: "in_period AND is_ar_account AND not_intercompany"
    output_step: true
```

### Example 3: DSO Calculation (System)

```yaml
graph_id: "dso"
graph_type: "metric"
version: "1.0"

metadata:
  name: "Days Sales Outstanding"
  description: "Average days to collect payment after sale"
  category: "working_capital"
  source: "system"
  tags: ["ar", "collection", "working-capital"]

output:
  type: "scalar"
  metric_id: "dso"
  unit: "days"
  decimal_places: 1

# Link to required filter graphs
requires_filters:
  - graph_id: "technical_quality"
    required: true
  - graph_id: "q1_2025_external_ar"  # Or any scope filter with AR focus
    required: true

parameters:
  days_in_period:
    type: "integer"
    default: 30
    options: [30, 90, 365]
    description: "Analysis period length"

dependencies:

  # Level 1: Extract base data
  accounts_receivable:
    level: 1
    type: "extract"
    source:
      standard_field: "accounts_receivable"
      statement: "balance_sheet"
    aggregation: "end_of_period"
    validation:
      - condition: "value >= 0"
        severity: "error"
        message: "AR cannot be negative"

  revenue:
    level: 1
    type: "extract"
    source:
      standard_field: "revenue"
      statement: "income_statement"
    aggregation: "sum"
    validation:
      - condition: "value > 0"
        severity: "error"
        message: "Revenue must be positive for DSO"

  days_in_period:
    level: 1
    type: "constant"
    parameter: "days_in_period"
    default: 30

  # Level 2: Calculate DSO
  dso:
    level: 2
    type: "formula"
    expression: "(accounts_receivable / revenue) * days_in_period"
    depends_on: ["accounts_receivable", "revenue", "days_in_period"]
    output_step: true
    validation:
      - condition: "value >= 0 AND value <= 365"
        severity: "warning"
        message: "DSO outside typical range"

interpretation:
  ranges:
    - min: 0
      max: 30
      label: "EXCELLENT"
      description: "Very efficient collection"
    - min: 31
      max: 45
      label: "GOOD"
      description: "Strong collection performance"
    - min: 46
      max: 60
      label: "CONCERNING"
      description: "Review collection processes"
    - min: 61
      max: 90
      label: "POOR"
      description: "Significant working capital tied up"
    - min: 91
      max: 999
      label: "CRITICAL"
      description: "Urgent intervention required"
```

---

## Execution Model

### Execution Order

```
1. Load all required graphs (filters + metrics)
2. Build combined dependency DAG
3. Execute filters first (all filter graphs)
4. Execute metrics on filtered data (all metric graphs)
5. Persist all step results with traces
```

### Unified Executor

```python
class TransformationGraphExecutor:
    """Executes both filter and metric graphs with same engine."""

    async def execute(
        self,
        graph: TransformationGraph,
        parameters: dict,
        data_context: DataContext,
    ) -> GraphExecution:

        execution = GraphExecution(
            execution_id=str(uuid4()),
            graph_id=graph.graph_id,
            graph_type=graph.graph_type,
            graph_version=graph.version,
            parameters=parameters,
        )

        # Execute level by level
        for level, steps in graph.get_execution_order():
            for step in steps:
                result = await self._execute_step(step, execution, data_context)
                execution.step_results.append(result)

        # Compute final output
        if graph.graph_type == "filter":
            execution.output = self._compute_classification(execution)
        else:
            execution.output = self._compute_metric(execution)

        return execution

    async def _execute_step(
        self,
        step: GraphStep,
        execution: GraphExecution,
        data_context: DataContext,
    ) -> StepResult:

        match step.type:
            case "extract":
                value = await self._extract_data(step, data_context)
            case "constant":
                value = self._get_constant(step, execution.parameters)
            case "predicate":
                value = await self._evaluate_predicate(step, data_context)
            case "formula":
                value = self._calculate_formula(step, execution.step_results)
            case "composite":
                value = self._combine_steps(step, execution.step_results)

        return StepResult(
            step_id=step.step_id,
            level=step.level,
            type=step.type,
            value=value,
            # Trace info for visualization
            inputs_used={...},
            source_query=...,
            rows_affected=...,
        )
```

---

## Persistence Model

```python
class GraphExecution(Base):
    """Unified execution record for both filters and metrics."""

    __tablename__ = "graph_executions"

    execution_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Graph identification
    graph_id: Mapped[str] = mapped_column(String, nullable=False)
    graph_type: Mapped[str] = mapped_column(String, nullable=False)  # "filter" | "metric"
    graph_version: Mapped[str] = mapped_column(String, nullable=False)

    # Source tracking
    source: Mapped[str] = mapped_column(String, nullable=False)  # "system" | "user" | "llm"

    # Execution context
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    period: Mapped[str | None] = mapped_column(String, nullable=True)
    is_period_final: Mapped[bool] = mapped_column(Boolean, default=False)

    # Results
    output_value: Mapped[Any] = mapped_column(JSON, nullable=False)  # Metric value or classification counts
    output_interpretation: Mapped[str | None] = mapped_column(String, nullable=True)

    # Traceability
    step_results: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    execution_hash: Mapped[str] = mapped_column(String, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Links
    depends_on_executions: Mapped[list[str]] = mapped_column(JSON, default=list)  # Filter executions this metric used


class StepResultRecord(Base):
    """Individual step results for drill-down."""

    __tablename__ = "step_results"

    result_id: Mapped[str] = mapped_column(String, primary_key=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("graph_executions.execution_id"))

    step_id: Mapped[str] = mapped_column(String, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String, nullable=False)

    # Value (polymorphic)
    value_scalar: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_boolean: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    value_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Trace
    inputs_used: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_query: Mapped[str | None] = mapped_column(String, nullable=True)
    rows_affected: Mapped[int | None] = mapped_column(Integer, nullable=True)
```

---

## Visualization Export

Same JSON format for React Flow, regardless of graph type:

```json
{
  "execution_id": "exec-123",
  "graph_id": "dso",
  "graph_type": "metric",

  "nodes": [
    {
      "id": "accounts_receivable",
      "type": "extractNode",
      "data": {
        "label": "Accounts Receivable",
        "value": 100000,
        "unit": "currency",
        "level": 1,
        "source": "balance_sheet.accounts_receivable",
        "expandable": true,
        "drilldown": {
          "query": "SELECT * FROM clean_transactions WHERE account_type = 'AR'",
          "row_count": 1247
        }
      }
    },
    {
      "id": "revenue",
      "type": "extractNode",
      "data": {
        "label": "Revenue",
        "value": 60000,
        "level": 1
      }
    },
    {
      "id": "dso",
      "type": "formulaNode",
      "data": {
        "label": "DSO",
        "value": 50.0,
        "unit": "days",
        "level": 2,
        "formula": "(AR / Revenue) × Days",
        "interpretation": "CONCERNING",
        "inputs": {
          "accounts_receivable": 100000,
          "revenue": 60000,
          "days_in_period": 30
        }
      }
    }
  ],

  "edges": [
    {"source": "accounts_receivable", "target": "dso"},
    {"source": "revenue", "target": "dso"},
    {"source": "days_in_period", "target": "dso"}
  ]
}
```

For filter graphs, same structure but different node types:

```json
{
  "graph_type": "filter",

  "nodes": [
    {
      "id": "amount_valid",
      "type": "predicateNode",
      "data": {
        "label": "Amount Valid",
        "condition": "amount > 0",
        "pass_count": 45000,
        "fail_count": 47,
        "fail_action": "quarantine",
        "level": 1
      }
    }
  ]
}
```

---

## Directory Structure

```
config/
└── graphs/
    ├── filters/
    │   ├── system/
    │   │   ├── technical_quality.yaml
    │   │   ├── null_checks.yaml
    │   │   └── type_validation.yaml
    │   └── user/
    │       ├── q1_2025_scope.yaml
    │       └── exclude_test_data.yaml
    │
    └── metrics/
        ├── working_capital/
        │   ├── dso.yaml
        │   ├── dpo.yaml
        │   └── cash_conversion_cycle.yaml
        ├── liquidity/
        │   ├── cash_runway.yaml
        │   └── current_ratio.yaml
        └── profitability/
            ├── gross_margin.yaml
            └── operating_margin.yaml

src/dataraum_context/
└── graphs/
    ├── __init__.py
    ├── models.py          # TransformationGraph, GraphStep, StepResult, GraphExecution
    ├── loader.py          # Parse YAML, validate structure, build DAG
    ├── executor.py        # Unified execution engine (filters + metrics)
    ├── persistence.py     # Save/load executions to DB
    └── export.py          # JSON export for React Flow visualizer
```

### Module Responsibilities

| File | Responsibility |
|------|----------------|
| `models.py` | Pydantic models for graphs, steps, executions |
| `loader.py` | Parse YAML, validate schema, resolve dependencies, build execution DAG |
| `executor.py` | Execute any graph (filter or metric), produce traces |
| `persistence.py` | `GraphExecution` and `StepResultRecord` SQLAlchemy models, save/load |
| `export.py` | Convert execution traces to React Flow JSON format |

---

## Migration Path

Since no backward compatibility is needed:

1. **Delete** existing `quality/filtering/` module (replaced by unified graphs)
2. **Don't create** separate `calculations/` module (merged into graphs)
3. **Create** new `src/dataraum_context/graphs/` module with unified schema
4. **Adapt** calculation graph YAMLs from prototype to unified schema in `config/graphs/metrics/`
5. **Create** filter graph YAMLs in `config/graphs/filters/`
6. **Keep** web visualizer from prototype, update JSON export format
7. **Reference** prototype Python code for patterns, but generate fresh code

### What Gets Deleted

| Path | Reason |
|------|--------|
| `src/dataraum_context/quality/filtering/` | Replaced by `graphs/` |
| `prototypes/calculation-graphs/*.py` | Reference only, new code in `graphs/` |

### What Gets Kept/Adapted

| From | To | Notes |
|------|-----|-------|
| `prototypes/calculation-graphs/*.yaml` | `config/graphs/metrics/` | Adapt to unified schema |
| `prototypes/calculation-graphs/web_visualizer/` | Keep as-is | Update JSON export format |
| `config/filters/*.yaml` | `config/graphs/filters/user/` | Convert to graph format |

---

## Success Criteria

- [ ] Single YAML schema for filters and metrics
- [ ] Same executor handles both graph types
- [ ] Same visualizer displays both graph types
- [ ] Filter graphs can be system, user, or LLM sourced
- [ ] Metric graphs declare filter dependencies
- [ ] Step-by-step trace available for both
- [ ] Drill-down to source data works for both
