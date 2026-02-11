# Filtering Redesign

## Problem Statement

Current filtering has good foundations but lacks integration with calculations:
- `scope_filters` and `quality_filters` are well-separated
- `calculation_impacts` field exists but isn't populated systematically
- No bi-directional link between filters and calculation graphs
- Filter generation doesn't consider what calculations need

**Goal**: Tight integration where:
1. Filters know which calculations they affect
2. Calculations define what filters they need
3. Users can trace: "Why was this row excluded from my DSO calculation?"

---

## Design Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BI-DIRECTIONAL INTEGRATION                           │
│                                                                         │
│   Calculation Graphs                         Filter System              │
│   ──────────────────                         ─────────────              │
│                                                                         │
│   ┌─────────────────────┐                   ┌─────────────────────┐    │
│   │ dso.yaml            │                   │ FilterRecommendation │    │
│   │                     │                   │                      │    │
│   │ filter_requirements:│◀─────────────────▶│ calculation_impacts: │    │
│   │   - date_scope      │   references      │   - dso              │    │
│   │   - positive_amounts│                   │   - cash_runway      │    │
│   │                     │                   │                      │    │
│   └─────────────────────┘                   └─────────────────────┘    │
│            │                                          │                 │
│            │ defines                                  │ produces        │
│            ▼                                          ▼                 │
│   ┌─────────────────────┐                   ┌─────────────────────┐    │
│   │ Required Filters    │                   │ Applied Filters     │    │
│   │ (abstract)          │                   │ (concrete SQL)      │    │
│   │                     │                   │                      │    │
│   │ "need date filter   │      LLM/User     │ "transaction_date    │    │
│   │  for period scope"  │ ───────────────▶  │  BETWEEN ... AND ..." │    │
│   │                     │   generates       │                      │    │
│   └─────────────────────┘                   └─────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Filter Types (Unchanged)

```yaml
# Scope Filters: Define WHAT data to analyze
scope_filters:
  - type: "period"
    column: "transaction_date"
    condition: "BETWEEN '2025-01-01' AND '2025-03-31'"
    reason: "Q1 2025 analysis scope"

  - type: "entity"
    column: "company_id"
    condition: "= 'ACME-001'"
    reason: "Single entity analysis"

# Quality Filters: Define CLEAN data (remove bad rows)
quality_filters:
  - type: "validity"
    column: "amount"
    condition: "> 0"
    reason: "Invalid negative amounts detected"
    quarantine: true  # Move to quarantine, don't delete

  - type: "completeness"
    column: "account_id"
    condition: "IS NOT NULL"
    reason: "Missing account reference"
    quarantine: true
```

### 2. Calculation Graph Filter Requirements (NEW)

```yaml
# In config/calculations/dso.yaml

graph_id: "dso"
version: "1.0"

# NEW: What filters does this calculation need?
filter_requirements:

  # Required: Calculation cannot run without these
  required:
    - requirement_id: "period_scope"
      type: "scope"
      description: "Date range for the analysis period"
      applies_to:
        - dependency: "accounts_receivable"
          reason: "AR balance at period end"
        - dependency: "revenue"
          reason: "Revenue within period"
      hints:
        column_patterns: ["date", "transaction_date", "posting_date"]
        typical_conditions: ["BETWEEN", ">=", "<="]

    - requirement_id: "positive_amounts"
      type: "quality"
      description: "Amounts must be positive for valid calculation"
      applies_to:
        - dependency: "accounts_receivable"
        - dependency: "revenue"
      hints:
        column_patterns: ["amount", "value", "balance"]
        condition: "> 0"

  # Recommended: Improve accuracy but not blocking
  recommended:
    - requirement_id: "exclude_intercompany"
      type: "scope"
      description: "Exclude intercompany transactions for external DSO"
      applies_to:
        - dependency: "accounts_receivable"
      hints:
        column_patterns: ["intercompany", "ic_flag", "related_party"]
        condition: "= FALSE OR IS NULL"

dependencies:
  accounts_receivable:
    # ... existing definition ...

  revenue:
    # ... existing definition ...
```

### 3. Filter Recommendations with Calculation Links (ENHANCED)

```python
@dataclass
class ScopeFilter:
    """A scope filter that defines analysis boundaries."""

    filter_id: str
    column: str
    condition: str
    reason: str

    # NEW: Which calculations does this affect?
    affects_calculations: list[CalculationImpact]

    # NEW: Was this generated from a calculation requirement?
    source_requirement: str | None  # e.g., "dso.period_scope"


@dataclass
class CalculationImpact:
    """How a filter affects a specific calculation."""

    calculation_id: str        # e.g., "dso"
    affected_dependencies: list[str]  # e.g., ["accounts_receivable", "revenue"]
    impact_type: str           # "scope_change" | "data_quality" | "exclusion"
    severity: str              # "critical" | "significant" | "minor"

    # Human-readable explanation
    explanation: str           # "Changes the date range for AR and revenue"


@dataclass
class FilteringRecommendation:
    """Complete filtering recommendation with calculation integration."""

    recommendation_id: str
    table_id: str

    # Filter categories
    scope_filters: list[ScopeFilter]
    quality_filters: list[QualityFilter]
    flags: list[DataFlag]

    # NEW: Aggregated calculation impacts
    calculation_impacts: dict[str, CalculationImpact]

    # NEW: Unmet requirements (calculations that need filters not yet defined)
    unmet_requirements: list[UnmetRequirement]

    # Metadata
    source: str  # "llm" | "user_rule" | "calculation_requirement"
    confidence: float
```

---

## Filter Generation Flow

### Step 1: Gather Calculation Requirements

```python
async def gather_filter_requirements(
    calculation_ids: list[str],
) -> list[FilterRequirement]:
    """
    Collect all filter requirements from requested calculations.

    Example:
        calculations = ["dso", "cash_runway"]
        requirements = await gather_filter_requirements(calculations)

        # Returns:
        # - dso.period_scope (required)
        # - dso.positive_amounts (required)
        # - cash_runway.period_scope (required)
        # - cash_runway.exclude_one_time (recommended)
    """
    requirements = []

    for calc_id in calculation_ids:
        graph = load_calculation_graph(calc_id)

        for req in graph.filter_requirements.required:
            requirements.append(FilterRequirement(
                source_calculation=calc_id,
                requirement=req,
                priority="required",
            ))

        for req in graph.filter_requirements.recommended:
            requirements.append(FilterRequirement(
                source_calculation=calc_id,
                requirement=req,
                priority="recommended",
            ))

    return deduplicate_requirements(requirements)
```

### Step 2: Generate Filters (LLM or User Rules)

```python
async def generate_filters_for_requirements(
    requirements: list[FilterRequirement],
    table_schema: TableSchema,
    existing_filters: list[Filter],
    llm_service: LLMService | None,
) -> FilteringRecommendation:
    """
    Generate concrete filters that satisfy calculation requirements.

    Process:
    1. Check which requirements are already satisfied by existing filters
    2. For unmet requirements, generate new filters (LLM or rules)
    3. Link each filter to the calculations it affects
    """

    # Check existing coverage
    satisfied, unmet = check_requirement_coverage(requirements, existing_filters)

    # Generate filters for unmet requirements
    new_filters = []

    for req in unmet:
        if llm_service:
            # LLM generates filter based on hints and schema
            filter_result = await generate_filter_with_llm(
                requirement=req,
                schema=table_schema,
                hints=req.hints,
                llm_service=llm_service,
            )
        else:
            # Fall back to rule-based generation
            filter_result = generate_filter_from_rules(req, table_schema)

        if filter_result:
            # Link to source calculation
            filter_result.source_requirement = f"{req.source_calculation}.{req.requirement_id}"
            filter_result.affects_calculations = [
                CalculationImpact(
                    calculation_id=req.source_calculation,
                    affected_dependencies=req.applies_to,
                    impact_type="scope_change" if req.type == "scope" else "data_quality",
                    severity="critical" if req.priority == "required" else "minor",
                )
            ]
            new_filters.append(filter_result)

    return FilteringRecommendation(
        scope_filters=[f for f in new_filters if f.type == "scope"],
        quality_filters=[f for f in new_filters if f.type == "quality"],
        calculation_impacts=aggregate_impacts(new_filters),
        unmet_requirements=[r for r in unmet if not any(f.source_requirement == f"{r.source_calculation}.{r.requirement_id}" for f in new_filters)],
    )
```

### Step 3: User Review with Calculation Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Filter Review Interface                               │
│                                                                          │
│  Scope Filters                                                          │
│  ─────────────                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ ☑ Period Filter                                                 │    │
│  │   Column: transaction_date                                      │    │
│  │   Condition: BETWEEN '2025-01-01' AND '2025-03-31'             │    │
│  │                                                                 │    │
│  │   Affects calculations:                                         │    │
│  │   • DSO - accounts_receivable, revenue (critical)              │    │
│  │   • Cash Runway - operating_expenses (critical)                │    │
│  │                                                                 │    │
│  │   [Edit] [Remove]                                               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Quality Filters                                                        │
│  ───────────────                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ ☑ Positive Amounts                                              │    │
│  │   Column: amount                                                │    │
│  │   Condition: > 0                                                │    │
│  │   Quarantine: 47 rows                                           │    │
│  │                                                                 │    │
│  │   Affects calculations:                                         │    │
│  │   • DSO - accounts_receivable (critical)                       │    │
│  │   • All amount-based calculations                               │    │
│  │                                                                 │    │
│  │   [Edit] [Remove] [View Quarantined]                           │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ⚠ Unmet Requirements                                                   │
│  ─────────────────────                                                  │
│  │ Cash Runway needs: exclude_one_time_items (recommended)         │    │
│  │ [Generate Filter] [Skip - Accept Risk]                          │    │
│                                                                          │
│  [Apply Filters] [Save as Template]                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Persistence Models

### FilterRequirementRecord (NEW)

```python
class FilterRequirementRecord(Base):
    """Tracks which filter requirements exist for calculations."""

    __tablename__ = "filter_requirements"

    requirement_id: Mapped[str] = mapped_column(String, primary_key=True)
    calculation_id: Mapped[str] = mapped_column(String, nullable=False)
    requirement_type: Mapped[str] = mapped_column(String, nullable=False)  # "scope" | "quality"
    priority: Mapped[str] = mapped_column(String, nullable=False)  # "required" | "recommended"

    description: Mapped[str] = mapped_column(String, nullable=False)
    applies_to_dependencies: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    hints: Mapped[dict] = mapped_column(JSON, nullable=False)

    # From which graph version
    graph_version: Mapped[str] = mapped_column(String, nullable=False)
```

### FilterCalculationLink (NEW)

```python
class FilterCalculationLink(Base):
    """Links applied filters to affected calculations."""

    __tablename__ = "filter_calculation_links"

    link_id: Mapped[str] = mapped_column(String, primary_key=True)

    # The filter
    filter_execution_id: Mapped[str] = mapped_column(
        ForeignKey("filtering_executions.execution_id"), nullable=False
    )
    filter_type: Mapped[str] = mapped_column(String, nullable=False)  # "scope" | "quality"
    filter_column: Mapped[str] = mapped_column(String, nullable=False)

    # The calculation affected
    calculation_id: Mapped[str] = mapped_column(String, nullable=False)
    affected_dependencies: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    # Impact assessment
    impact_type: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)

    # Source tracking
    source_requirement_id: Mapped[str | None] = mapped_column(String, nullable=True)
```

---

## Integration with Calculation Execution

### Execution Checks Filters

```python
async def execute_calculation(
    graph: CalculationGraph,
    parameters: dict,
    filter_execution_id: str,
    session: AsyncSession,
) -> CalculationExecution:
    """
    Execute calculation with filter validation.
    """

    # Step 1: Validate required filters are present
    filter_execution = await get_filter_execution(filter_execution_id, session)

    missing_required = check_required_filters(
        graph.filter_requirements.required,
        filter_execution.applied_filters,
    )

    if missing_required:
        raise MissingRequiredFiltersError(
            calculation=graph.graph_id,
            missing=missing_required,
            message=f"Cannot execute {graph.graph_id}: missing required filters {missing_required}"
        )

    # Step 2: Warn about missing recommended filters
    missing_recommended = check_required_filters(
        graph.filter_requirements.recommended,
        filter_execution.applied_filters,
    )

    warnings = []
    if missing_recommended:
        warnings.append(f"Recommended filters not applied: {missing_recommended}")

    # Step 3: Execute calculation
    result = await execute_graph_steps(graph, parameters, filter_execution)

    # Step 4: Record the filter-calculation link
    await record_filter_calculation_link(
        filter_execution_id=filter_execution_id,
        calculation_id=graph.graph_id,
        affected_filters=get_relevant_filters(filter_execution, graph),
        session=session,
    )

    return result
```

### Traceability Query

```python
async def explain_calculation_filters(
    execution_id: str,
    session: AsyncSession,
) -> FilterExplanation:
    """
    Explain which filters affected a calculation result.

    User asks: "Why is my DSO different from last month?"

    Returns:
        FilterExplanation with:
        - scope_filters applied and their impact
        - quality_filters applied and rows excluded
        - comparison with previous execution's filters
    """

    execution = await get_calculation_execution(execution_id, session)
    filter_links = await get_filter_links(execution.filter_execution_id, session)

    return FilterExplanation(
        calculation=execution.graph_id,
        period=execution.period,

        scope_filters=[
            {
                "filter": link.filter_column,
                "condition": link.condition,
                "affected": link.affected_dependencies,
                "impact": link.impact_type,
            }
            for link in filter_links if link.filter_type == "scope"
        ],

        quality_filters=[
            {
                "filter": link.filter_column,
                "condition": link.condition,
                "rows_excluded": link.rows_excluded,
                "affected": link.affected_dependencies,
            }
            for link in filter_links if link.filter_type == "quality"
        ],

        quarantined_summary=await get_quarantine_summary(
            execution.filter_execution_id, session
        ),
    )
```

---

## User Rules Integration

User-defined rules in YAML can also specify calculation impacts:

```yaml
# config/filters/user_rules.yaml

filtering_rules:
  - name: "exclude_test_transactions"
    priority: "OVERRIDE"

    match:
      column_patterns: ["description", "memo", "reference"]

    filter:
      condition: "NOT LIKE '%TEST%'"
      reason: "Exclude test transactions from analysis"

    # NEW: Explicit calculation impacts
    calculation_impacts:
      - calculation: "all"
        severity: "minor"
        reason: "Test data should not affect any metrics"

  - name: "intercompany_exclusion"
    priority: "EXTEND"

    match:
      column_patterns: ["intercompany", "ic_flag"]

    filter:
      condition: "= FALSE"
      reason: "Exclude intercompany for external metrics"

    calculation_impacts:
      - calculation: "dso"
        severity: "significant"
        reason: "Intercompany AR inflates DSO"
      - calculation: "dpo"
        severity: "significant"
        reason: "Intercompany AP inflates DPO"
```

---

## Period Handling (Simplified)

Period is a **parameter**, not a state machine (complex period open/close is future work).

```python
@dataclass
class FilterParameters:
    """Parameters that affect filter behavior."""

    # Period scope
    period_start: date
    period_end: date

    # For calculations that need it
    period_type: str = "monthly"  # "monthly" | "quarterly" | "annual"

    # Simple flag - no state machine
    # User/system sets this based on business rules
    is_period_closed: bool = False


async def apply_period_filter(
    params: FilterParameters,
    date_column: str,
) -> ScopeFilter:
    """Generate period scope filter from parameters."""

    return ScopeFilter(
        filter_id=f"period_{params.period_start}_{params.period_end}",
        column=date_column,
        condition=f"BETWEEN '{params.period_start}' AND '{params.period_end}'",
        reason=f"Analysis period: {params.period_start} to {params.period_end}",
        affects_calculations=[],  # Will be populated by requirement matching
    )
```

---

## Implementation Phases

### Phase C.1: Add Calculation Links to Existing Models
- [ ] Add `affects_calculations` to `ScopeFilter` and `QualityFilter`
- [ ] Add `source_requirement` field
- [ ] Add `unmet_requirements` to `FilteringRecommendation`
- [ ] Create `FilterCalculationLink` persistence model

### Phase C.2: Calculation Graph Filter Requirements
- [ ] Add `filter_requirements` section to YAML schema
- [ ] Update graph loader to parse requirements
- [ ] Implement `gather_filter_requirements()`

### Phase C.3: Filter Generation with Requirements
- [ ] Implement `generate_filters_for_requirements()`
- [ ] Update LLM prompt to use requirement hints
- [ ] Implement requirement coverage checking

### Phase C.4: Execution Integration
- [ ] Add filter validation to calculation execution
- [ ] Record filter-calculation links on execution
- [ ] Implement `explain_calculation_filters()`

### Phase C.5: User Rules Enhancement
- [ ] Add `calculation_impacts` to user rule schema
- [ ] Update rules merger to preserve impacts
- [ ] Validate impacts against known calculations

---

## Success Criteria

- [ ] Every filter knows which calculations it affects
- [ ] Every calculation defines its filter requirements
- [ ] Missing required filters block calculation execution
- [ ] Users can trace: filter → affected calculations → impacted values
- [ ] User rules can specify calculation impacts
- [ ] Filter review shows calculation context
