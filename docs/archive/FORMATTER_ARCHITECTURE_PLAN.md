# Formatter Architecture Plan

**Created:** 2025-12-14
**Updated:** 2025-12-15
**Status:** In Progress - Phase 7
**Related:** METRICS_ANALYSIS.md, quality/formatting/base.py

---

## Implementation Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Configuration Infrastructure | ✅ Done | `quality/formatting/config.py` |
| 2 | Statistical Formatter | ✅ Done | `quality/formatting/statistical.py` |
| 3 | Temporal Formatter | ✅ Done | `quality/formatting/temporal.py` |
| 4 | Topological Formatter | ✅ Done | `quality/formatting/topological.py` |
| 5 | Domain Formatter | ✅ Done | `quality/formatting/domain.py` |
| 6 | Calculation Graphs + Mapping | ✅ Done | `calculations/` package |
| 6b | Business Cycles Formatter | ✅ Done | `quality/formatting/business_cycles.py` |
| 7 | Filter Generation | 🔄 Next | Extend existing `quality/filtering/` |
| 8 | Integration | ⏳ Pending | Full pipeline wiring |

---

## Key Architectural Insights

### 1. Calculations Drive Quality Focus
The calculation graphs define what data matters. Quality issues in columns that feed DSO/Cash Runway are more critical than issues in unused columns.

### 2. Aggregation Lives in Calculation Graphs
Calculation graphs (e.g., `dso_calculation_graph.yaml`) define:
- Abstract fields (`revenue`, `accounts_receivable`)
- Aggregation method (`sum`, `end_of_period`)
- Validation rules

Schema mapping ONLY binds concrete columns to abstract fields. The LLM matcher does NOT determine aggregation - it just says "transactions.amount → revenue".

### 3. Origin Tables vs Target Schema
```
ORIGIN TABLES (raw)              TARGET SCHEMA (for calculations)
─────────────────────            ────────────────────────────────
transactions.amount        →     revenue = SUM(amount) GROUP BY period
ledger.ar_balance          →     accounts_receivable = END_OF_PERIOD(ar_balance)
```

- Filters apply to ORIGIN tables (data quality)
- Aggregations are defined in graphs, not mappings

### 4. Business Cycles Are Context for Filtering
Business cycles (AR, AP, Revenue cycles) are detected cross-table process flows. They provide context for:
- **Filter Generation** (primary): "This table is part of the AR cycle, prioritize quality"
- **Quality Assessment** (light): "AR cycle is incomplete, missing payments table"

Business cycles are NOT needed for schema mapping.

### 5. Scope vs Quality Filters
Two types of filters, potentially from same LLM call:
- **Scope filters**: Row selection for calculation (e.g., `WHERE type = 'sale'` for revenue)
- **Quality filters**: Data cleaning (e.g., `WHERE amount IS NOT NULL AND amount > 0`)

---

## Problem Statement

The quality formatting system needs to transform raw metrics into structured, interpretable context for LLM and human consumption. The core challenge is that **thresholds are not universal**:

- **Domain-specific:** Financial data has different acceptable thresholds than marketing data
- **Dataset-specific:** Company A may tolerate 5% nulls; Company B requires <1%
- **Context-dependent:** A VIF of 8 might be acceptable for exploratory analysis but problematic for regulatory reporting

Currently, `quality/formatting/base.py` provides `ThresholdConfig` with hardcoded defaults. The multicollinearity formatter in `quality/formatting/multicollinearity.py` uses these but without external configuration.

---

## Goals

1. **Configurable thresholds** that can be overridden per domain, dataset, or context
2. **Consistent formatter pattern** across all pillars (statistical, temporal, topological, domain)
3. **LLM-friendly output** with interpretations and recommendations
4. **Human-readable** context that explains "what" and "why"
5. **Extensibility** for new metrics and domains

---

## Current State Analysis

### What Exists

| Component | Status | Notes |
|-----------|--------|-------|
| `base.py` | ✅ Implemented | ThresholdConfig, CommonThresholds, interpretation utilities |
| `multicollinearity.py` | ✅ Implemented | Full formatter with VIF, CI, dependency groups |
| Statistical formatter | ❌ Missing | Benford, outliers, distribution |
| Temporal formatter | ❌ Missing | Staleness, gaps, seasonality |
| Topological formatter | ❌ Missing | Betti numbers, cycles, components |
| Domain formatter | ❌ Missing | Rule violations, financial checks |
| Configuration system | ❌ Missing | YAML-based threshold overrides |

### Metrics by Pillar (from METRICS_ANALYSIS.md)

| Pillar | Total Metrics | In Synthesis | For Context | Formatter Complexity |
|--------|---------------|--------------|-------------|---------------------|
| Statistical | 42 | 12 | 30 | HIGH - Many metric types |
| Topological | 18 | 4 | 14 | MEDIUM - Specialized interpretation |
| Temporal | 36 | 8 | 28 | MEDIUM - Time-based patterns |
| Correlation | 31 | 3 | 28 | LOW - Multicollinearity done |
| Domain | 10+ | 10+ | - | HIGH - Business rules |

---

## Proposed Architecture

### 1. Configuration Hierarchy

```
defaults (base.py)
    └── domain overrides (config/formatter_thresholds/{domain}.yaml)
        └── dataset overrides (inline or config/datasets/{dataset}.yaml)
            └── runtime overrides (API parameters)
```

**Resolution order:** Runtime > Dataset > Domain > Defaults

### 2. Configuration Schema

```yaml
# config/formatter_thresholds/financial.yaml
statistical:
  null_ratio:
    none: 0.001      # Financial: <0.1% nulls acceptable
    low: 0.01
    moderate: 0.05
    high: 0.1
    severe_above: 0.1

  outlier_ratio:
    none: 0.005      # Tighter for financial
    low: 0.02
    moderate: 0.05
    high: 0.1

  benford:
    p_value_threshold: 0.05
    chi_square_critical: 15.507  # df=8, alpha=0.05

temporal:
  staleness_days:
    none: 1          # Financial: daily updates expected
    low: 3
    moderate: 7
    high: 30
    severe_above: 90

  completeness_ratio:
    none: 0.999      # Very strict for financial
    low: 0.99
    moderate: 0.95
    high: 0.8

multicollinearity:
  vif:
    none: 1.0
    low_to_moderate: 2.5
    high: 5.0
    serious_above: 10.0

  condition_index:
    none: 10
    moderate: 30
    severe_above: 30

topological:
  orphaned_components:
    none: 0
    low: 1
    moderate: 3
    severe_above: 5

  anomalous_cycles:
    none: 0
    low: 1
    moderate: 2
    severe_above: 3

domain:
  balance_tolerance: 0.01      # 1 cent tolerance
  sign_compliance_threshold: 0.99
```

### 3. Formatter Interface

```python
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseFormatter(ABC):
    """Base class for all quality formatters."""

    def __init__(self, thresholds: ThresholdSet | None = None):
        """Initialize with optional threshold overrides."""
        self.thresholds = thresholds or self.default_thresholds()

    @abstractmethod
    def default_thresholds(self) -> ThresholdSet:
        """Return default threshold configuration."""
        ...

    @abstractmethod
    def format_for_llm(self, data: T) -> dict[str, Any]:
        """Transform raw data into LLM-friendly structure."""
        ...

    @abstractmethod
    def get_severity(self, data: T) -> str:
        """Determine overall severity from data."""
        ...

    @abstractmethod
    def get_interpretation(self, data: T) -> str:
        """Generate natural language interpretation."""
        ...

    @abstractmethod
    def get_recommendations(self, data: T) -> list[str]:
        """Generate actionable recommendations."""
        ...
```

### 4. Formatter Outputs

Each formatter should produce a consistent structure:

```python
{
    "{metric}_assessment": {
        "overall_severity": "moderate",  # none, low, moderate, high, severe, critical
        "summary": "Human-readable summary of findings",
        "interpretation": "What this means in business context",
        "metrics": {
            # Pillar-specific metrics with values and individual severities
        },
        "issues": [
            # Specific problems detected
        ],
        "recommendations": [
            # Actionable next steps
        ],
        "technical_details": {
            # Statistical evidence for advanced users
        }
    }
}
```

---

## Proposed Formatters

### 1. Statistical Quality Formatter

**Input:** `StatisticalQualityResult` (Pydantic model)

**Key metrics to format:**
- Benford's Law analysis (compliance, chi-square, p-value)
- Outlier detection (IQR ratio, IF score, examples)
- Distribution stability (KS test, shift detection)
- Basic completeness (null ratio, cardinality)

**Interpretation examples:**
- Benford violation: "First-digit distribution deviates significantly from Benford's Law (chi² = 45.2, p < 0.001). This pattern is unusual for naturally occurring financial data and may warrant investigation."
- High outliers: "8.3% of values are statistical outliers (>3 IQR from median). Sample outliers: $-15,234, $892,000. These may represent data entry errors or legitimate extreme values."

**Configuration needs:**
- Benford p-value threshold (varies by sample size)
- Outlier percentage tolerance (domain-dependent)
- Distribution stability thresholds

### 2. Temporal Quality Formatter

**Input:** `TemporalQualityMetrics` (DB model) or Pydantic equivalent

**Key metrics to format:**
- Staleness (days since update, is_stale flag)
- Completeness (gap count, largest gap, completeness ratio)
- Patterns (seasonality, trends, change points)
- Update frequency (regularity, expected vs actual)

**Interpretation examples:**
- Stale data: "Data was last updated 45 days ago. For financial reporting with daily update expectations, this represents a significant staleness issue."
- Gaps: "3 temporal gaps detected totaling 12 days. Largest gap: 2024-03-15 to 2024-03-22 (7 days). This may affect time-series analysis accuracy."
- Seasonality: "Strong weekly seasonality detected (strength: 0.82) with peaks on Fridays. Consider this pattern when analyzing daily variations."

**Configuration needs:**
- Staleness thresholds (hours/days/weeks based on expected update frequency)
- Gap tolerance (varies by granularity)
- Seasonality strength thresholds

### 3. Topological Quality Formatter

**Input:** `TopologicalQualityMetrics` (DB model) or Pydantic equivalent

**Key metrics to format:**
- Betti numbers (b0 = components, b1 = cycles, b2 = voids)
- Orphaned components
- Anomalous cycles
- Structural stability

**Interpretation examples:**
- Fragmented: "Data structure has 4 disconnected components (b0 = 4). This suggests separate clusters that may need explicit linkage or represent distinct data populations."
- Cycles: "2 anomalous cycles detected in the data flow. This may indicate circular dependencies or unexpected relationships: Order → Invoice → Payment → Order."
- Stability: "Topological structure changed significantly from baseline (bottleneck distance: 0.45). Review recent data changes."

**Configuration needs:**
- Expected Betti number ranges (dataset-dependent)
- Cycle detection sensitivity
- Stability thresholds

### 4. Domain Quality Formatter

**Input:** `DomainQualityMetrics`, `FinancialQualityMetrics` (DB models)

**Key metrics to format:**
- Rule violations (by severity)
- Financial: balance checks, sign conventions, period integrity
- Generic: domain compliance score

**Interpretation examples:**
- Balance failure: "Double-entry balance violated: Debits ($1,234,567.89) ≠ Credits ($1,234,565.34). Difference: $2.55. This exceeds the tolerance of $0.01."
- Sign violation: "3 revenue entries have negative signs, violating sign convention for revenue accounts. Affected accounts: REV-001, REV-047, REV-199."
- Period cutoff: "5 transactions dated after period close (2024-03-31) found in March ledger. This violates period cutoff rules."

**Configuration needs:**
- Balance tolerance (precision requirements vary)
- Sign convention rules (account type → expected sign)
- Period boundary definitions (fiscal calendar)

---

## Implementation Phases

### Phase 1: Configuration Infrastructure (Foundation)

1. Create `ThresholdSet` class to hold all thresholds for a pillar
2. Create YAML loader for threshold configuration files
3. Implement threshold resolution (defaults → domain → dataset → runtime)
4. Add threshold configuration directory structure
5. Add per-column pattern overrides (`*_id`, `*_optional`)

**Files to create:**
- `quality/formatting/config.py` - Configuration loading
- `config/formatter_thresholds/defaults.yaml` - Base defaults
- `config/formatter_thresholds/financial.yaml` - Financial domain

### Phase 2: Statistical Formatter

1. Define `StatisticalFormatterConfig` with thresholds
2. Implement `format_statistical_quality_for_llm()`
3. Add interpretation templates for each metric type
4. Add tests with different threshold configurations

**Files to create:**
- `quality/formatting/statistical.py`
- `tests/quality/test_statistical_formatting.py`

### Phase 3: Temporal Formatter

1. Define `TemporalFormatterConfig` with thresholds
2. Implement `format_temporal_quality_for_llm()`
3. Handle granularity-dependent interpretations
4. Add tests

**Files to create:**
- `quality/formatting/temporal.py`
- `tests/quality/test_temporal_formatting.py`

### Phase 4: Topological Formatter

1. Define `TopologicalFormatterConfig` with thresholds
2. Implement `format_topological_quality_for_llm()`
3. Create human-friendly Betti number explanations
4. Add tests

**Files to create:**
- `quality/formatting/topological.py`
- `tests/quality/test_topological_formatting.py`

### Phase 5: Domain Formatter

1. Define `DomainFormatterConfig` with thresholds
2. Implement `format_domain_quality_for_llm()`
3. Handle financial-specific formatting
4. Add tests

**Files to create:**
- `quality/formatting/domain.py`
- `tests/quality/test_domain_formatting.py`

### Phase 6: Calculation Graph & Schema Mapping ✅ DONE

**What was built:**
- `calculations/graphs.py` - Graph loader (`GraphLoader`, `CalculationGraph`, `AbstractField`)
- `calculations/mapping.py` - Schema mapping models (`SchemaMapping`, `ColumnMapping`, `DatasetSchemaMapping`)
- `calculations/matcher.py` - LLM-based schema matcher (`SchemaMatcherLLM`)

**Key clarification:** The calculation graph DEFINES the aggregation method (e.g., `aggregation: "sum"` for revenue). The schema mapping ONLY binds concrete columns to abstract fields. The LLM matcher says "transactions.amount maps to revenue with 85% confidence" - it does NOT determine how revenue is aggregated.

```
Calculation Graph                    Schema Mapping (LLM)
─────────────────                    ────────────────────
revenue:                             revenue:
  aggregation: "sum"                   - transactions.amount (confidence: 0.85)
  required: true                       - sales.total (confidence: 0.70)
```

**Note:** `AggregationDefinition` in `mapping.py` is for OVERRIDES only - when a specific dataset needs different aggregation than the graph default. Most mappings should NOT specify aggregation.

### Phase 7: Filter Generation (LLM) - 🔄 IN PROGRESS

**Existing infrastructure to extend (NOT replace):**
- `quality/filtering/models.py` - `FilteringRecommendations`, `FilteringRule`, `FilteringResult`
- `quality/filtering/llm_filter_agent.py` - `analyze_quality_for_filtering()`
- `quality/filtering/executor.py` - `execute_filtering()`
- `quality/filtering/rules_merger.py` - `merge_filtering_rules()`

**Changes needed:**

1. **Extend `FilteringRecommendations`** to include:
   - `scope_filters`: Row selection for calculations (e.g., `type = 'sale'`)
   - `quality_filters`: Data cleaning (e.g., `amount IS NOT NULL`)
   - `flags`: Issues that can't be filtered, only flagged
   - `confidence`: LLM confidence in recommendations
   - `requires_acknowledgment`: Whether human review needed

2. **Add business cycles context** to `analyze_quality_for_filtering()`:
   - Which business cycle(s) this table belongs to
   - Cycle completeness (are related tables present?)
   - Business value (high/medium/low priority)

3. **Add schema mapping context** to filter generation:
   - Which calculations use this table's columns
   - Required vs optional fields
   - Downstream impact of quality issues

4. **Update prompt** (`config/prompts/filtering_analysis.yaml`):
   - Include business cycle context
   - Include calculation dependency context
   - Generate both scope and quality filters

**Files to modify:**
- `quality/filtering/models.py` - Extend `FilteringRecommendations`
- `quality/filtering/llm_filter_agent.py` - Add business cycles + schema mapping context
- `config/prompts/filtering_analysis.yaml` - Update prompt template

**Key principle:** One LLM call generates both scope filters (calculation boundaries) and quality filters (data cleaning). They can be separated later if needed.

### Phase 8: Integration

1. Update `quality/context.py` to use all formatters
2. Wire formatter output → schema mapping → filter generation
3. Add formatter selection based on detected domain
4. Add API/MCP parameters for threshold overrides
5. Expose FilterResponse to downstream agents
6. Integration tests with full pipeline

---

## Design Decisions

### 1. Per-Column Thresholds

**Decision:** YES - Support column-pattern overrides in configuration.

```yaml
statistical:
  null_ratio:
    default:
      none: 0.01
    patterns:
      "*_id": { none: 0.0 }  # IDs should never be null
      "*_optional": { none: 0.5 }  # Optional fields can be sparse
```

### 2. Output Versioning

**Decision:** LATER - Can be added for A/B testing when needed. Not blocking.

---

## Open Question: LLM vs Templates for Filter Generation

### The Problem

Formatters produce interpretations and recommendations. But the actionable output is **filters for clean data**:

```
Raw Data → Quality Metrics → Filters → Clean Data Subset
```

Example:
```
Metrics: null_ratio=0.15, outlier_samples=[-15234, 892000]
Filters: WHERE amount IS NOT NULL AND amount BETWEEN 0 AND 100000
```

### What Context Does LLM Need?

| Context Type | Example | Why Needed |
|--------------|---------|------------|
| Metric values | `null_ratio: 0.15` | Raw facts |
| Column semantics | `semantic_role: measure, entity_type: currency` | Interpretation |
| Sample outliers | `[-15234, 892000]` | Understand what's excluded |
| Domain | `financial_reporting` | Business acceptability |
| Thresholds | `null_ratio > 0.1 = high` | Severity judgment |
| Downstream use | `regulatory vs exploratory` | Filter strictness |
| Column relationships | `derived_from: price * quantity` | Root cause analysis |

**Key insight:** LLM needs to understand **why** values are problematic, not just **that** they are.

### Approach Comparison

| Aspect | Template-Based | LLM-Based |
|--------|---------------|-----------|
| Reproducibility | Deterministic | Needs techniques |
| Nuance | Limited | Can reason about edge cases |
| Speed | Fast | Slower |
| Testability | Easy | Harder |
| Example | `WHERE col IS NOT NULL` | "This -15234 might be a legitimate refund" |

### Reproducibility Techniques for LLM

| Technique | How It Helps |
|-----------|--------------|
| `temperature=0` | More deterministic (not fully) |
| Structured output | JSON schema enforcement |
| Prompt versioning | Track which prompt → which output |
| Response caching | Same input → same cached output |
| Persist with analysis | Store generated filters alongside metrics |
| Human approval gate | LLM suggests, human approves |

### Proposed Hybrid Approach

```
Metrics → Template Rules → Candidate Filters
                              ↓
              LLM Review (opt-in) → Refined Filters + Reasoning
                              ↓
                      Human Approval → Applied Filters (stored)
```

**Strictness levels:**
- `strict`: Template rules only, fully deterministic
- `balanced`: Templates + LLM refinement + caching
- `exploratory`: LLM-heavy, human review required

### Resolved Design Questions

**1. Primary Consumer:**
- LLM creates SQL filters
- Humans can interpret and overwrite
- Once set, filters are used downstream

**2. Approval Mechanism:**
- Auto-approve above thresholds
- Human must **acknowledge** (not approve)
- Editing mode designed later

**3. Filter Conflicts → Downstream Impact Analysis:**

Use calculation graphs (see `prototypes/calculation-graphs/*_graph.yaml`) to evaluate column importance.

**IMPORTANT:** Calculation graphs define HOW things are calculated using abstract field names (e.g., `revenue`, `accounts_receivable`), NOT which actual columns they need. The system must MAP abstract fields to the schema at hand.

```yaml
# Graph defines abstract dependency:
revenue:
  source:
    statement: "income_statement"
    standard_field: "revenue"   # Abstract field name
    aggregation: "sum"          # How to aggregate
  required: true
  nullable: false
```

### Origin Tables vs Target Schema

**Critical distinction:** Filters apply to ORIGIN tables, aggregations produce TARGET schema.

```
ORIGIN TABLES (raw)              TARGET SCHEMA (for calculations)
─────────────────────            ────────────────────────────────
transactions.amount        →     revenue = SUM(amount) GROUP BY period
transactions.date          →
ledger.ar_balance          →     accounts_receivable = END_OF_PERIOD(ar_balance)
costs.line_items           →     operating_expenses = SUM(amount) BY category
```

**Data flow:**
```
Origin Tables → [FILTERS] → Clean Origin Data → [AGGREGATIONS] → Target Schema
     ↑              ↑                                  ↑
     │              │                                  │
  Quality       Filters defined              Schema mapping defines
  metrics       by LLM, apply                aggregation logic
  computed      to origin tables             (independent of filters)
  here
```

**Key insight:**
- Filters are about data quality at the SOURCE level (origin tables)
- Aggregations are about transforming clean data into calculation inputs
- These are independent concerns - filters don't know about aggregations

### Schema Mapping (Two Parts)

**Part 1: Origin Column Mapping**
```
Abstract field: revenue
Origin columns: transactions.amount, transactions.date, transactions.type
Filter target:  transactions (the origin table)
```

**Part 2: Aggregation Definition**
```
Target field:   revenue
Aggregation:    SUM(transactions.amount) WHERE type = 'sale' GROUP BY period
```

The LLM/schema matcher must identify:
- Which origin columns feed into which abstract fields
- What aggregation logic produces the target schema
- Filters apply to origin columns, not aggregated results

---

## Separation of Concerns

### What Template Rules Do (Formatters)

**Formatters CONTEXTUALIZE metrics, they do NOT generate filters.**

**Key Decision: Deterministic formatters with metric grouping.**

```
Raw Metrics → Formatter → Contextualized Metrics
                              ↓
                    - Severity level (based on thresholds)
                    - Interpretation (what this means)
                    - Downstream dependencies (what's affected)
```

### Metric Grouping (Reduces 127 metrics → ~15-20 interpretation groups)

Instead of interpreting each metric individually, group related metrics:

| Group | Metrics | Single Interpretation |
|-------|---------|----------------------|
| **Benford Analysis** | chi_square, p_value, compliant, digit_distribution | "First-digit distribution [conforms/deviates]" |
| **Outlier Detection** | iqr_*, isolation_forest_*, samples | "X% outliers detected, samples: [...]" |
| **Completeness** | null_ratio, null_count, total_count | "X% missing values" |
| **Temporal Gaps** | gap_count, largest_gap_days, completeness_ratio | "X gaps totaling Y days" |
| **Seasonality** | has_seasonality, strength, period, peaks | "Weekly pattern detected, peaks Friday" |
| **Staleness** | is_stale, freshness_days, last_update | "Data is X days old" |
| **Topology** | betti_0, betti_1, orphaned_components | "X disconnected components" |
| **Multicollinearity** | vif, condition_index, dependency_groups | Already implemented |

**Benefits:**
- Deterministic and testable
- ~15-20 groups vs 127 individual interpretations
- Each group has one ThresholdConfig and one interpretation template
- Business impact synthesis can be added via LLM later (optional layer)

Example formatter output:
```python
{
    "column": "transactions.total_amount",
    "metrics": {
        "null_ratio": {
            "value": 0.02,
            "severity": "low",
            "interpretation": "2% of values are missing",
            "business_impact": "May affect aggregate calculations"
        },
        "outlier_ratio": {
            "value": 0.08,
            "severity": "moderate",
            "interpretation": "8% statistical outliers detected",
            "samples": [-15234.00, 892000.00],
            "business_impact": "Large values may skew averages"
        }
    },
    "downstream_calculations": [
        {
            "calculation": "DSO",
            "mapped_field": "revenue",
            "is_required": true,
            "is_nullable": false
        }
    ],
    "domain": "financial_reporting",
    "threshold_profile": "strict"
}
```

### What LLM Does (Filter Generation)

**LLM receives contextualized metrics and generates filters with structured JSON output.**

The LLM response is:
1. **Stored** - persisted for reproducibility
2. **Reusable** - downstream agents consume the same format
3. **Editable** - humans can modify after generation

```python
@dataclass
class FilterResponse:
    """LLM-generated filter response. Stored and reused by downstream agents."""

    # Identification
    response_id: str
    generated_at: datetime
    prompt_version: str

    # Filters
    filters: list[FilterDefinition]
    flags: list[QualityFlag]

    # Metadata
    reasoning: str
    confidence: float
    requires_acknowledgment: bool

    # Human edits
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    human_modified: bool = False

@dataclass
class FilterDefinition:
    """Single filter condition."""
    column: str
    condition: str           # SQL WHERE clause fragment
    reason: str              # Why this filter
    rows_excluded_pct: float
    auto_approve: bool
    review_note: str | None = None

@dataclass
class QualityFlag:
    """Issue that can't be filtered, only flagged."""
    issue_type: str
    column: str
    description: str
    recommendation: str
```

### Schema for LLM Response (Enforced)

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["filters", "flags", "reasoning"],
    "properties": {
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["column", "condition", "reason", "rows_excluded_pct", "auto_approve"],
                "properties": {
                    "column": {"type": "string"},
                    "condition": {"type": "string"},
                    "reason": {"type": "string"},
                    "rows_excluded_pct": {"type": "number"},
                    "auto_approve": {"type": "boolean"},
                    "review_note": {"type": "string"}
                }
            }
        },
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["issue_type", "column", "description"],
                "properties": {
                    "issue_type": {"type": "string"},
                    "column": {"type": "string"},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"}
                }
            }
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    }
}
```

---

## Complete Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. QUALITY ANALYSIS (on origin tables)                              │
│                                                                      │
│    Origin Tables → Profiling → Quality Metrics                      │
│    + Business Cycle Detection → Which tables form AR/AP/etc cycles  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CONTEXTUALIZATION (Formatters - Deterministic, Grouped) ✅ DONE  │
│                                                                      │
│    Quality Metrics + Thresholds + Domain → Contextualized Metrics   │
│    Business Cycles → BusinessCyclesOutput (severity, interpretation)│
│                                                                      │
│    Formatters: statistical, temporal, topological, domain,          │
│                multicollinearity, business_cycles                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. SCHEMA MAPPING (LLM - column binding only) ✅ DONE               │
│                                                                      │
│    Input: Calculation graph abstract fields + Dataset columns       │
│    Output: Which concrete columns map to which abstract fields      │
│                                                                      │
│    NOTE: Aggregation is defined in graph, NOT in mapping            │
│          LLM just says "transactions.amount → revenue"              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. FILTER GENERATION (LLM) 🔄 IN PROGRESS                           │
│                                                                      │
│    Input:                                                           │
│      - Contextualized quality metrics                               │
│      - Business cycles (which process is this table part of?)       │
│      - Schema mapping (which calculations use these columns?)       │
│                                                                      │
│    Output (extends FilteringRecommendations):                       │
│      - scope_filters: Row selection for calculations                │
│      - quality_filters: Data cleaning                               │
│      - flags: Issues to flag (can't filter)                         │
│      - rationale: Explanation for each                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. FILTER EXECUTION (existing executor.py)                          │
│                                                                      │
│    FilteringRecommendations + User Rules → Merged Rules             │
│    Merged Rules → execute_filtering() → clean_view + quarantine     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 6. CALCULATION EXECUTION                                            │
│                                                                      │
│    Clean Views + Schema Mapping + Graph Aggregations                │
│    → Execute calculation SQL from graph templates                   │
│    → DSO, Cash Runway, OCF, etc.                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principles:**
- Filters and Aggregations are independent concerns
- Filters = data quality (which rows to include from origin)
- Aggregations = defined in calculation graphs, applied to clean data
- Schema mapping = column binding only (which column → which field)
- Business cycles = context for prioritizing quality focus

---

## Other Open Questions

### Threshold Validation

Should thresholds be validated on load? E.g., ensure `none < low < moderate < high < severe`?

**Recommendation:** Yes, validate with clear error messages. Invalid thresholds are a configuration bug.

### Metric Dependencies

Some interpretations depend on multiple metrics (e.g., staleness interpretation depends on detected granularity).

**Recommendation:** Formatters receive the full metrics object and can cross-reference as needed.

---

## Success Criteria

### Formatters (Contextualization) ✅ DONE
1. [x] All 6 formatters implemented (statistical, temporal, topological, domain, multicollinearity, business_cycles)
2. [x] YAML configuration system working with resolution hierarchy
3. [x] Per-column pattern overrides (`*_id`, `*_optional`)
4. [x] Default threshold configurations in `quality/formatting/config.py`
5. [x] Formatters output contextualized metrics (severity, interpretation, recommendations)

### Schema Mapping ✅ DONE
6. [x] Calculation graph loader extracts abstract field definitions (`calculations/graphs.py`)
7. [x] LLM-based schema mapping: abstract fields → concrete columns (`calculations/matcher.py`)
8. [x] Schema mappings modeled (`calculations/mapping.py`)
9. [x] Clarified: Aggregation in graph, mapping is column binding only

### Filter Generation 🔄 IN PROGRESS
10. [ ] Extend `FilteringRecommendations` with scope_filters, quality_filters, flags
11. [ ] Add business cycles context to filter generation
12. [ ] Add schema mapping context (downstream impact) to filter generation
13. [ ] Update `filtering_analysis.yaml` prompt
14. [ ] Test full filter generation with business cycle + mapping context

### Integration ⏳ PENDING
15. [ ] Full pipeline: metrics → formatters → mapping → filters → execution
16. [ ] Clean view creation with both scope and quality filters
17. [ ] Calculation execution using clean views + graph aggregations

---

## References

- `docs/METRICS_ANALYSIS.md` - Complete metrics inventory
- `quality/formatting/multicollinearity.py` - Reference implementation
- `quality/formatting/base.py` - Core utilities
- `prototypes/calculation-graphs/*_graph.yaml` - Downstream calculation dependencies
  - `cash_runway_graph.yaml` - Cash runway calculation
  - `dso_calculation_graph.yaml` - Days Sales Outstanding
  - `ocf_calculation_graph.yaml` - Operating Cash Flow
