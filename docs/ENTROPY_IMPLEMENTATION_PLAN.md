# Entropy Layer Implementation Plan

## Executive Summary

This document outlines the plan for adding an entropy management layer to the dataraum-context engine. The entropy layer will quantify uncertainty in data to enable LLM-driven analytics to make deterministic, reliable decisions.

**Scope for Phase 1:**
- Structural entropy (Schema, Types, Relations)
- Semantic entropy (Business Meaning, Units, Temporal, Categorical)
- Value entropy (Categories, Null Semantics, Outliers, Ranges, Patterns)
- Computational entropy (Derived Values, Business Rules, Filters, Aggregations)

**Deferred:**
- Source entropy (requires access/provenance infrastructure not yet built)
- Query entropy (requires query agent not yet implemented)

---

## Part 1: Current State Analysis

### Existing Analysis Modules → Entropy Dimension Mapping

| Module | Location | Current Output | Supports Entropy Dimension |
|--------|----------|----------------|---------------------------|
| `typing` | `analysis/typing/` | TypeCandidate, TypeDecision, patterns, units | Structural.Types, Value.Patterns |
| `statistics` | `analysis/statistics/` | ColumnProfile (counts, distributions, histograms) | Value.Ranges, Value.Nulls, Value.Outliers |
| `correlation` | `analysis/correlation/` | NumericCorrelation, CategoricalAssociation, FunctionalDependency, DerivedColumn | Computational.DerivedValues, Structural.Relations |
| `relationships` | `analysis/relationships/` | JoinCandidate, RelationshipCandidate | Structural.Relations |
| `semantic` | `analysis/semantic/` | SemanticAnnotation, EntityDetection, Relationship | Semantic.BusinessMeaning, Semantic.Categorical |
| `temporal` | `analysis/temporal/` | TemporalAnalysisResult (gaps, trends, seasonality, change points) | Semantic.Temporal, Value.Patterns |
| `topology` | `analysis/topology/` | TopologicalQualityResult (Betti numbers, cycles) | Structural.Relations (graph structure) |
| `slicing` | `analysis/slicing/` | SliceRecommendation | Semantic.Categorical |
| `temporal_slicing` | `analysis/temporal_slicing/` | TemporalAnalysisResult (per-slice temporal) | Semantic.Temporal, Value.Patterns |
| `cycles` | `analysis/cycles/` | DetectedCycle, BusinessCycleAnalysis | Computational.BusinessRules |
| `validation` | `analysis/validation/` | ValidationResult | Computational.BusinessRules, Value.* |
| `quality_summary` | `analysis/quality_summary/` | ColumnQualitySummary, QualityIssue | Synthesis layer (not entropy source) |

### Gaps in Current Coverage

#### Structural Entropy
| Sub-dimension | Current Coverage | Gap |
|---------------|-----------------|-----|
| Schema.naming_clarity | ❌ Not computed | Need abbreviation detection, naming consistency scoring |
| Schema.naming_consistency | ❌ Not computed | Need cross-table naming pattern analysis |
| Schema.stability | ❌ Not tracked | Need schema versioning (out of scope for Phase 1) |
| Schema.organization | ❌ Not computed | Need table grouping logic detection |
| Types.type_fidelity | ✅ Via `typing` module | Has type candidates, parse success rate |
| Types.type_consistency | ⚠️ Partial | Need cross-table type consistency check |
| Types.nullability_accuracy | ⚠️ Partial | Statistics has null counts, needs declared vs actual comparison |
| Types.precision_appropriateness | ❌ Not computed | Need precision analysis for decimals |
| Relations.cardinality_clarity | ✅ Via `relationships` | Has cardinality detection |
| Relations.join_path_determinism | ⚠️ Partial | Need to identify ambiguous paths |
| Relations.referential_integrity | ✅ Via `relationships.evaluator` | Has orphan detection |
| Relations.relationship_semantics | ⚠️ Partial | Need semantic labeling of relationships |

#### Semantic Entropy
| Sub-dimension | Current Coverage | Gap |
|---------------|-----------------|-----|
| BusinessMeaning.definition_exists | ⚠️ Partial | Has `business_description` but no glossary integration |
| BusinessMeaning.definition_precision | ❌ Not computed | Need LLM to score definition quality |
| BusinessMeaning.cross_reference_consistency | ❌ Not computed | Need cross-table term consistency check |
| BusinessMeaning.example_availability | ❌ Not tracked | Need sample values linked to definitions |
| Units.unit_declared | ✅ Via `typing` module | Has unit detection (Pint) |
| Units.unit_consistency | ❌ Not computed | Need cross-column unit consistency check |
| Units.conversion_available | ❌ Not tracked | No currency/unit conversion infrastructure |
| Units.scale_clarity | ❌ Not computed | Need magnitude detection (thousands, millions) |
| Temporal.accumulation_type | ⚠️ Partial | `temporal` detects trends but not stock/flow classification |
| Temporal.point_in_time_clarity | ⚠️ Partial | Has `detected_granularity` |
| Temporal.timezone_handling | ❌ Not computed | No timezone detection |
| Temporal.period_alignment | ⚠️ Partial | `fiscal_calendar` detection exists |
| Categorical.hierarchy_definition | ❌ Not computed | Need hierarchy inference |
| Categorical.level_semantics | ❌ Not computed | Need level meaning detection |
| Categorical.completeness | ⚠️ Partial | Slicing identifies dimensions, no orphan analysis |
| Categorical.stability | ❌ Not tracked | No category versioning |

#### Value Entropy
| Sub-dimension | Current Coverage | Gap |
|---------------|-----------------|-----|
| Categories.value_consistency | ❌ Not computed | Need variant grouping (Active/active/A) |
| Categories.vocabulary_completeness | ❌ Not computed | Need enum definition vs actual comparison |
| Categories.variant_density | ❌ Not computed | Need variant counting per concept |
| Categories.coverage | ❌ Not computed | Need recognized vs unknown value ratio |
| NullSemantics.null_meaning_defined | ❌ Not computed | Need null meaning metadata |
| NullSemantics.null_consistency | ⚠️ Partial | Statistics has null ratio |
| NullSemantics.null_representation | ❌ Not computed | Need variant detection (NULL, "", "N/A") |
| NullSemantics.null_handling_rules | ❌ Not tracked | No aggregation rule metadata |
| Outliers.outlier_rate | ✅ Via `statistics` | IQR/z-score outlier detection |
| Outliers.outlier_documentation | ❌ Not tracked | No outlier explanation metadata |
| Outliers.outlier_handling | ❌ Not tracked | No inclusion/exclusion rules |
| Outliers.detection_method | ✅ Tracked | Method stored in quality metrics |
| Ranges.bounds_defined | ❌ Not tracked | No declared min/max metadata |
| Ranges.bounds_enforced | ⚠️ Partial | Statistics has observed min/max |
| Ranges.distribution_expected | ❌ Not tracked | No expected distribution metadata |
| Ranges.boundary_semantics | ❌ Not tracked | No bound meaning metadata |
| Patterns.format_consistency | ✅ Via `typing` | Pattern match rate computed |
| Patterns.pattern_documentation | ⚠️ Partial | Detected patterns stored, not "expected" |
| Patterns.conformance_rate | ✅ Via `typing` | `pattern_match_rate` available |
| Patterns.variant_handling | ❌ Not tracked | No normalization rules |

#### Computational Entropy
| Sub-dimension | Current Coverage | Gap |
|---------------|-----------------|-----|
| DerivedValues.formula_documented | ✅ Via `correlation` | DerivedColumn detection with formula |
| DerivedValues.reproducibility | ✅ Via `correlation` | match_rate computed |
| DerivedValues.component_traceability | ✅ Via `correlation` | source_column_ids tracked |
| DerivedValues.calculation_timing | ❌ Not tracked | No refresh metadata |
| BusinessRules.rules_documented | ⚠️ Partial | `cycles` detects processes, `validation` has specs |
| BusinessRules.rule_coverage | ❌ Not computed | No coverage analysis |
| BusinessRules.exception_handling | ❌ Not tracked | No exception metadata |
| BusinessRules.rule_versioning | ❌ Not tracked | No rule history |
| Filters.default_filters_documented | ❌ Not tracked | No default filter metadata |
| Filters.filter_scope | ❌ Not tracked | No filter scope metadata |
| Filters.override_possibility | ❌ Not tracked | No override mechanism |
| Filters.filter_rationale | ❌ Not tracked | No filter reasoning |
| Aggregations.aggregation_rules | ❌ Not tracked | No aggregation rule metadata |
| Aggregations.weighting_documented | ❌ Not tracked | No weight column metadata |
| Aggregations.null_handling_in_agg | ❌ Not tracked | No aggregation null handling |
| Aggregations.hierarchy_aggregation | ❌ Not tracked | No rollup rule metadata |

### Graph Agent Context Analysis

The graph agent (`graphs/agent.py`) uses `GraphExecutionContext` from `graphs/context.py` which aggregates:
- Table/column metadata from storage
- Statistical profiles
- Semantic annotations
- Temporal profiles
- Type decisions
- Relationships
- Slice definitions
- Quality grades from quality_summary
- Derived column info from correlation
- Business cycles

**Current gap:** The context builder does not expose entropy scores or resolution hints. It provides raw metrics but no uncertainty quantification.

### Cleanup Leftovers Identified

1. **Quality Module (`src/dataraum_context/quality/`)**: Contains formatting utilities (`formatting/`) and synthesis logic (`synthesis.py`). The synthesis logic overlaps with `quality_summary` module. **Action:** Merge useful formatting into a common utility, remove synthesis.py.

2. **Duplicate Phase Scripts**: Phase 6 (`run_phase6_correlation.py`) vs Phase 4b (`run_phase4b_correlations.py`) - need to understand differences.

3. **Temporal Analysis Duplication**: `analysis/temporal/` and `analysis/temporal_slicing/` have overlapping temporal analysis. **Action:** Clarify responsibilities - temporal is per-column, temporal_slicing is per-slice.

---

## Part 2: Entropy Layer Architecture

### Entropy Object Model

```python
@dataclass
class EntropyObject:
    """Core entropy measurement object."""

    # Identity
    object_id: str
    layer: str  # structural, semantic, value, computational
    dimension: str  # schema, types, relations, etc.
    sub_dimension: str  # naming_clarity, type_fidelity, etc.
    target: str  # column:table.column, table:table_name, relationship:t1-t2

    # Measurement
    score: float  # 0.0 = deterministic, 1.0 = maximum uncertainty
    confidence: float  # How confident are we in this score

    # Evidence
    evidence: list[dict]  # Type-specific evidence

    # Resolution
    resolution_options: list[ResolutionOption]

    # Context for different consumers
    llm_context: dict  # For query agent
    human_context: dict  # For UI/admin

    # Metadata
    computed_at: datetime
    source_analysis_id: str  # Link to source analysis (stats, semantic, etc.)
```

### Entropy Detector Interface

```python
class EntropyDetector(ABC):
    """Base class for entropy dimension detectors."""

    @abstractmethod
    def dimension(self) -> str:
        """Return the dimension this detector measures."""
        pass

    @abstractmethod
    async def detect(
        self,
        session: AsyncSession,
        duckdb_conn: DuckDBPyConnection,
        table_ids: list[str],
    ) -> list[EntropyObject]:
        """Detect entropy for the given tables."""
        pass
```

### Module Structure

```
src/dataraum_context/
├── entropy/                           # NEW: Entropy layer
│   ├── __init__.py
│   ├── models.py                      # EntropyObject, ResolutionOption
│   ├── db_models.py                   # Persistence
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                    # EntropyDetector ABC
│   │   ├── structural/
│   │   │   ├── schema.py              # NamingClarity, NamingConsistency
│   │   │   ├── types.py               # TypeFidelity, TypeConsistency
│   │   │   └── relations.py           # JoinPathDeterminism, CardinalityClarity
│   │   ├── semantic/
│   │   │   ├── business_meaning.py    # DefinitionExists, DefinitionPrecision
│   │   │   ├── units.py               # UnitDeclared, UnitConsistency, ScaleClarity
│   │   │   ├── temporal.py            # AccumulationType, PointInTimeClarity
│   │   │   └── categorical.py         # HierarchyDefinition, LevelSemantics
│   │   ├── value/
│   │   │   ├── categories.py          # ValueConsistency, VariantDensity
│   │   │   ├── null_semantics.py      # NullMeaning, NullRepresentation
│   │   │   ├── outliers.py            # OutlierRate, OutlierHandling
│   │   │   ├── ranges.py              # BoundsEnforced, BoundarySemantics
│   │   │   └── patterns.py            # FormatConsistency, PatternDocumentation
│   │   └── computational/
│   │       ├── derived_values.py      # FormulaDocumented, Reproducibility
│   │       ├── business_rules.py      # RulesDocumented, RuleCoverage
│   │       ├── filters.py             # FilterDocumented, FilterScope
│   │       └── aggregations.py        # AggregationRules, NullHandling
│   ├── aggregation.py                 # Aggregate entropy across dimensions
│   ├── scoring.py                     # Compute composite entropy scores
│   └── context.py                     # Build entropy context for LLM
```

---

## Part 3: Implementation Plan

### Phase 1: Foundation (Entropy Core)

**Goal:** Create entropy infrastructure and implement high-value detectors using existing analysis data.

#### Step 1.1: Core Models and Storage
- Create `entropy/models.py` with `EntropyObject`, `ResolutionOption`
- Create `entropy/db_models.py` with SQLAlchemy models
- Create database migration

#### Step 1.2: Detector Infrastructure
- Create `EntropyDetector` ABC in `entropy/detectors/base.py`
- Create `EntropyRegistry` for detector discovery
- Create `entropy/processor.py` for running all detectors

#### Step 1.3: Structural Entropy Detectors (from existing data)

| Detector | Source Data | Priority |
|----------|-------------|----------|
| `TypeFidelityDetector` | `typing/TypeCandidate.parse_success_rate` | High |
| `TypeConsistencyDetector` | Cross-table type comparison | Medium |
| `CardinalityClarity Detector` | `relationships/JoinCandidate.cardinality` | High |
| `JoinPathDeterminismDetector` | Multiple paths between tables | High |
| `NamingClarityDetector` | Abbreviation detection, column name analysis | Medium |

#### Step 1.4: Value Entropy Detectors (from existing data)

| Detector | Source Data | Priority |
|----------|-------------|----------|
| `NullRatioDetector` | `statistics/ColumnProfile.null_ratio` | High |
| `OutlierRateDetector` | `statistics/quality.iqr_outlier_ratio` | High |
| `RangeBoundsDetector` | `statistics/ColumnProfile.numeric_stats` | Medium |
| `PatternConsistencyDetector` | `typing/TypeCandidate.pattern_match_rate` | Medium |

#### Step 1.5: Semantic Entropy Detectors (from existing data)

| Detector | Source Data | Priority |
|----------|-------------|----------|
| `BusinessMeaningDetector` | `semantic/SemanticAnnotation.business_description` | High |
| `UnitDeclaredDetector` | `typing/TypeCandidate.detected_unit` | High |
| `TemporalClarityDetector` | `temporal/TemporalAnalysisResult.detected_granularity` | Medium |

#### Step 1.6: Computational Entropy Detectors (from existing data)

| Detector | Source Data | Priority |
|----------|-------------|----------|
| `DerivedValueDetector` | `correlation/DerivedColumn.formula, match_rate` | High |
| `BusinessRuleDetector` | `cycles/DetectedCycle`, `validation/ValidationSpec` | Medium |

### Phase 2: Context Integration

**Goal:** Integrate entropy into graph agent context.

#### Step 2.1: Entropy Context Builder
- Extend `graphs/context.py` to include entropy scores
- Add `EntropyContext` dataclass with per-column/table entropy
- Create `format_entropy_for_prompt()` function

#### Step 2.2: Graph Agent Enhancement
- Update graph agent to consume entropy context
- Add entropy-aware SQL generation hints
- Implement assumption tracking when entropy is high

#### Step 2.3: Resolution Hint API
- Create endpoint/function to get resolution hints for high-entropy items
- Prioritize hints by expected entropy reduction × effort

### Phase 3: Cleanup and Consolidation

**Goal:** Remove redundancy and streamline the pipeline.

#### Step 3.1: Quality Module Cleanup
- Move `quality/formatting/` utilities to `core/formatting/` or `analysis/common/`
- Remove `quality/synthesis.py` (functionality in `quality_summary`)
- Delete `quality/` module

#### Step 3.2: Context Generation Streamlining
- Consolidate context generation across modules
- Single source of truth for building LLM context
- Deprecate redundant context builders

#### Step 3.3: Phase Script Consolidation
- Review phase script overlap (phase 4b vs 6)
- Create unified pipeline runner
- Document phase dependencies clearly

### Phase 4: Graph Agent Completion

**Goal:** Finish the graph agent with entropy-aware execution.

#### Step 4.1: Complete Field Mapping
- Ensure all business concepts map to columns
- Handle ambiguous mappings using entropy scores

#### Step 4.2: Multi-Table Graph Execution
- Support graphs spanning multiple tables
- Use relationship entropy to validate join paths

#### Step 4.3: Graph Validation
- Validate generated SQL against data quality
- Flag high-uncertainty calculations

### Phase 5: API and MCP Server

**Goal:** Expose entropy and graph functionality via API.

#### Step 5.1: FastAPI Implementation
- Create `api/routes/entropy.py`
- Create `api/routes/graphs.py`
- Create `api/routes/context.py`

#### Step 5.2: MCP Server Implementation
- Implement MCP tools:
  - `get_entropy_context` - Primary entropy retrieval
  - `get_resolution_hints` - Actionable improvements
  - `execute_graph` - Run transformation graph
  - `query` - Execute SQL with context

### Phase 6: UI Foundation

**Goal:** Create basic UI for entropy visualization.

#### Step 6.1: Entropy Dashboard
- Show entropy by dimension
- Color-coded severity (green/yellow/red)
- Drill-down to specific items

#### Step 6.2: Resolution Workflow
- Display resolution hints
- Allow applying resolutions
- Track resolution history

---

## Part 4: Entropy Calculation Formulas

### Column-Level Entropy Score

For each column, calculate dimension-specific scores then aggregate:

```
column_entropy = weighted_average([
    structural_entropy * 0.25,
    semantic_entropy * 0.30,
    value_entropy * 0.30,
    computational_entropy * 0.15
])
```

### Structural Entropy (Column)
```
structural = max(
    type_fidelity_entropy,      # 1 - parse_success_rate
    naming_clarity_entropy      # abbreviation_score
)
```

### Semantic Entropy (Column)
```
semantic = weighted_average([
    business_meaning_entropy,   # 1 if no description, 0 if well-defined
    unit_entropy,               # 1 if numeric with no unit, 0 if unit declared
    temporal_entropy            # Only for timestamp columns
])
```

### Value Entropy (Column)
```
value = weighted_average([
    null_entropy,               # Scaled null_ratio (high nulls = high entropy)
    outlier_entropy,            # Scaled outlier_ratio
    pattern_entropy,            # 1 - pattern_match_rate
    category_entropy            # Variant count / distinct count (for categoricals)
])
```

### Computational Entropy (Column)
```
computational = max(
    derived_entropy,            # 1 - match_rate if derived, 0 otherwise
    aggregation_entropy         # 1 if measure with no agg rules, 0 otherwise
)
```

---

## Part 5: Priority Matrix

### High Priority (Phase 1 Focus)

| Entropy Item | Why High Priority | Existing Data |
|--------------|-------------------|---------------|
| Type Fidelity | Affects all SQL generation | typing module |
| Null Semantics | Affects aggregations | statistics module |
| Outlier Handling | Affects averages/sums | statistics module |
| Business Meaning | Core for LLM understanding | semantic module |
| Derived Values | Risk of hidden calculations | correlation module |
| Join Path Determinism | Prevents wrong joins | relationships module |

### Medium Priority (Phase 2)

| Entropy Item | Why Medium | Dependencies |
|--------------|------------|--------------|
| Unit Consistency | Cross-table analysis | Needs multi-table context |
| Temporal Accumulation | Needs LLM analysis | Temporal patterns detected |
| Category Variants | Normalization needed | Category detection needed |
| Aggregation Rules | No existing metadata | Needs configuration |

### Lower Priority (Future)

| Entropy Item | Why Lower | Notes |
|--------------|-----------|-------|
| Naming Consistency | Cosmetic impact | Abbreviation detection is harder |
| Schema Stability | Needs versioning | Infrastructure not built |
| Source Entropy | Access/provenance | No provenance tracking yet |
| Query Entropy | Query agent | No query agent yet |

---

## Part 6: Module Evaluation - Keep/Remove/Merge

### Analysis Modules Assessment

| Module | Verdict | Rationale |
|--------|---------|-----------|
| `typing` | **KEEP** | Core structural entropy source |
| `statistics` | **KEEP** | Core value entropy source |
| `correlation` | **KEEP** | Computational entropy source |
| `relationships` | **KEEP** | Structural entropy source |
| `semantic` | **KEEP** | Semantic entropy source |
| `temporal` | **KEEP** | Temporal entropy source |
| `topology` | **KEEP** (simplify) | Useful for relationship graph structure |
| `slicing` | **KEEP** | Useful for categorical entropy |
| `temporal_slicing` | **EVALUATE** | May overlap with temporal; keep if slice-specific analysis valuable |
| `cycles` | **KEEP** | Business rule entropy source |
| `validation` | **KEEP** | Business rule entropy source |
| `quality_summary` | **KEEP** | Synthesis layer, produces UI-friendly grades |

### Quality Module Disposition

| Component | Action | Reason |
|-----------|--------|--------|
| `quality/formatting/` | **MOVE** to `core/formatting/` | Useful utilities for any output |
| `quality/db_models.py` | **REMOVE** | Superseded by quality_summary |
| `quality/models.py` | **REMOVE** | Superseded by quality_summary |
| `quality/synthesis.py` | **REMOVE** | Functionality in quality_summary |
| `quality/__init__.py` | **REMOVE** | Module deleted |

---

## Part 7: Success Criteria

### Phase 1 Complete When:
- [ ] EntropyObject model defined and stored
- [ ] At least 10 entropy detectors implemented
- [ ] Entropy scores computed for all columns in test dataset
- [ ] Entropy displayed in graph agent context

### Phase 2 Complete When:
- [ ] Graph agent uses entropy to qualify assumptions
- [ ] Resolution hints generated for high-entropy items
- [ ] Quality module removed

### Phase 3 Complete When:
- [ ] Graph agent executes multi-table graphs
- [ ] API exposes entropy and graph endpoints
- [ ] Basic MCP server operational

### Phase 4 Complete When:
- [ ] UI displays entropy dashboard
- [ ] Resolution workflow functional
- [ ] End-to-end pipeline runs automatically

---

## Appendix A: Existing Analysis → Entropy Mapping Detail

### typing module → Entropy

```python
# TypeCandidate → Structural.Types.type_fidelity
entropy_score = 1.0 - type_candidate.parse_success_rate

# TypeCandidate → Value.Patterns.format_consistency
entropy_score = 1.0 - (type_candidate.pattern_match_rate or 1.0)

# TypeCandidate → Semantic.Units.unit_declared
entropy_score = 0.0 if type_candidate.detected_unit else 0.8  # 0.8 if numeric without unit
```

### statistics module → Entropy

```python
# ColumnProfile → Value.NullSemantics
null_entropy = min(1.0, profile.null_ratio * 2)  # Scale so 50% nulls = 1.0

# StatisticalQualityMetrics → Value.Outliers
outlier_entropy = min(1.0, quality.iqr_outlier_ratio * 10)  # Scale so 10% = 1.0

# ColumnProfile → Value.Ranges
if numeric_stats:
    # High entropy if observed range very different from typical
    range_entropy = detect_range_anomaly(numeric_stats)
```

### semantic module → Entropy

```python
# SemanticAnnotation → Semantic.BusinessMeaning
if not annotation.business_description:
    entropy_score = 1.0
elif len(annotation.business_description) < 20:
    entropy_score = 0.7  # Too brief
else:
    entropy_score = 0.2  # Has description, assume decent quality

# SemanticAnnotation → Semantic.Categorical
if annotation.semantic_role == "dimension":
    if not has_hierarchy_definition():
        entropy_score = 0.6
```

### correlation module → Entropy

```python
# DerivedColumn → Computational.DerivedValues
entropy_score = 1.0 - derived.match_rate  # Low match = high uncertainty

# If derived column has no formula
if not derived.formula:
    entropy_score = 1.0
```

### relationships module → Entropy

```python
# Multiple join paths → Structural.Relations.join_path_determinism
paths = find_paths_between(table_a, table_b)
if len(paths) > 1:
    entropy_score = 0.7  # Ambiguous
elif len(paths) == 0:
    entropy_score = 0.9  # No path
else:
    entropy_score = 0.1  # Single clear path
```

---

## Part 8: Session Tracking Strategy

### Recommended Approach: Markdown-Based Tracking

For Claude Code, **markdown files in the repository** are the most effective tracking method because:

1. **Persistence**: Files persist across sessions automatically
2. **Direct Access**: Claude Code can read/update them without external tools
3. **Version Control**: Changes are tracked in git history
4. **No Context Switching**: No need to navigate to external tools

### Tracking Files Structure

```
docs/
├── ENTROPY_IMPLEMENTATION_PLAN.md    # This document (the "what")
├── PROGRESS.md                        # Progress log (the "done")
└── BACKLOG.md                         # Prioritized backlog (the "next")
```

### PROGRESS.md Format

```markdown
# Progress Log

## Current Sprint
- [ ] Task in progress
- [x] Completed task (2024-01-07)

## Session Log
### 2024-01-07
- Created entropy implementation plan
- Fixed lint/type errors in temporal_slicing

### 2024-01-06
- ...
```

### BACKLOG.md Format

```markdown
# Backlog

## Priority 1 (Current Focus)
- [ ] Create entropy core models
- [ ] Implement TypeFidelity detector

## Priority 2 (Next)
- [ ] Extend semantic agent for entropy enrichment
- [ ] Cleanup quality module

## Priority 3 (Later)
- [ ] UI migration from npm to bun
- [ ] MCP server implementation

## Blocked/Waiting
- [ ] Query entropy (waiting for query agent)
```

### Why Not GitHub Projects

While GitHub Projects works well for team coordination, for Claude Code sessions:
- External API calls add latency
- Context switching breaks flow
- Markdown is more flexible for detailed notes
- Git history provides natural audit trail

---

## Part 9: Additional Considerations (User Feedback)

### 9.1 Topology Module Clarification

The codebase has **two topology-related components**:

| Component | Location | Purpose | Keep? |
|-----------|----------|---------|-------|
| Graph topology | `analysis/relationships/graph_topology.py` | NetworkX-based table graph analysis (hub/spoke, star schema) | **KEEP** - Used by context builder |
| TDA topology | `analysis/topology/` | Persistence diagrams, Betti numbers, homology | **SIMPLIFY** - Keep core metrics, remove unused complexity |

**Action**: The `analysis/topology/` module should be simplified to extract useful structural complexity metrics without full TDA overhead. The simpler graph structure analysis stays in `relationships/`.

### 9.2 Semantic Agent Extension

The existing semantic agent (`analysis/semantic/agent.py`) can be extended to enrich:

| Entropy Dimension | Current | Extension Opportunity |
|-------------------|---------|----------------------|
| Structural.Schema.naming_clarity | ❌ | Add abbreviation detection, naming analysis |
| Structural.Types.type_consistency | ⚠️ | Cross-table type comparison in semantic pass |
| Semantic.Units.scale_clarity | ❌ | Detect magnitude (thousands, millions) |
| Semantic.Categorical.hierarchy | ❌ | Infer hierarchies from dimension tables |
| Semantic.Temporal.accumulation_type | ⚠️ | Stock vs flow classification |

**Action**: Create entropy-enrichment prompts for the semantic agent to populate entropy-related fields during its analysis pass. This is more efficient than separate entropy detection.

### 9.3 Config Folder Evaluation

Current config structure:

```
config/
├── null_values.yaml              # Null value variants (KEEP - core)
├── patterns/default.yaml         # Type detection patterns (KEEP - core)
├── llm.yaml                      # LLM provider config (KEEP - core)
├── prompts/                      # LLM prompts (KEEP - extend for entropy)
│   ├── semantic_analysis.yaml
│   ├── quality_summary.yaml
│   ├── slicing_analysis.yaml
│   └── graph_sql_generation.yaml
├── ontologies/                   # Domain ontologies (KEEP - extend)
│   └── financial_reporting.yaml
├── validations/financial/        # Validation specs (KEEP - feeds entropy)
│   ├── double_entry.yaml
│   ├── trial_balance.yaml
│   └── ...
├── cycles/                       # Business cycle vocab (KEEP)
│   └── cycle_vocabulary.yaml
├── graphs/                       # Graph configs (KEEP - core)
│   ├── metrics/                  # Metric definitions
│   └── filters/rules/            # Column filtering rules
└── formatter_thresholds/         # Threshold configs (EVALUATE)
    ├── defaults.yaml
    └── financial.yaml
```

**Actions**:
1. **Keep**: null_values, patterns, llm, prompts, ontologies, validations, cycles, graphs
2. **Extend**: prompts/ with entropy-related prompts
3. **Evaluate**: formatter_thresholds/ - may move to entropy/config/ if useful for entropy scoring

### 9.4 Quality Formatting Utilities

The `quality/formatting/` module contains **useful utilities** that should be preserved:

| File | Purpose | Recommendation |
|------|---------|----------------|
| `base.py` | SeverityLevel enum, ThresholdConfig, interpretation templates | **KEEP** - Move to `core/formatting/` |
| `config.py` | Threshold config loading from YAML | **KEEP** - Move to `core/formatting/` |
| `statistical.py` | Stats-specific formatting | **EVALUATE** - May integrate with entropy |
| `temporal.py` | Temporal formatting | **EVALUATE** |
| `topological.py` | Topology formatting | **EVALUATE** |
| `domain.py` | Domain-specific formatting | **EVALUATE** |
| `business_cycles.py` | Cycle formatting | **EVALUATE** |

**Action**: Move `base.py` and `config.py` to `core/formatting/`. Other formatters need evaluation - they may be useful for entropy context generation or obsolete.

### 9.5 UI Prototype Migration

The UI prototype in `prototypes/calculation-graphs/web_visualizer/` is a **Vite + npm** project that should be:

1. **Migrated** to a new location: `ui/` at project root
2. **Converted** from npm to bun
3. **Extended** to support entropy visualization

Current stack:
- Vite (build tool)
- Tailwind CSS (styling)
- npm (package manager) → migrate to **bun**

**Migration Steps**:
1. Create `ui/` directory
2. Copy web_visualizer source files
3. Convert `package.json` for bun compatibility
4. Run `bun install`
5. Update import paths as needed
6. Add entropy dashboard components

---

## Appendix B: File Changes Summary

### New Files
```
src/dataraum_context/entropy/__init__.py
src/dataraum_context/entropy/models.py
src/dataraum_context/entropy/db_models.py
src/dataraum_context/entropy/processor.py
src/dataraum_context/entropy/aggregation.py
src/dataraum_context/entropy/scoring.py
src/dataraum_context/entropy/context.py
src/dataraum_context/entropy/detectors/__init__.py
src/dataraum_context/entropy/detectors/base.py
src/dataraum_context/entropy/detectors/structural/*.py
src/dataraum_context/entropy/detectors/semantic/*.py
src/dataraum_context/entropy/detectors/value/*.py
src/dataraum_context/entropy/detectors/computational/*.py
```

### Modified Files
```
src/dataraum_context/graphs/context.py  # Add entropy to context
src/dataraum_context/graphs/agent.py    # Use entropy in SQL generation
```

### Deleted Files
```
src/dataraum_context/quality/db_models.py
src/dataraum_context/quality/models.py
src/dataraum_context/quality/synthesis.py
src/dataraum_context/quality/__init__.py
```

### Moved Files
```
src/dataraum_context/quality/formatting/* → src/dataraum_context/core/formatting/*
```
