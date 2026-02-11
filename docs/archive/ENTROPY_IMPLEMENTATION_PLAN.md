# Entropy Layer Implementation Plan

## Executive Summary

This document outlines the plan for adding an entropy management layer to the dataraum-context engine. The entropy layer will quantify uncertainty in data to enable LLM-driven analytics to make deterministic, reliable decisions.

**Design Goal:** End-to-end testability with static data. All entropy dimensions must be computable from existing analysis module outputs without requiring live data connections.

**Scope for Phase 1:**
- Structural entropy (Schema, Types, Relations)
- Semantic entropy (Business Meaning, Units, Temporal, Categorical)
- Value entropy (Categories, Null Semantics, Outliers, Ranges, Patterns)
- Computational entropy (Derived Values, Business Rules, Filters, Aggregations)

**Deferred (requires infrastructure not yet built):**
- Source entropy: Physical, Provenance, Access Rights (requires access/provenance tracking)
- Query entropy (requires query agent implementation)
- Entropy history/trending (requires periodic snapshot infrastructure)

**Related Documentation:**
- [ENTROPY_MODELS.md](./ENTROPY_MODELS.md) - Detailed schema specifications
- [ENTROPY_CONTRACTS.md](./ENTROPY_CONTRACTS.md) - Data readiness thresholds by use case
- [ENTROPY_QUERY_BEHAVIOR.md](./ENTROPY_QUERY_BEHAVIOR.md) - Agent response policies
- [entropy-management-framework.md](../entropy-management-framework.md) - Original specification
- [entropy-framework-future-considerations.md](../entropy-framework-future-considerations.md) - UX and workflow considerations

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
| `temporal` | `analysis/temporal/` | TemporalAnalysisResult (gaps, trends, seasonality, change points, **staleness**) | Semantic.Temporal, Value.Patterns, **Source.Lifecycle.freshness** |
| `topology` | `analysis/topology/` | TopologicalQualityResult (Betti numbers, cycles) | Structural.Relations (graph structure) |
| `slicing` | `analysis/slicing/` | SliceRecommendation | Semantic.Categorical |
| `temporal_slicing` | `analysis/temporal_slicing/` | TemporalAnalysisResult (per-slice temporal) | Semantic.Temporal, Value.Patterns |
| `cycles` | `analysis/cycles/` | DetectedCycle, BusinessCycleAnalysis | Computational.BusinessRules |
| `validation` | `analysis/validation/` | ValidationResult | Computational.BusinessRules, Value.* |
| `quality_summary` | `analysis/quality_summary/` | ColumnQualitySummary, QualityIssue | Synthesis layer (not entropy source) |

### Gaps in Current Coverage

#### Structural Entropy
| Sub-dimension | Current Coverage | Gap | Implementation Path |
|---------------|-----------------|-----|---------------------|
| Schema.naming_clarity | ❌ Not computed | Need abbreviation detection, naming consistency scoring | Extend semantic agent prompt |
| Schema.naming_consistency | ❌ Not computed | Need cross-table naming pattern analysis | New detector with regex patterns |
| Schema.stability | ❌ Not tracked | Need schema versioning | **Deferred** - requires versioning infrastructure |
| Schema.organization | ❌ Not computed | Need table grouping logic detection | Low priority - minimal LLM impact |
| Types.type_fidelity | ✅ Via `typing` module | Has type candidates, parse success rate | Direct mapping |
| Types.type_consistency | ⚠️ Partial | Need cross-table type consistency check | New detector comparing same-name columns |
| Types.nullability_accuracy | ⚠️ Partial | Statistics has null counts, needs declared vs actual comparison | Extend existing statistics |
| Types.precision_appropriateness | ❌ Not computed | Need precision analysis for decimals | New detector for financial columns |
| Relations.cardinality_clarity | ✅ Via `relationships` | Has cardinality detection | Direct mapping |
| Relations.join_path_determinism | ⚠️ Partial | Need to identify ambiguous paths | New detector using graph_topology |
| Relations.referential_integrity | ✅ Via `relationships.evaluator` | Has orphan detection | Direct mapping |
| Relations.relationship_semantics | ⚠️ Partial | Need semantic labeling of relationships | Extend semantic agent |

#### Semantic Entropy
| Sub-dimension | Current Coverage | Gap | Implementation Path |
|---------------|-----------------|-----|---------------------|
| BusinessMeaning.definition_exists | ⚠️ Partial | Has `business_description` but no glossary integration | Check for non-empty description |
| BusinessMeaning.definition_precision | ❌ Not computed | Need LLM to score definition quality | Extend semantic agent with quality scoring |
| BusinessMeaning.cross_reference_consistency | ❌ Not computed | Need cross-table term consistency check | New detector comparing business_concept usage |
| BusinessMeaning.example_availability | ❌ Not tracked | Need sample values linked to definitions | Add to semantic annotation |
| Units.unit_declared | ✅ Via `typing` module | Has unit detection (Pint) | Direct mapping |
| Units.unit_consistency | ❌ Not computed | Need cross-column unit consistency check | New detector for same business_concept |
| Units.conversion_available | ❌ Not tracked | No currency/unit conversion infrastructure | **Deferred** - requires conversion service |
| Units.scale_clarity | ❌ Not computed | Need magnitude detection (thousands, millions) | Extend semantic agent prompt |
| Temporal.accumulation_type | ⚠️ **Mostly implemented** | `temporal` detects monotonic patterns and resets | Add classification label (ytd/period/all-time) |
| Temporal.point_in_time_clarity | ✅ Via `temporal` | Has `detected_granularity` | Direct mapping |
| Temporal.timezone_handling | ❌ Not computed | No timezone detection | New detector checking timestamp patterns |
| Temporal.period_alignment | ⚠️ Partial | `fiscal_calendar` detection exists | Direct mapping |
| Categorical.hierarchy_definition | ❌ Not computed | Need hierarchy inference | Extend semantic agent for dimension tables |
| Categorical.level_semantics | ❌ Not computed | Need level meaning detection | Part of hierarchy inference |
| Categorical.completeness | ⚠️ Partial | Slicing identifies dimensions, no orphan analysis | New detector for orphan dimension values |
| Categorical.stability | ❌ Not tracked | No category versioning | **Deferred** - requires versioning |

#### Value Entropy
| Sub-dimension | Current Coverage | Gap | Implementation Path |
|---------------|-----------------|-----|---------------------|
| Categories.value_consistency | ❌ Not computed | Need variant grouping (Active/active/A) | New detector with fuzzy matching |
| Categories.vocabulary_completeness | ❌ Not computed | Need enum definition vs actual comparison | Requires vocabulary config |
| Categories.variant_density | ❌ Not computed | Need variant counting per concept | Part of value_consistency detector |
| Categories.coverage | ❌ Not computed | Need recognized vs unknown value ratio | Part of vocabulary_completeness |
| NullSemantics.null_meaning_defined | ❌ Not computed | Need null meaning metadata | Add to semantic annotation |
| NullSemantics.null_consistency | ⚠️ Partial | Statistics has null ratio | Direct mapping with threshold |
| NullSemantics.null_representation | ⚠️ **Data exists** | `config/null_values.yaml` has variants | Wire typing module's null detection to entropy |
| NullSemantics.null_handling_rules | ❌ Not tracked | No aggregation rule metadata | Add to semantic annotation |
| Outliers.outlier_rate | ✅ Via `statistics` | IQR/z-score outlier detection | Direct mapping |
| Outliers.outlier_documentation | ❌ Not tracked | No outlier explanation metadata | Add to quality metadata |
| Outliers.outlier_handling | ❌ Not tracked | No inclusion/exclusion rules | Add to semantic annotation |
| Outliers.detection_method | ✅ Tracked | Method stored in quality metrics | Direct mapping |
| Ranges.bounds_defined | ❌ Not tracked | No declared min/max metadata | Add to semantic annotation |
| Ranges.bounds_enforced | ⚠️ Partial | Statistics has observed min/max | Compare observed vs declared |
| Ranges.distribution_expected | ❌ Not tracked | No expected distribution metadata | Low priority |
| Ranges.boundary_semantics | ❌ Not tracked | No bound meaning metadata | Low priority |
| Patterns.format_consistency | ✅ Via `typing` | Pattern match rate computed | Direct mapping |
| Patterns.pattern_documentation | ⚠️ Partial | Detected patterns stored, not "expected" | Add expected_pattern to annotation |
| Patterns.conformance_rate | ✅ Via `typing` | `pattern_match_rate` available | Direct mapping |
| Patterns.variant_handling | ❌ Not tracked | No normalization rules | Add to semantic annotation |

#### Computational Entropy
| Sub-dimension | Current Coverage | Gap | Implementation Path |
|---------------|-----------------|-----|---------------------|
| DerivedValues.formula_documented | ✅ Via `correlation` | DerivedColumn detection with formula | Direct mapping |
| DerivedValues.reproducibility | ✅ Via `correlation` | match_rate computed | Direct mapping |
| DerivedValues.component_traceability | ✅ Via `correlation` | source_column_ids tracked | Direct mapping |
| DerivedValues.calculation_timing | ❌ Not tracked | No refresh metadata | **Deferred** |
| BusinessRules.rules_documented | ⚠️ Partial | `cycles` detects processes, `validation` has specs | Combine sources |
| BusinessRules.rule_coverage | ❌ Not computed | No coverage analysis | New detector |
| BusinessRules.exception_handling | ❌ Not tracked | No exception metadata | Low priority |
| BusinessRules.rule_versioning | ❌ Not tracked | No rule history | **Deferred** |
| Filters.default_filters_documented | ❌ Not tracked | No default filter metadata | Add to graph metadata |
| Filters.filter_scope | ❌ Not tracked | No filter scope metadata | Part of graph metadata |
| Filters.override_possibility | ❌ Not tracked | No override mechanism | Part of graph execution |
| Filters.filter_rationale | ❌ Not tracked | No filter reasoning | Part of graph metadata |
| Aggregations.aggregation_rules | ❌ Not tracked | No aggregation rule metadata | **Add to SemanticAnnotation** |
| Aggregations.weighting_documented | ❌ Not tracked | No weight column metadata | Part of aggregation rules |
| Aggregations.null_handling_in_agg | ❌ Not tracked | No aggregation null handling | Part of aggregation rules |
| Aggregations.hierarchy_aggregation | ❌ Not tracked | No rollup rule metadata | Part of hierarchy definition |

### Source.Lifecycle.Freshness - Already Implemented

**Important clarification:** The temporal module already computes freshness/staleness:
- `TemporalColumnProfile.is_stale` flag exists
- `UpdateFrequencyAnalysis` detects update patterns
- `staleness_hours` computed from last update

This can be directly mapped to `Source.Lifecycle.freshness` entropy without new implementation.

---

## Part 2: Entropy Layer Architecture

### Core Design Principles

1. **Compute from existing analysis outputs** - No new data scanning required
2. **Testable with static fixtures** - All detectors work on pre-computed analysis results
3. **Compound risk detection** - Identify dangerous dimension combinations
4. **Resolution cascade tracking** - One fix may reduce multiple entropy scores
5. **Use-case-specific thresholds** - Different tolerance for regulatory vs exploratory

### Module Structure

```
src/dataraum_context/
├── entropy/                           # NEW: Entropy layer
│   ├── __init__.py
│   ├── models.py                      # EntropyObject, ResolutionOption, CompoundRisk
│   ├── db_models.py                   # SQLAlchemy persistence
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                    # EntropyDetector ABC, DetectorRegistry
│   │   ├── structural/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py              # NamingClarityDetector, NamingConsistencyDetector
│   │   │   ├── types.py               # TypeFidelityDetector, TypeConsistencyDetector
│   │   │   └── relations.py           # JoinPathDeterminismDetector, CardinalityClarityDetector
│   │   ├── semantic/
│   │   │   ├── __init__.py
│   │   │   ├── business_meaning.py    # DefinitionExistsDetector, DefinitionPrecisionDetector
│   │   │   ├── units.py               # UnitDeclaredDetector, UnitConsistencyDetector, ScaleClarityDetector
│   │   │   ├── temporal.py            # AccumulationTypeDetector, GranularityClarityDetector
│   │   │   └── categorical.py         # HierarchyDefinitionDetector
│   │   ├── value/
│   │   │   ├── __init__.py
│   │   │   ├── categories.py          # ValueConsistencyDetector, VariantDensityDetector
│   │   │   ├── null_semantics.py      # NullMeaningDetector, NullRepresentationDetector
│   │   │   ├── outliers.py            # OutlierRateDetector
│   │   │   ├── ranges.py              # BoundsEnforcedDetector
│   │   │   └── patterns.py            # FormatConsistencyDetector
│   │   └── computational/
│   │       ├── __init__.py
│   │       ├── derived_values.py      # FormulaDocumentedDetector, ReproducibilityDetector
│   │       ├── business_rules.py      # RulesDocumentedDetector
│   │       └── aggregations.py        # AggregationRulesDetector
│   ├── aggregation.py                 # Aggregate entropy across dimensions
│   ├── compound_risk.py               # Detect dangerous dimension combinations
│   ├── resolution.py                  # Resolution hints and cascade effects
│   ├── contracts.py                   # Data readiness contract evaluation
│   ├── scoring.py                     # Compute composite entropy scores
│   ├── context.py                     # Build entropy context for LLM
│   └── processor.py                   # Run all detectors, orchestrate pipeline
├── core/
│   └── formatting/                    # MOVED from quality/formatting/
│       ├── __init__.py
│       ├── base.py                    # SeverityLevel, ThresholdConfig
│       └── config.py                  # YAML threshold loading
```

### Entropy Object Model

See [ENTROPY_MODELS.md](./ENTROPY_MODELS.md) for complete schema specifications.

Key models:
- `EntropyObject` - Core measurement with evidence, resolution options, context
- `ResolutionOption` - Actionable fix with effort and expected entropy reduction
- `CompoundRisk` - Dangerous dimension combination with multiplied impact
- `ResolutionCascade` - Single fix affecting multiple entropy dimensions
- `EntropyContext` - Aggregated entropy for graph agent consumption

### Entropy Detector Interface

```python
class EntropyDetector(ABC):
    """Base class for entropy dimension detectors."""

    @property
    @abstractmethod
    def layer(self) -> str:
        """Return the entropy layer (structural, semantic, value, computational)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Return the dimension this detector measures."""
        pass

    @property
    @abstractmethod
    def sub_dimension(self) -> str:
        """Return the sub-dimension this detector measures."""
        pass

    @abstractmethod
    async def detect(
        self,
        session: AsyncSession,
        table_ids: list[str],
    ) -> list[EntropyObject]:
        """Detect entropy for the given tables.

        Detectors read from existing analysis tables (statistics, semantic, etc.)
        and produce EntropyObject instances. No direct data access required.
        """
        pass
```

---

## Part 3: Implementation Plan

### Phase 1: Foundation (Entropy Core + File Migrations)

**Goal:** Create entropy infrastructure, implement high-value detectors, and migrate files to avoid mid-stream breakage.

#### Step 1.1: File Migrations (Do First)
- Move `quality/formatting/base.py` → `core/formatting/base.py`
- Move `quality/formatting/config.py` → `core/formatting/config.py`
- Update all imports referencing moved files
- Run tests to verify no breakage

#### Step 1.2: Core Models and Storage
- Create `entropy/models.py` with full schema (see [ENTROPY_MODELS.md](./ENTROPY_MODELS.md))
- Create `entropy/db_models.py` with SQLAlchemy models
- Create database migration for entropy tables
- Create `entropy/compound_risk.py` for dangerous combination detection
- Create `entropy/resolution.py` for cascade tracking

#### Step 1.3: Detector Infrastructure
- Create `EntropyDetector` ABC in `entropy/detectors/base.py`
- Create `DetectorRegistry` for detector discovery
- Create `entropy/processor.py` for running all detectors
- Create test fixtures with known entropy characteristics

#### Step 1.4: High-Priority Detectors (Phase 1 Focus)

These detectors directly impact graph agent reliability:

| Detector | Source Data | Why Critical |
|----------|-------------|--------------|
| `TypeFidelityDetector` | `typing/TypeCandidate.parse_success_rate` | Affects all SQL generation |
| `NullRatioDetector` | `statistics/ColumnProfile.null_ratio` | Affects aggregation correctness |
| `OutlierRateDetector` | `statistics/quality.iqr_outlier_ratio` | Affects means/sums reliability |
| `BusinessMeaningDetector` | `semantic/SemanticAnnotation.business_description` | Core for LLM understanding |
| `DerivedValueDetector` | `correlation/DerivedColumn.formula, match_rate` | Risk of hidden calculations |
| `JoinPathDeterminismDetector` | `relationships` + graph analysis | Prevents wrong joins |

#### Step 1.5: Medium-Priority Detectors

| Detector | Source Data | Priority |
|----------|-------------|----------|
| `PatternConsistencyDetector` | `typing/TypeCandidate.pattern_match_rate` | Medium |
| `UnitDeclaredDetector` | `typing/TypeCandidate.detected_unit` | Medium |
| `TemporalClarityDetector` | `temporal/TemporalAnalysisResult` | Medium |
| `RangeBoundsDetector` | `statistics/ColumnProfile.numeric_stats` | Medium |
| `FreshnessDetector` | `temporal/TemporalColumnProfile.is_stale` | Medium |

#### Step 1.6: Compound Risk Detection
- Implement dangerous combination detection (see Part 5)
- Create compound risk scoring that multiplies base entropy
- Generate specific warnings for critical combinations

#### Step 1.7: Aggregation and Scoring
- Implement dimension-level aggregation
- Implement column-level composite scoring
- Implement table-level rollup

### Phase 2: Context Integration + Contract Evaluation

**Goal:** Integrate entropy into graph agent context and implement readiness contracts.

#### Step 2.1: Entropy Context Builder
- Create `entropy/context.py` with `EntropyContext` dataclass
- Extend `graphs/context.py` to include entropy scores
- Add `entropy_scores` and `resolution_hints` to `ColumnContext`
- Add `table_entropy` and `readiness_for_use` to `TableContext`
- Add `relationship_entropy` to `RelationshipContext`

#### Step 2.2: Prompt Formatting
- Create `format_entropy_for_prompt()` function
- Include entropy summary section highlighting risky calculations
- Include dangerous combination warnings
- Include assumption disclosure templates

#### Step 2.3: Contract Evaluation
- Create `entropy/contracts.py` (see [ENTROPY_CONTRACTS.md](./ENTROPY_CONTRACTS.md))
- Load readiness profiles from `config/entropy/contracts.yaml`
- Evaluate current entropy against contract thresholds
- Generate compliance report by use case

#### Step 2.4: Graph Agent Enhancement
- Update graph agent to consume entropy context
- Add entropy-aware SQL generation hints
- Implement query-time behavior (see [ENTROPY_QUERY_BEHAVIOR.md](./ENTROPY_QUERY_BEHAVIOR.md))
- Track assumptions when entropy is high

### Phase 3: Cleanup and API

**Goal:** Remove deprecated modules and expose entropy via API.

#### Step 3.1: Quality Module Cleanup
- Remove `quality/synthesis.py` (verify no remaining usage)
- Remove `quality/db_models.py` and `quality/models.py`
- Delete `quality/` module after all migrations complete
- Update imports throughout codebase

#### Step 3.2: Simplify Topology Module
- Keep core metrics: β₁ (cycles), stability scores
- Remove unused TDA complexity
- Document which outputs feed entropy detection

#### Step 3.3: API Endpoints
- Create `api/routes/entropy.py`
  - `GET /entropy/table/{table_id}` - Full entropy profile
  - `GET /entropy/column/{table_id}/{column_name}` - Column entropy
  - `GET /entropy/contracts/{use_case}` - Readiness status
  - `GET /entropy/resolution-hints` - Prioritized resolution hints

#### Step 3.4: MCP Server
- Implement MCP tools in `mcp/`:
  - `get_entropy_context` - Primary entropy retrieval
  - `get_resolution_hints` - Actionable improvements
  - `accept_assumption` - Record assumption for query
- Implement MCP resources:
  - `entropy://table/{table_name}`
  - `entropy://column/{table}.{column}`
  - `entropy://contract/{use_case}`

### Phase 4: Graph Agent Completion + UI

**Goal:** Finish graph agent with entropy awareness and basic UI.

#### Step 4.1: Field Mapping with Entropy
- Handle ambiguous mappings using entropy scores
- Prefer lower-entropy columns when multiple candidates exist
- Generate warnings for high-entropy field resolutions

#### Step 4.2: Multi-Table Graph Execution
- Use relationship entropy to validate join paths
- Flag non-deterministic joins in generated SQL
- Track join assumptions in execution metadata

#### Step 4.3: Graph Validation
- Validate generated SQL against data quality
- Flag high-uncertainty calculations
- Include entropy context in execution results

#### Step 4.4: UI Foundation
- Migrate web_visualizer to `ui/` directory
- Convert from npm to bun
- Create entropy dashboard component
- Create resolution workflow component

---

## Part 4: Entropy Calculation Formulas

### Column-Level Entropy Score

For each column, calculate dimension-specific scores then aggregate:

```python
column_entropy = weighted_average([
    structural_entropy * 0.25,
    semantic_entropy * 0.30,
    value_entropy * 0.30,
    computational_entropy * 0.15
])
```

### Structural Entropy (Column)
```python
structural = max(
    type_fidelity_entropy,      # 1 - parse_success_rate
    naming_clarity_entropy      # abbreviation_score
)
```

### Semantic Entropy (Column)
```python
semantic = weighted_average([
    business_meaning_entropy,   # 1 if no description, 0 if well-defined
    unit_entropy,               # 1 if numeric with no unit, 0 if unit declared
    temporal_entropy            # Only for timestamp columns
])
```

### Value Entropy (Column)
```python
value = weighted_average([
    null_entropy,               # Scaled null_ratio (high nulls = high entropy)
    outlier_entropy,            # Scaled outlier_ratio
    pattern_entropy,            # 1 - pattern_match_rate
    category_entropy            # Variant count / distinct count (for categoricals)
])
```

### Computational Entropy (Column)
```python
computational = max(
    derived_entropy,            # 1 - match_rate if derived, 0 otherwise
    aggregation_entropy         # 1 if measure with no agg rules, 0 otherwise
)
```

---

## Part 5: Compound Risk Detection

### Dangerous Combinations

Certain dimension combinations create multiplicative risk that exceeds the sum of individual scores:

| Combination | Risk Level | Impact | Multiplier |
|-------------|------------|--------|------------|
| High Semantic.Units + High Computational.Aggregations | **Critical** | Summing unknown currencies with unknown rollup rules | 2.0x |
| High Structural.Relations + High Computational.Filters | **High** | Wrong joins with implicit exclusions | 1.5x |
| High Value.Nulls + High Computational.Aggregations | **High** | Silent exclusion or inclusion errors | 1.5x |
| High Semantic.Temporal + High Value.Ranges | **Medium** | Wrong time periods applied to out-of-range values | 1.3x |

### Detection Algorithm

```python
def detect_compound_risks(
    column_entropy: dict[str, dict[str, float]]  # column -> {dimension: score}
) -> list[CompoundRisk]:
    """Detect dangerous dimension combinations."""
    risks = []

    for column, scores in column_entropy.items():
        # Check for critical: units + aggregations
        if scores.get("semantic.units", 0) > 0.5 and scores.get("computational.aggregations", 0) > 0.5:
            risks.append(CompoundRisk(
                target=column,
                dimensions=["semantic.units", "computational.aggregations"],
                risk_level="critical",
                impact="Summing values with unknown currencies and undefined rollup rules",
                multiplier=2.0,
                combined_score=min(1.0, (scores["semantic.units"] + scores["computational.aggregations"]) * 2.0 / 2)
            ))

        # Check for high: relations + filters
        # ... additional combinations

    return risks
```

### Application to Final Score

When compound risks exist, the effective entropy is:

```python
effective_entropy = base_entropy * max(risk.multiplier for risk in compound_risks)
```

---

## Part 6: Resolution Cascade Tracking

### Concept

Some resolutions fix multiple entropy dimensions simultaneously:

| Resolution | Dimensions Improved | Example |
|------------|---------------------|---------|
| Rename column with unit suffix | Schema.naming_clarity, Units.unit_declared, BusinessMeaning.definition_exists | `c_amt` → `credit_amount_eur` |
| Add FK constraint | Relations.cardinality_clarity, Relations.referential_integrity, Relations.join_path_determinism | Add `orders.customer_id` FK |
| Create semantic view | Relations.join_path_determinism, Filters.default_filters_documented, Aggregations.aggregation_rules | Create `vw_active_customers` |

### Tracking Structure

```python
@dataclass
class ResolutionCascade:
    """A single resolution that improves multiple entropy dimensions."""
    resolution_id: str
    action: str  # e.g., "rename_column"
    parameters: dict
    affected_targets: list[str]  # columns/tables affected
    entropy_reductions: dict[str, float]  # dimension -> expected reduction
    total_reduction: float  # Sum of all reductions
    effort: str  # low, medium, high
    priority_score: float  # total_reduction / effort_factor
```

### Priority Calculation

Resolutions are prioritized by:
```python
priority = total_entropy_reduction / effort_factor
# where effort_factor: low=1, medium=2, high=4
```

---

## Part 7: Success Criteria

### Phase 1 Complete When:
- [ ] File migrations completed (formatting utilities in `core/`)
- [ ] EntropyObject and ResolutionOption models implemented per spec
- [ ] CompoundRisk detection implemented for all critical combinations
- [ ] All 6 high-priority detectors produce valid scores for test dataset
- [ ] All 5 medium-priority detectors produce valid scores
- [ ] Entropy scores have test coverage with known-entropy fixtures
- [ ] No detector crashes on null/empty/edge-case input

### Phase 2 Complete When:
- [ ] GraphExecutionContext includes entropy_scores for all columns
- [ ] format_entropy_for_prompt() produces readable entropy summary
- [ ] Contract evaluation returns compliance status for all 5 use cases
- [ ] Graph agent includes entropy warnings in SQL comments
- [ ] High-entropy assumptions are tracked in execution metadata

### Phase 3 Complete When:
- [ ] Quality module fully removed (no remaining imports)
- [ ] Topology module simplified to core metrics only
- [ ] API endpoints return entropy data
- [ ] MCP tools functional for entropy retrieval

### Phase 4 Complete When:
- [ ] Field mapping prefers lower-entropy columns
- [ ] Multi-table graphs validate join entropy
- [ ] UI displays entropy dashboard
- [ ] End-to-end pipeline runs on test dataset

---

## Part 8: Module Evaluation - Keep/Remove/Merge

### Analysis Modules Assessment

| Module | Verdict | Rationale |
|--------|---------|-----------|
| `typing` | **KEEP** | Core structural entropy source |
| `statistics` | **KEEP** | Core value entropy source |
| `correlation` | **KEEP** | Computational entropy source |
| `relationships` | **KEEP** | Structural entropy source |
| `semantic` | **KEEP** | Semantic entropy source |
| `temporal` | **KEEP** | Temporal entropy source + freshness |
| `topology` | **SIMPLIFY** | Keep β₁ (cycles), stability; remove unused complexity |
| `slicing` | **KEEP** | Useful for categorical entropy |
| `temporal_slicing` | **KEEP** | Different scope: per-slice temporal analysis |
| `cycles` | **KEEP** | Business rule entropy source |
| `validation` | **KEEP** | Business rule entropy source |
| `quality_summary` | **KEEP** | Synthesis layer for UI, separate from entropy |

### Quality Module Disposition

| Component | Action | Timing | Reason |
|-----------|--------|--------|--------|
| `quality/formatting/base.py` | **MOVE** to `core/formatting/` | Phase 1.1 | Useful utilities |
| `quality/formatting/config.py` | **MOVE** to `core/formatting/` | Phase 1.1 | Threshold loading |
| `quality/formatting/*.py` (others) | **EVALUATE** | Phase 3 | May integrate with entropy context |
| `quality/db_models.py` | **REMOVE** | Phase 3 | Superseded by quality_summary |
| `quality/models.py` | **REMOVE** | Phase 3 | Superseded by quality_summary |
| `quality/synthesis.py` | **REMOVE** | Phase 3 | Verify no usage first |
| `quality/__init__.py` | **REMOVE** | Phase 3 | Module deleted |

### Topology Module Simplification

**Keep (feed entropy detection):**
- β₁ (Betti-1): Number of independent cycles - indicates relationship complexity
- Stability metrics (bottleneck distance): How stable is structure over time
- Basic connectivity metrics: Components, isolated tables

**Remove (not used for entropy):**
- Full persistence diagrams
- β₂ (voids) - not meaningful for tabular data
- Complex homology computations

### Phase Script Clarification

The scripts are **NOT duplicates** - they have different scopes:

| Script | Scope | Keep? |
|--------|-------|-------|
| `run_phase4b_correlations.py` | Within-table correlations (Pearson, Spearman, Cramér's V) | **KEEP** |
| `run_phase6_correlation.py` | Cross-table correlations (requires confirmed relationships) | **KEEP** |

---

## Part 9: Additional Considerations

### 9.1 Semantic Agent Extension for Entropy

The semantic agent can be extended to enrich entropy-related fields. Add to `config/prompts/semantic_analysis.yaml`:

**New fields to request:**
```yaml
column_enrichment:
  # Existing fields...

  # New entropy-relevant fields:
  - naming_clarity_score: "Rate 1-5 how clear this column name is (1=cryptic abbreviation, 5=fully self-documenting)"
  - scale_indicator: "If numeric, what scale? (raw, thousands, millions, percent)"
  - accumulation_type: "For time-series values: 'period' (single period), 'ytd' (year-to-date), 'all_time' (running total)"
  - suggested_aggregation: "Recommended aggregation: sum, avg, count, min, max, none"
  - null_interpretation: "What does NULL mean for this column? (not_applicable, unknown, not_yet_set)"
```

### 9.2 Test Fixtures for Entropy

Create fixtures in `tests/entropy/fixtures/`:

```
fixtures/
├── high_entropy_dataset/        # Known high-entropy characteristics
│   ├── tables.json              # Table metadata
│   ├── statistics.json          # Pre-computed statistics
│   ├── semantic.json            # Semantic annotations
│   └── expected_entropy.json    # Expected entropy scores
├── low_entropy_dataset/         # Known low-entropy (clean data)
│   └── ...
└── compound_risk_dataset/       # Known dangerous combinations
    └── ...
```

### 9.3 Config Structure

```
config/
├── entropy/                     # NEW
│   ├── contracts.yaml           # Readiness thresholds by use case
│   ├── compound_risks.yaml      # Dangerous combination definitions
│   └── resolution_templates.yaml # Standard resolution options
├── prompts/
│   └── entropy_enrichment.yaml  # NEW: Entropy-specific prompts
└── ... (existing)
```

---

## Appendix A: Existing Analysis → Entropy Mapping Detail

### typing module → Entropy

```python
# TypeCandidate → Structural.Types.type_fidelity
entropy_score = 1.0 - type_candidate.parse_success_rate

# TypeCandidate → Value.Patterns.format_consistency
entropy_score = 1.0 - (type_candidate.pattern_match_rate or 1.0)

# TypeCandidate → Semantic.Units.unit_declared
if type_candidate.detected_unit:
    entropy_score = 0.0
elif type_candidate.is_numeric:
    entropy_score = 0.8  # Numeric without unit = high entropy
else:
    entropy_score = 0.0  # Non-numeric doesn't need unit
```

### statistics module → Entropy

```python
# ColumnProfile → Value.NullSemantics.null_consistency
null_entropy = min(1.0, profile.null_ratio * 2)  # Scale so 50% nulls = 1.0

# StatisticalQualityMetrics → Value.Outliers.outlier_rate
outlier_entropy = min(1.0, quality.iqr_outlier_ratio * 10)  # Scale so 10% = 1.0

# ColumnProfile → Value.Ranges.bounds_enforced
if numeric_stats:
    # High entropy if observed range is extreme
    range_entropy = detect_range_anomaly(numeric_stats)
```

### semantic module → Entropy

```python
# SemanticAnnotation → Semantic.BusinessMeaning.definition_exists
if not annotation.business_description:
    entropy_score = 1.0
elif len(annotation.business_description) < 20:
    entropy_score = 0.7  # Too brief to be useful
else:
    entropy_score = 0.2  # Has description, moderate confidence

# SemanticAnnotation → Semantic.Categorical.hierarchy_definition
if annotation.semantic_role == "dimension":
    if not has_hierarchy_definition(annotation):
        entropy_score = 0.6  # Dimension without hierarchy
```

### temporal module → Entropy

```python
# TemporalColumnProfile → Source.Lifecycle.freshness
if profile.is_stale:
    entropy_score = 0.8
else:
    # Scale by staleness hours
    entropy_score = min(1.0, profile.staleness_hours / 168)  # 1 week = 1.0

# TemporalAnalysisResult → Semantic.Temporal.accumulation_type
if result.monotonic_increase and result.resets_detected:
    entropy_score = 0.3  # Pattern detected, needs labeling
elif result.monotonic_increase:
    entropy_score = 0.5  # Could be cumulative or just growing
else:
    entropy_score = 0.2  # Likely period values
```

### correlation module → Entropy

```python
# DerivedColumn → Computational.DerivedValues.formula_documented
if derived.formula:
    entropy_score = 1.0 - derived.match_rate  # Low match = high uncertainty
else:
    entropy_score = 1.0  # No formula = maximum uncertainty
```

### relationships module → Entropy

```python
# Graph analysis → Structural.Relations.join_path_determinism
paths = find_paths_between(table_a, table_b)
if len(paths) > 1:
    entropy_score = 0.7  # Ambiguous - multiple valid paths
elif len(paths) == 0:
    entropy_score = 0.9  # No path - cannot join
else:
    entropy_score = 0.1  # Single clear path
```

---

## Appendix B: File Changes Summary

### New Files (Phase 1)
```
src/dataraum_context/entropy/__init__.py
src/dataraum_context/entropy/models.py
src/dataraum_context/entropy/db_models.py
src/dataraum_context/entropy/processor.py
src/dataraum_context/entropy/aggregation.py
src/dataraum_context/entropy/compound_risk.py
src/dataraum_context/entropy/resolution.py
src/dataraum_context/entropy/contracts.py
src/dataraum_context/entropy/scoring.py
src/dataraum_context/entropy/context.py
src/dataraum_context/entropy/detectors/__init__.py
src/dataraum_context/entropy/detectors/base.py
src/dataraum_context/entropy/detectors/structural/*.py
src/dataraum_context/entropy/detectors/semantic/*.py
src/dataraum_context/entropy/detectors/value/*.py
src/dataraum_context/entropy/detectors/computational/*.py
config/entropy/contracts.yaml
config/entropy/compound_risks.yaml
config/prompts/entropy_enrichment.yaml
docs/ENTROPY_MODELS.md
docs/ENTROPY_CONTRACTS.md
docs/ENTROPY_QUERY_BEHAVIOR.md
```

### Moved Files (Phase 1.1)
```
src/dataraum_context/quality/formatting/base.py → src/dataraum_context/core/formatting/base.py
src/dataraum_context/quality/formatting/config.py → src/dataraum_context/core/formatting/config.py
```

### Modified Files (Phase 2)
```
src/dataraum_context/graphs/context.py  # Add entropy to context
src/dataraum_context/graphs/agent.py    # Use entropy in SQL generation
```

### Deleted Files (Phase 3)
```
src/dataraum_context/quality/db_models.py
src/dataraum_context/quality/models.py
src/dataraum_context/quality/synthesis.py
src/dataraum_context/quality/__init__.py
```

---

## Appendix C: Session Tracking

### Tracking Files Structure

```
docs/
├── ENTROPY_IMPLEMENTATION_PLAN.md    # This document (the "what")
├── ENTROPY_MODELS.md                  # Detailed schema specifications
├── ENTROPY_CONTRACTS.md               # Readiness thresholds by use case
├── ENTROPY_QUERY_BEHAVIOR.md          # Agent response policies
├── PROGRESS.md                        # Progress log (the "done")
└── BACKLOG.md                         # Prioritized backlog (the "next")
```

### Document Cross-References

| Document | Purpose | When to Reference |
|----------|---------|-------------------|
| This plan | Implementation roadmap | Planning and task breakdown |
| [ENTROPY_MODELS.md](./ENTROPY_MODELS.md) | Data structures | Implementing models and detectors |
| [ENTROPY_CONTRACTS.md](./ENTROPY_CONTRACTS.md) | Readiness thresholds | Implementing contract evaluation |
| [ENTROPY_QUERY_BEHAVIOR.md](./ENTROPY_QUERY_BEHAVIOR.md) | Agent policies | Implementing graph agent changes |
| [entropy-management-framework.md](../entropy-management-framework.md) | Original spec | Understanding dimension definitions |
| [entropy-framework-future-considerations.md](../entropy-framework-future-considerations.md) | UX/workflow | Future phase planning |
