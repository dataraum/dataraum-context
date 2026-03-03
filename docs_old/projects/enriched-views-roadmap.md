# Enriched Views Roadmap

Living document for ideas around enriched views, star schema flattening, and analyses that benefit from pre-joined data.

## Current State (March 2026)

### What works

- **Enriched views**: `enriched_{fact}` = fact table LEFT JOINed with confirmed dimension tables via grain-safe cardinalities (many-to-one, one-to-one). Grain verified via COUNT(*) post-creation.
- **Enrichment agent**: LLM recommends which dimensions to join and which columns to include (high/medium value filtering). Falls back to deterministic join selection. Enabled in config, wired up, working.
- **Column naming**: `{fk_column}__{dim_column}` handles multiple FKs to same dimension without collision.
- **Slicing on enriched dimensions**: LLM can recommend slicing by columns that only exist in the enriched view (e.g., customer country from a dimension table). Full chain: slicing_phase -> slicing_view_phase -> slice_analysis_phase -> temporal_slice_analysis_phase.
- **Graph execution context**: Enriched views surface in `GraphExecutionContext` as `EnrichedViewContext`, formatted for the graph/query agent prompt.

### What's dead or incomplete

| Item | Location | Status |
|------|----------|--------|
| `EnrichedView.view_table_id` | `analysis/views/db_models.py:44` | Never populated. Intended to register view as `Table(layer="enriched")`. Low impact — downstream queries use `EnrichedView` directly. |
| `topology_similarity` | `analysis/relationships/utils.py` | Always 0.0. Referenced in semantic agent prompt but never computed from TDA results. Adds no signal. |
| `cross_table_quality` phase | `pipeline/phases/cross_table_quality_phase.py` | De-configured in `pipeline.yaml`. Does its own ad-hoc JOINs instead of reusing enriched views. |

### Current limitation: 1 fact table per view

Each `enriched_{fact}` flattens one fact table with its dimensions. No support for joining two fact tables sharing dimensions (e.g., orders + returns both joining customers). The data model supports multiple independent enriched views — they just can't be combined yet.

---

## Near-term Ideas (leverage existing infrastructure)

### 1. Cross-table derived columns on enriched views

**What**: Run `detect_derived_columns()` on `enriched_{fact}` instead of only on typed tables.

**Why**: Catches derivations like `order_total = quantity * unit_price` where `unit_price` comes from a joined product dimension. The detection algorithm already works — it just needs the enriched view as input instead of (or in addition to) the typed table.

**Effort**: Small. Wire `enriched_{fact}` view name into the correlations phase's derived column detection. May need to handle the FK-prefixed column names in formula output.

**Value**: High for financial/transactional data where totals are commonly derived from dimension attributes.

### 2. Rewire cross_table_quality to use enriched views

**What**: The de-configured `cross_table_quality` phase does its own JOINs per relationship. Refactor it to query enriched views instead.

**Why**: Simplifies code (no ad-hoc JOIN construction), guarantees grain safety (enriched views already verified), and makes the phase ready to re-enable.

**Effort**: Medium. Need to map relationship pairs to enriched view columns, filter to numeric columns, run correlation analysis.

**Value**: Re-enables a disabled phase. Cross-table correlations are genuinely useful — finding that `customer_age` correlates with `order_value` across a join is the kind of insight that single-table analysis misses.

### 3. Dimension coverage analysis

**What**: For each enriched dimension column, report what percentage of fact rows have NULL (unmatched FK).

**Why**: High NULL rates indicate referential integrity issues. The current RI metrics exist in relationship evidence but aren't surfaced in a user-friendly way. This is a simple `SELECT COUNT(*) FILTER (WHERE col IS NULL)` per dimension column.

**Effort**: Small. Could run as part of enriched_views phase (post-view-creation) or as a lightweight sub-step.

**Value**: Direct data quality signal. "15% of orders have no matching customer" is actionable.

### 4. Enriched view statistics

**What**: Run the regular statistics profiler on enriched dimension columns and store results in `StatisticalProfile`.

**Why**: Currently, enriched dimension columns get ad-hoc stats during slicing but they're not persisted. Storing them lets entropy detectors and quality summaries reason about dimension distributions in fact-table context (e.g., "80% of orders go to 3 countries" is different from "the countries table has 50 entries").

**Effort**: Medium. Need to register enriched view columns properly (or use a separate profile table) so they don't collide with base table column profiles.

**Value**: Enables downstream phases to reason about dimension column distributions without re-querying DuckDB.

---

## Medium-term Ideas (new analyses)

### 5. Conditional distribution profiles

**What**: For each enriched dimension column with low cardinality (the slice dimensions), compute how fact-table metrics distribute per dimension value.

**Why**: Richer than slicing. Slicing gives you "here's the data for US customers." This gives you "revenue is 60% US / 25% EU / 15% APAC" as a compact profile. It's the difference between "filter to segment" and "profile by segment."

**How**: GROUP BY dimension column, compute basic stats (count, sum, mean, stddev) on numeric fact columns. Store as a new model (`DimensionProfile` or similar).

**Value**: Surfaces concentration risk (95% of revenue from one region), distribution skew, and segment-level anomalies without materializing full slice tables.

### 6. Cross-dimension interaction effects

**What**: When a fact table joins 2+ dimensions, detect surprising combinations.

**Why**: "95% of Electronics sales go to US" might be expected — or might indicate a data pipeline that only loads US electronics. Interaction analysis surfaces these patterns.

**How**: Contingency tables on pairs of dimension columns, chi-squared test for independence, Cramer's V for effect size. Flag combinations where observed frequency deviates significantly from expected (under independence assumption).

**Effort**: Medium-high. Need to decide which dimension pairs to test (combinatorial explosion with many dimensions), handle sparse contingency tables.

**Value**: Catches data quality issues invisible to single-dimension analysis. Also useful for business insight (market concentration, product-channel affinity).

### 7. Temporal trends on enriched dimensions

**What**: If the fact table has a time column, compute how dimension distributions shift over time.

**Why**: "Customer region mix was 50/30/20 in Q1 but shifted to 40/40/20 in Q2" is a business-relevant signal. The temporal_slice_analysis phase already does drift detection on slices — this would be drift detection on dimension distributions within the full fact table.

**How**: GROUP BY time_period, dimension_column, compute distribution per period, JS divergence between periods. Similar to temporal_slice_analysis but operating on the enriched view directly rather than on individual slices.

**Difference from temporal_slice_analysis**: That phase analyzes drift within each slice. This would analyze drift of the dimension distribution itself across the full dataset.

---

## Longer-term Ideas (multi-fact-table)

### 8. Shared dimension detection

**What**: Detect when two fact tables join the same dimension table (possibly via different FK columns).

**Why**: Prerequisite for federation. "orders.customer_id -> customers" and "returns.customer_id -> customers" share a dimension. This enables cross-fact queries like "return rate by customer segment."

**How**: Query the relationship graph for dimension tables connected to 2+ fact tables. Verify join compatibility (same PK column in dimension, compatible cardinalities).

**Effort**: Small analysis, but the implications are large (drives federation design).

### 9. Conformed dimension verification

**What**: Before federating, verify that shared dimensions have consistent semantics across fact tables.

**Why**: Two fact tables might join `customers` but one uses `customer_id` (current customers) and another uses `billing_customer_id` (historical). The joins work but the semantics differ.

**How**: Compare join column value distributions, check temporal alignment, verify referential integrity from both fact tables. Surface as a compatibility score.

### 10. Federated enriched views

**What**: Create views that combine two fact tables through shared dimensions.

**Why**: Enables cross-fact analysis: "return rate by product category", "average days between order and return by region."

**How**: Not a simple UNION or JOIN — need to think about grain carefully. Options:
- **Dimension-grain aggregation**: Aggregate both fact tables to dimension grain, then join. `orders_by_customer JOIN returns_by_customer ON customer_id`.
- **Wide fact view**: If fact tables share grain (same entity, same time period), FULL OUTER JOIN them.
- **Metric computation layer**: Don't materialize a view — instead, let the graph execution agent compose queries that reference both enriched views.

**Risk**: Grain mismatches, fan traps, performance on large datasets.

### 11. Cross-fact derived metrics

**What**: Detect derivations across fact tables. `return_rate = returns.count / orders.count` grouped by shared dimensions.

**Why**: These are the highest-value business metrics and they can't be found by single-table analysis.

**How**: Requires shared dimension detection (idea 8) + federation (idea 10). The derived column algorithm would need to work on aggregated data rather than row-level data.

---

## Open Questions

- Should enriched views register as `Table(layer="enriched")` records? Pro: uniform discovery. Con: columns aren't real `Column` records, could cause confusion in phases that expect typed table columns.
- Should dimension coverage (idea 3) be an entropy detector rather than a pipeline phase? It's a data quality signal that fits the entropy model well.
- For cross-table correlations (idea 2), should we compute on the full enriched view or only on the dimension columns that the enrichment agent rated as "high value"?
- How should conditional distribution profiles (idea 5) interact with the existing slicing system? They're complementary but risk information overlap.

---

## Revision History

- 2026-03-03: Initial document from codebase exploration
