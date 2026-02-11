# Transformation Graphs - Backlog

This document tracks planned improvements and features for the unified transformation graphs module.

## Completed

- [x] Unified agent architecture (GraphAgent handles both filters and metrics)
- [x] LLM-based SQL generation from graph specifications
- [x] In-memory caching for generated code
- [x] Database persistence for generated code (GeneratedCodeRecord)
- [x] SQL validation before execution
- [x] Table schema extraction for LLM context
- [x] Tests for GraphAgent

## Important (High Priority)

### Cache Invalidation

**Problem:** Currently, cached generated code is never invalidated. If the graph definition changes or the table schema changes, stale SQL may be used.

**Solution:**
1. Add `graph_hash` field to GeneratedCodeRecord (hash of graph YAML)
2. Add `schema_hash` field (hash of table column names + types)
3. On cache lookup, verify hashes match before using cached code
4. If hash mismatch, regenerate and update cache

**Files to modify:**
- `src/dataraum_context/graphs/agent.py` - Add hash computation and verification
- `src/dataraum_context/storage/models_v2/graphs.py` - Add hash columns

### Filter Integration

**Problem:** Metrics should execute on filtered data (clean_* views), not raw data. Need to link filter executions to metric executions.

**Solution:**
1. Add `filter_execution_id` parameter to `ExecutionContext`
2. When filter_execution_id provided, execute SQL against clean_* view
3. Link metric execution to filter execution in GraphExecutionRecord
4. Support chaining: filter graph -> metric graph with automatic view selection

**Files to modify:**
- `src/dataraum_context/graphs/agent.py` - Add filter execution linking
- `src/dataraum_context/graphs/models.py` - Add filter_execution_id to ExecutionContext
- `src/dataraum_context/storage/models_v2/graphs.py` - Add FK relationship

### Schema Mapping Persistence

**Problem:** LLM-generated column mappings (abstract_field -> concrete_column) are cached per graph but not reusable across graphs.

**Solution:**
1. Persist SchemaMapping objects separately from generated code
2. Create DatasetSchemaMapping record per dataset
3. Allow user override of LLM-generated mappings
4. Share mappings across graphs operating on same dataset

**Files to modify:**
- `src/dataraum_context/storage/models_v2/graphs.py` - Add SchemaMappingRecord
- `src/dataraum_context/graphs/agent.py` - Query shared mappings before generating

### Context Integration for GraphAgent

**Problem:** GraphAgent currently only receives basic table schema (column names + types). Rich metadata from the enrichment and quality pipelines is not utilized, leading to suboptimal SQL generation.

**Key Insight:** The `quality/formatting` package contains **LLM-ready formatters** that transform raw metrics into structured context with severity levels, interpretations, and recommendations. GraphAgent should use these formatters rather than raw data.

---

#### Available Raw Context (already computed)

| Context Type | Source Module | Key Data | Use Case |
|--------------|---------------|----------|----------|
| Relationships | `enrichment/relationships` | join paths, cardinality, confidence | JOIN construction |
| Semantic | `storage/models_v2/semantic_context` | semantic_role, entity_type, business_name | Field mapping |
| Temporal | `storage/models_v2/temporal_context` | granularity, seasonality, staleness | Time filtering |
| Statistical | `storage/models_v2/statistical_context` | null_ratio, outliers, cardinality | NULL handling |
| Topological | `storage/models_v2/topological_context` | betti numbers, cycles | Schema understanding |
| Quality Issues | `quality/context` | flags, issues, recommendations | Warnings/filters |

---

#### Available Formatters (quality/formatting package)

These formatters transform raw metrics into **pre-interpreted, LLM-ready output** with:
- Severity levels (none/low/moderate/high/severe/critical)
- Natural language interpretations
- Actionable recommendations
- Configurable thresholds (YAML-based)

| Formatter | Function | Output |
|-----------|----------|--------|
| **Statistical** | `format_statistical_quality()` | Completeness (nulls), outliers (IQR, Isolation Forest), Benford compliance |
| **Temporal** | `format_temporal_quality()` | Freshness/staleness, temporal completeness, seasonality, trends, change points, distribution stability |
| **Topological** | `format_topological_quality()` | Structure (Betti numbers), cycles (business/anomalous), complexity, topology stability |
| **Domain** | `format_domain_quality()` | Compliance scores, double-entry balance, sign conventions, fiscal periods |
| **Multicollinearity** | `format_multicollinearity_for_llm()` | VIF per column, condition index, dependency groups with recommendations |
| **Cross-Table** | `format_cross_table_multicollinearity_for_llm()` | Cross-table dependencies, join paths, unified condition index |
| **Business Cycles** | `format_business_cycles_for_llm()` | AR/AP/Revenue/Expense cycles, completeness, business value |

**Example formatter output (temporal):**
```python
{
    "temporal_quality": {
        "overall_severity": "moderate",
        "groups": {
            "freshness": {
                "severity": "low",
                "interpretation": "Data is slightly stale but still acceptable",
                "metrics": {
                    "staleness_days": {
                        "value": 3.5,
                        "severity": "low",
                        "interpretation": "Data is 3.5 days old"
                    }
                }
            },
            "completeness": {...},
            "patterns": {...},
            "stability": {...}
        }
    }
}
```

---

#### Solution

1. **Relationship Context** - Use `gather_relationships()` + graph analysis:
   - `EnrichedRelationship[]` with cardinality, confidence, detection_method
   - `GraphAnalysisResult` with business cycles (AR, AP, Revenue, etc.)
   - Format with `format_business_cycles_for_llm()` for cycle context

2. **Quality Context** - Use formatters from `quality/formatting`:
   - `format_statistical_quality()` → null handling, outlier awareness
   - `format_temporal_quality()` → time filtering, granularity
   - `format_topological_quality()` → schema fragmentation, cycles
   - `format_domain_quality()` → financial compliance (if applicable)
   - `format_multicollinearity_for_llm()` → column redundancy warnings

3. **Semantic Context** - Query `SemanticAnnotation` + `TableEntity`:
   - `semantic_role`: identifier/measure/dimension → aggregation decisions
   - `entity_type`: customer/product/transaction → field mapping
   - `is_fact_table`/`is_dimension_table` → schema understanding

**Implementation Steps:**

1. Create `GraphAgentContext` dataclass combining formatted outputs
2. Add `gather_agent_context(table_ids, session, duckdb_conn)` that:
   - Queries raw metrics from DB
   - Passes through appropriate formatters
   - Returns unified context structure
3. Update prompt template with sections for each context type
4. Modify `_get_table_schema()` to call `gather_agent_context()`
5. GraphAgent receives pre-interpreted context with severity + recommendations

**Example: Context-Aware SQL Generation**

```
Input: "Calculate customer lifetime value"

Formatted Context:
- Relationships: customers → orders [1:N, 0.95 confidence]
- Semantic: orders.amount is 'measure', customers.customer_id is 'identifier'
- Statistical: {
    "overall_severity": "low",
    "groups": {
      "completeness": {
        "severity": "low",
        "interpretation": "5% nulls in orders.amount - minor impact",
        "recommendation": "Consider COALESCE for aggregations"
      }
    }
  }
- Temporal: {
    "freshness": {"severity": "none", "interpretation": "Data is current"}
  }

Generated SQL:
SELECT c.customer_id,
       SUM(COALESCE(o.amount, 0)) as lifetime_value  -- COALESCE per recommendation
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id  -- 1:N from relationships
GROUP BY c.customer_id
/* Quality note: 5% nulls handled with COALESCE */
```

**Files to modify:**
- `src/dataraum_context/graphs/agent.py` - Add context gathering, use formatters
- `src/dataraum_context/graphs/models.py` - Add GraphAgentContext dataclass
- `config/prompts/graph_sql_generation.yaml` - Extend prompt with formatted context sections

**Files to use (formatters):**
- `src/dataraum_context/quality/formatting/statistical.py` - `format_statistical_quality()`
- `src/dataraum_context/quality/formatting/temporal.py` - `format_temporal_quality()`
- `src/dataraum_context/quality/formatting/topological.py` - `format_topological_quality()`
- `src/dataraum_context/quality/formatting/domain.py` - `format_domain_quality()`
- `src/dataraum_context/quality/formatting/multicollinearity.py` - `format_multicollinearity_for_llm()`
- `src/dataraum_context/quality/formatting/business_cycles.py` - `format_business_cycles_for_llm()`

**Files to use (raw data):**
- `src/dataraum_context/enrichment/relationships/models.py` - EnrichedRelationship
- `src/dataraum_context/enrichment/relationships/gathering.py` - gather_relationships()
- `src/dataraum_context/quality/context.py` - format_dataset_quality_context()

## Nice to Have (Lower Priority)

### Visualization

**Problem:** Users want to see the generated SQL alongside the graph definition in the React Flow visualization.

**Solution:**
1. Add `generated_sql` field to export_to_react_flow output
2. Show SQL in a collapsible panel below each step node
3. Highlight which abstract fields mapped to which columns

**Files to modify:**
- `src/dataraum_context/graphs/export.py` - Include SQL in export
- Frontend components (external)

### Q&A Integration

**Problem:** Users want to query past executions and get explanations of calculations.

**Solution:**
1. Add MCP tool `get_metric_explanation(metric_id, execution_id)`
2. Return: graph definition + generated SQL + actual values + interpretation
3. Add MCP tool `list_metric_executions(metric_id, limit)`
4. Enable time-series views of metric values

**Files to create:**
- `src/dataraum_context/mcp/tools/metrics.py` - MCP tools for metrics
- `src/dataraum_context/graphs/explanation.py` - Explanation generator

### Multi-Table Support

**Problem:** Current implementation assumes single table. Many metrics span multiple tables with joins.

**Solution:**
1. Extend TableSchema to include relationships between tables
2. Pass multiple table schemas to LLM
3. Allow graph steps to reference columns from different tables
4. LLM generates JOINs as needed

**Note:** This is enabled by "Context Integration for GraphAgent" (High Priority) which provides relationship data with join paths, cardinality, and confidence scores.

**Files to modify:**
- `src/dataraum_context/graphs/agent.py` - Support multi-table context
- `config/prompts/graph_sql_generation.yaml` - Update prompt for joins

### Parameterized Graphs

**Problem:** Graphs with runtime parameters (e.g., DSO for specific customer) need parameter injection into SQL.

**Solution:**
1. Validate parameters against graph.parameters definition
2. Pass parameters to LLM context
3. LLM generates parameterized SQL with placeholders
4. Agent substitutes parameter values safely (preventing SQL injection)

**Files to modify:**
- `src/dataraum_context/graphs/agent.py` - Add parameter validation and injection
- `config/prompts/graph_sql_generation.yaml` - Update prompt for parameters

### Incremental Execution

**Problem:** For large datasets, re-executing entire metric on every call is expensive.

**Solution:**
1. Track last execution timestamp per metric
2. Detect new/changed rows since last execution
3. Compute delta and merge with previous result
4. Applicable for additive aggregations (SUM, COUNT, etc.)

**Files to create:**
- `src/dataraum_context/graphs/incremental.py` - Incremental execution logic

### Graph Composition

**Problem:** Complex metrics are built from simpler metrics. Currently no way to compose graphs.

**Solution:**
1. Allow `depends_on: [other_graph_id]` in graph definition
2. Agent executes dependencies first
3. Results from dependencies available as CTEs
4. Enable metric reuse without duplication

**Files to modify:**
- `src/dataraum_context/graphs/models.py` - Add depends_on field
- `src/dataraum_context/graphs/agent.py` - Handle dependency resolution

## Technical Debt

### Prompt Optimization

**Issue:** Current prompt may be suboptimal for specific LLM providers.

**Action:**
- Test with different models (Claude, GPT-4, Llama)
- Measure SQL quality and token usage
- Create provider-specific prompt variants if needed

### Error Recovery

**Issue:** If LLM generates invalid SQL, we fail. Could retry with error context.

**Action:**
- Catch SQL validation errors
- Feed error message back to LLM with original context
- Limit retries (e.g., 2 attempts)
- Log failed generations for analysis

### Test Coverage

**Issue:** Integration tests use mocked LLM. Need real LLM tests.

**Action:**
- Create integration tests with real LLM calls (marked slow)
- Test with sample datasets from examples/data/
- Verify generated SQL is semantically correct

---

## Notes

This backlog was created to track items deferred during the initial implementation of the unified graph architecture. Items should be prioritized based on user needs and stability of the core framework.

Last updated: 2025-12-15
