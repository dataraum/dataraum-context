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

Last updated: 2025-01
