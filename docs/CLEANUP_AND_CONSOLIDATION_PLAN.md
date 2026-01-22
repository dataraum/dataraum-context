# Cleanup and Consolidation Plan

A comprehensive audit of the dataraum-context codebase to prepare for API/MCP development.

---

## Executive Summary

| Category | Finding | Priority |
|----------|---------|----------|
| Dead Code | 5 unused functions (~400 lines) | High |
| Print Statements | 37 statements across 5 files | High |
| Rarely Used Models | 6 of 44 tables | Medium |
| Missing Indexes | ~15 tables need indexes | Medium |
| Redundant Columns | ~10 denormalized columns | Low |
| CLI Enhancements | Need metrics, instrumentation | Medium |

---

## 1. Dead Code Removal

### 1.1 Sequential Functions Replaced by Parallel Versions

These functions were replaced during performance optimization but never removed:

| File | Function | Line | Replacement | Lines to Remove |
|------|----------|------|-------------|-----------------|
| `analysis/statistics/profiler.py` | `_profile_column_stats` | 373 | `_profile_column_stats_parallel` (line 49) | ~160 lines |
| `analysis/statistics/quality.py` | `_assess_column_quality` | 515 | `_assess_column_quality_parallel` (line 339) | ~120 lines |
| `analysis/temporal/processor.py` | `_profile_temporal_column` | 312 | `_profile_temporal_column_parallel` (line 113) | ~100 lines |

**Action:** Delete these functions after verifying they're not called anywhere.

### 1.2 Potentially Unused Infrastructure

| File | Function | Line | Assessment |
|------|----------|------|------------|
| `llm/features/_base.py` | `_call_llm` | 41 | Implemented but never called by subclasses. Review if intended for future use. |
| `sources/base.py` | `_get_layer_prefix` | 112 | Utility method never used. `_sanitize_table_name` is used instead. |

**Action:** Investigate if these are planned features or legacy code.

### 1.3 Verification Commands

```bash
# Verify no calls to sequential functions
grep -rn "_profile_column_stats\b" src/ --include="*.py" | grep -v "def _profile_column_stats"
grep -rn "_assess_column_quality\b" src/ --include="*.py" | grep -v "def _assess_column_quality"
grep -rn "_profile_temporal_column\b" src/ --include="*.py" | grep -v "def _profile_temporal_column"
```

---

## 2. Print Statement Conversion

### 2.1 Summary by File

| File | Print Count | Type | Action |
|------|-------------|------|--------|
| `pipeline/runner.py` | 23 | User-facing verbose output | Convert to `logger.info()` with Rich formatting |
| `analysis/cycles/agent.py` | 4 | Debug/progress output | Convert to `logger.debug()` |
| `analysis/semantic/agent.py` | 3 | Informational | Convert to `logger.info()` |
| `analysis/correlation/algorithms/multicollinearity.py` | 3 | Debug (controlled by `_debug` flag) | Convert to `logger.debug()` |
| `analysis/temporal/__init__.py` | 1 | Docstring example | Keep as documentation |

### 2.2 Detailed Conversions

**pipeline/runner.py (23 statements):**
```python
# Before (current)
if config.verbose:
    print("Pipeline Run")
    print("=" * 60)
    print(f"Source: {config.source_path}")

# After (recommended)
if config.verbose:
    logger.info("Pipeline Run")
    logger.info("=" * 60)
    logger.info(f"Source: {config.source_path}")
```

**analysis/cycles/agent.py (4 statements):**
```python
# Before
print(f"   [Iteration {iteration}] stop={response.stop_reason}, tools={len(response.tool_calls)}")
print(f"     → {tool_name}({tool_input})")
print(f"   Warning: Tool output validation failed: {e}")

# After
logger.debug(f"[Iteration {iteration}] stop={response.stop_reason}, tools={len(response.tool_calls)}")
logger.debug(f"→ {tool_name}({tool_input})")
logger.warning(f"Tool output validation failed: {e}")
```

**analysis/semantic/agent.py (3 statements):**
```python
# Before
print(f"   Including within-table correlations: {total_fds} FDs, {total_corrs} numeric correlations")

# After
logger.info(f"Including within-table correlations: {total_fds} FDs, {total_corrs} numeric correlations")
```

**analysis/correlation/algorithms/multicollinearity.py (3 statements):**
```python
# Before (debug flag)
if _debug:
    print(f"DEBUG VDP dim={j}, eig={eigenvalue:.6f}, CI={condition_index:.1f}")

# After
logger.debug(f"VDP dim={j}, eig={eigenvalue:.6f}, CI={condition_index:.1f}")
# Remove _debug flag, use logging level instead
```

### 2.3 Files Needing Logger Import

```python
# Add to each file
import logging
logger = logging.getLogger(__name__)
```

---

## 3. Database Model Audit

### 3.1 Model Categories

**Core Entities (3 tables) - KEEP:**
- `sources` - Data source registry
- `tables` - Table metadata
- `columns` - Column metadata

**Actively Used Analysis (29 tables) - KEEP:**
- Type inference: `type_candidates`, `type_decisions`
- Statistics: `statistical_profiles`, `statistical_quality_metrics`
- Correlations: `column_correlations`, `categorical_associations`, `functional_dependencies`, `derived_columns`
- Relationships: `relationships`
- Temporal: `temporal_column_profiles`, `temporal_table_summaries`
- Semantic: `semantic_annotations`, `table_entities`
- Quality: `column_quality_reports`, `quality_summary_runs`
- Cycles: `business_cycle_analysis_runs`, `detected_business_cycles`
- Entropy: `entropy_objects`, `compound_risks`, `entropy_snapshots`, `entropy_interpretations`
- Graphs: `generated_code`, `graph_executions`, `step_results`
- Pipeline: `pipeline_runs`, `phase_checkpoints`

**Rarely Used (6 tables) - EVALUATE:**

| Table | Purpose | Usage | Recommendation |
|-------|---------|-------|----------------|
| `multicollinearity_groups` | VDP dependency groups | No pipeline phase uses it | DEPRECATE or integrate |
| `correlation_quality_issues` | Generic quality issues | Superseded by specific models | DEPRECATE |
| `validation_runs` | Validation tracking | Incomplete implementation | COMPLETE or DEPRECATE |
| `validation_results` | Validation results | Incomplete implementation | COMPLETE or DEPRECATE |
| `slice_time_matrix_entries` | Cross-slice matrix | Very specialized | Keep, low priority |
| `temporal_topology_analyses` | Temporal topology | Very specialized | Keep, low priority |

### 3.2 Missing Foreign Keys

These tables reference other tables via string IDs but lack FK constraints:

| Table | Column | Should FK To |
|-------|--------|--------------|
| `validation_runs` | `table_ids` (JSON) | tables.table_id (or normalize) |
| `temporal_slice_analyses` | `run_id` | temporal_slice_runs.run_id |
| `temporal_drift_analyses` | `run_id` | temporal_slice_runs.run_id |
| `slice_time_matrix_entries` | `run_id` | temporal_slice_runs.run_id |

**Action:** Add FK constraints for data integrity.

### 3.3 Missing Indexes

Tables that would benefit from indexes on frequently-queried columns:

| Table | Columns | Reason |
|-------|---------|--------|
| `semantic_annotations` | `column_id` | Frequent lookups |
| `table_entities` | `table_id` | Frequent lookups |
| `column_quality_reports` | `source_column_id`, `slice_column_id` | FK columns |
| `quality_summary_runs` | `source_table_id`, `slice_column_id` | FK columns |
| `detected_business_cycles` | `analysis_id` | FK column |
| `slice_definitions` | `table_id`, `column_id` | FK columns |

### 3.4 Redundant/Denormalized Columns

Columns that store data available via relationships:

| Table | Column | Available Via |
|-------|--------|---------------|
| `column_quality_reports` | `column_name` | `source_column.column_name` |
| `column_quality_reports` | `source_table_name` | `source_column.table.table_name` |
| `column_quality_reports` | `slice_column_name` | `slice_column.column_name` |
| `temporal_slice_analyses` | `slice_table_name` | `run.slice_table_name` |
| `temporal_slice_analyses` | `time_column` | `run.time_column` |

**Assessment:** Keep for query performance, but document as intentional denormalization.

### 3.5 Inconsistent Conventions

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Boolean as Integer | `temporal_slicing/*.py` | Use proper `Boolean` type |
| Inconsistent datetime defaults | Various | Standardize to `lambda: datetime.now(UTC)` |
| Missing `updated_at` | Most tables | Add to key analytical tables |

---

## 4. CLI Enhancement Recommendations

### 4.1 Current Commands

| Command | Purpose | Status |
|---------|---------|--------|
| `run` | Execute pipeline | Working |
| `status` | Show phase history | Working |
| `reset` | Delete databases | Working |
| `inspect` | Show graphs | Working |
| `phases` | List phases | Working |

### 4.2 Missing Information in CLI Output

**During `run` command:**
- No per-table progress within phases
- No LLM token usage tracking
- No memory usage monitoring
- No bottleneck identification

**In `status` command:**
- No quality score summary
- No entropy overview
- No relationship count
- No data profiling summary

### 4.3 Proposed New Commands

```bash
# Quality summary
dataraum quality ./pipeline_output
# Shows: quality scores, issue counts, entropy levels

# Data profiling summary
dataraum profile ./pipeline_output
# Shows: row counts, column types, null rates, cardinality

# Relationship overview
dataraum relationships ./pipeline_output
# Shows: detected FKs, join candidates, confidence scores

# Export context
dataraum export ./pipeline_output --format json|markdown
# Exports: full context document for external use
```

### 4.4 Proposed CLI Flags

```bash
# Detailed metrics output
dataraum run /data --metrics metrics.json
# Writes: phase timings, LLM usage, memory stats to JSON

# Profiling mode
dataraum run /data --profile
# Enables: Python cProfile for bottleneck analysis

# Verbose levels
dataraum run /data -v      # INFO level
dataraum run /data -vv     # DEBUG level
dataraum run /data -vvv    # TRACE level (SQL queries, LLM prompts)
```

---

## 5. Data Model Inventory for API/MCP

### 5.1 Primary API Endpoints (Based on Models)

| Endpoint | Source Tables | Description |
|----------|---------------|-------------|
| `GET /sources` | `sources` | List data sources |
| `GET /sources/{id}/tables` | `tables`, `columns` | Tables and columns |
| `GET /tables/{id}/profile` | `statistical_profiles`, `statistical_quality_metrics` | Column statistics |
| `GET /tables/{id}/types` | `type_candidates`, `type_decisions` | Type inference |
| `GET /tables/{id}/correlations` | `column_correlations`, `categorical_associations` | Correlations |
| `GET /tables/{id}/quality` | `column_quality_reports`, `quality_summary_runs` | Quality assessment |
| `GET /relationships` | `relationships` | Cross-table relationships |
| `GET /entropy` | `entropy_objects`, `entropy_snapshots` | Entropy metrics |
| `GET /cycles` | `business_cycle_analysis_runs`, `detected_business_cycles` | Business cycles |
| `GET /pipeline/status` | `pipeline_runs`, `phase_checkpoints` | Pipeline status |

### 5.2 MCP Tools Mapping

| MCP Tool | Source Tables | Use Case |
|----------|---------------|----------|
| `get_context` | All tables via `ExecutionContext` | Primary context for Claude |
| `query` | DuckDB data tables | Execute SQL |
| `get_metrics` | `graph_executions`, `step_results` | Metric values |
| `annotate` | `semantic_annotations` | Update semantics |

### 5.3 Key Data Structures for API

**ContextDocument (Primary Output):**
```python
class ContextDocument:
    tables: list[TableContext]       # From tables, columns, statistical_profiles
    relationships: list[Relationship] # From relationships
    quality: QualityOverview         # From quality_summary_runs, column_quality_reports
    entropy: EntropyOverview         # From entropy_snapshots, entropy_interpretations
    cycles: list[BusinessCycle]      # From detected_business_cycles
```

**Already Implemented:**
- `ExecutionContext` in `graphs/context.py`
- `RichContext` in `graphs/context.py`
- `format_context_for_prompt()` for LLM consumption

---

## 6. Implementation Order

### Phase 1: Dead Code Removal (Day 1)
1. Remove `_profile_column_stats` from `profiler.py`
2. Remove `_assess_column_quality` from `quality.py`
3. Remove `_profile_temporal_column` from `processor.py`
4. Review and decide on `_call_llm` and `_get_layer_prefix`
5. Run full test suite to verify

### Phase 2: Print to Logging (Day 1-2)
1. Add logging imports to 5 files
2. Convert 37 print statements to logger calls
3. Remove `_debug` flag from multicollinearity.py
4. Configure structlog for JSON output option
5. Run tests and verify CLI output

### Phase 3: Database Cleanup (Day 2-3)
1. Add missing FK constraints (4 tables)
2. Add missing indexes (6 tables)
3. Decide on rarely-used tables (keep/deprecate)
4. Add `updated_at` to key tables
5. Run migration and verify

### Phase 4: CLI Enhancement (Day 3-4)
1. Add `--metrics` flag to `run` command
2. Implement `quality` command
3. Implement `profile` command
4. Add verbose levels (-v, -vv, -vvv)
5. Document new features

### Phase 5: Documentation (Day 4-5)
1. Update CLAUDE.md with changes
2. Document data model for API/MCP development
3. Create API design document
4. Create MCP design document

---

## 7. Verification Checklist

After cleanup, verify:

- [ ] All 560+ tests pass
- [ ] Type checking passes (`uv run pyright`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] CLI commands work correctly
- [ ] Pipeline runs successfully on example data
- [ ] No regression in performance (compare timing)
- [ ] Documentation is updated

---

## 8. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Dead code removal | Low | Functions confirmed unused via grep |
| Print to logging | Low | Functional behavior unchanged |
| FK constraints | Medium | May expose data integrity issues |
| New indexes | Low | Performance improvement only |
| Table deprecation | Medium | Defer until confirmed unused |

---

## Appendix A: Full Model Inventory

See the exploration report for complete 44-table inventory with all columns, indexes, and relationships.

## Appendix B: Print Statement Locations

```
analysis/correlation/algorithms/multicollinearity.py:164-166 (3 statements)
analysis/semantic/agent.py:142-149 (3 statements)
analysis/cycles/agent.py:221-396 (4 statements)
pipeline/runner.py:180-336 (23 statements)
analysis/temporal/__init__.py:19 (1 docstring example)
```

## Appendix C: CLI Command Reference

```bash
# Current commands
dataraum run SOURCE [--output PATH] [--phase NAME] [--skip-llm] [--quiet] [--verbose]
dataraum status [OUTPUT_DIR]
dataraum reset [OUTPUT_DIR] [--force]
dataraum inspect [OUTPUT_DIR]
dataraum phases

# Proposed additions
dataraum quality [OUTPUT_DIR]
dataraum profile [OUTPUT_DIR]
dataraum relationships [OUTPUT_DIR]
dataraum export [OUTPUT_DIR] [--format json|markdown]
```
