# Backlog

Prioritized backlog for the dataraum-context project. Items are organized by priority and dependency.

---

## Priority 1: Entropy Core (Current Focus)

Foundation work that everything else depends on.

- [ ] Create `entropy/models.py` with EntropyObject, ResolutionOption, EntropyContext
- [ ] Create `entropy/db_models.py` with SQLAlchemy models
- [ ] Create database migration for entropy tables
- [ ] Create `entropy/detectors/base.py` with EntropyDetector ABC
- [ ] Implement TypeFidelityDetector (from typing module data)
- [ ] Implement NullSemanticsDetector (from statistics module data)
- [ ] Implement OutlierRateDetector (from statistics module data)
- [ ] Implement BusinessMeaningDetector (from semantic module data)
- [ ] Implement DerivedValueDetector (from correlation module data)
- [ ] Implement JoinPathDeterminismDetector (from relationships module data)

---

## Priority 2: Cleanup & Integration

Can be done in parallel with Priority 1.

### Cleanup
- [ ] Move `quality/formatting/base.py` to `core/formatting/`
- [ ] Move `quality/formatting/config.py` to `core/formatting/`
- [ ] Evaluate remaining quality/formatting/* files
- [ ] Remove `quality/synthesis.py` (superseded by quality_summary)
- [ ] Remove `quality/db_models.py` and `quality/models.py`
- [ ] Delete quality/ module after migration complete
- [ ] Simplify `analysis/topology/` - keep core metrics only

### Context Integration
- [ ] Add entropy scores to `graphs/context.py` GraphExecutionContext
- [ ] Create `entropy/context.py` for entropy context building
- [ ] Create `format_entropy_for_prompt()` function
- [ ] Update graph agent to consume entropy context

---

## Priority 3: Semantic Agent Extension

Extend semantic agent to enrich entropy-related fields.

- [ ] Add naming_clarity analysis to semantic prompts
- [ ] Add scale_clarity detection (thousands, millions)
- [ ] Add hierarchy inference for dimension tables
- [ ] Add stock vs flow classification for temporal columns
- [ ] Add cross-table type consistency check

---

## Priority 4: Graph Agent Completion

Finish the graph agent with entropy awareness.

- [ ] Complete field mapping (business concept → column)
- [ ] Handle ambiguous mappings using entropy scores
- [ ] Support multi-table graph execution
- [ ] Add graph validation with entropy-based warnings
- [ ] Track assumptions when entropy is high

---

## Priority 5: API & MCP Server

Expose functionality via HTTP and MCP.

- [ ] Create `api/routes/entropy.py`
- [ ] Create `api/routes/graphs.py`
- [ ] Create `api/routes/context.py`
- [ ] Implement MCP server in `mcp/`
- [ ] Implement `get_entropy_context` MCP tool
- [ ] Implement `get_resolution_hints` MCP tool
- [ ] Implement `execute_graph` MCP tool

---

## Priority 6: UI Foundation

Migrate and extend the UI prototype.

- [ ] Create `ui/` directory at project root
- [ ] Migrate web_visualizer from prototypes/calculation-graphs/
- [ ] Convert from npm to bun
- [ ] Create entropy dashboard component
- [ ] Create resolution workflow component
- [ ] Integrate with API endpoints

---

## Blocked/Waiting

Items that depend on external work or decisions.

- [ ] Query entropy layer (waiting for query agent implementation)
- [ ] Source entropy layer (waiting for provenance infrastructure)
- [ ] Schema stability tracking (waiting for schema versioning)

---

## Technical Debt

Items that should be addressed when time permits.

- [ ] Consolidate phase scripts (phase 4b vs phase 6 overlap)
- [ ] Clarify temporal vs temporal_slicing module responsibilities
- [ ] Review formatter_thresholds/ config usage
- [ ] Add tests for entropy detectors
- [ ] Add integration tests for entropy + graph agent

---

## Notes

- Items with [ ] are not started
- Items with [~] are in progress
- Items with [x] are completed (move to PROGRESS.md)
- Dependencies are indicated by item ordering within each priority
