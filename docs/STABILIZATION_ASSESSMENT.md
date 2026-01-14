# Stabilization Assessment

## Current State Analysis

### Dead Code Identified

| Item | Status | Action |
|------|--------|--------|
| `core/formatting/` | **UNUSED** - quality/ deleted | Remove |
| `config/formatter_thresholds/*.yaml` | **UNUSED** - nothing loads them | Remove |
| `sf-hamilton` dependency | **UNUSED** - not imported anywhere | Remove from deps |

### Modules in Active Use

| Module | Used By | Status |
|--------|---------|--------|
| `core/config.py` | 5+ modules (Settings, get_settings) | ✅ Keep |
| `core/models/` | 50+ imports (Result, DataType, etc.) | ✅ Keep |
| `entropy/` | graphs/, tests | ✅ Keep |
| `graphs/` | agent, context | ✅ Keep |
| `analysis/*` | pipeline phases | ✅ Keep |
| `llm/` | semantic, cycles, validation agents | ✅ Keep |
| `sources/` | CSV loading | ✅ Keep |
| `storage/` | all DB models | ✅ Keep |

### Dependencies Assessment

```toml
# KEEP - Core functionality
duckdb>=1.0.0          # Data compute engine
pyarrow>=15.0.0        # Data interchange
sqlalchemy>=2.0.0      # Metadata storage
aiosqlite>=0.19.0      # Async SQLite
pydantic>=2.0.0        # Models
pyyaml>=6.0.0          # Config loading

# KEEP - Analysis
scipy>=1.11.0          # Numerical (topology)
ripser>=0.6.0          # TDA
persim>=0.3.0          # TDA metrics
statsmodels>=0.14.0    # Time series
ruptures>=1.1.0        # Change detection
networkx>=3.6          # Graph algorithms

# KEEP - Utilities
structlog>=24.1.0      # Logging
typer>=0.9.0           # CLI
rich>=13.0.0           # Terminal output

# EVALUATE - Not actively used
sf-hamilton>=1.82.0    # Dataflow orchestration - REMOVE (not imported)
pint>=0.23             # Unit detection - Check if still used
pandas>=2.0.0          # Check if can be optional

# KEEP BUT OPTIONAL
fastapi>=0.110.0       # API (not used yet, planned)
uvicorn>=0.27.0        # API server
mcp>=0.1.0             # MCP (not used yet, planned)
```

---

## Cleanup Tasks

### Task 1: Remove Dead Code

```bash
# Files to remove
rm -rf src/dataraum_context/core/formatting/
rm -rf config/formatter_thresholds/
```

Update `core/__init__.py` to remove formatting exports.

### Task 2: Dependency Cleanup

Remove from `pyproject.toml`:
```toml
# Remove - not used
"sf-hamilton>=1.82.0",
```

Check if `pint` is still used after quality removal:
```bash
grep -r "from pint" src/
grep -r "import pint" src/
```

### Task 3: Evaluate pandas Usage

Check if pandas can be optional:
```bash
grep -r "import pandas" src/
grep -r "from pandas" src/
```

---

## Pipeline Orchestrator Evaluation

### Option A: Custom Orchestrator (Our Own)

**Pros:**
- No new dependencies
- Uses existing SQLite for checkpoints
- Full control over DAG structure
- Simpler to understand

**Cons:**
- More code to maintain
- Need to implement parallel execution ourselves

**Effort:** 2-3 days for basic implementation

### Option B: DBOS

**Pros:**
- Production-ready durable execution
- Automatic recovery from failures
- Decorator-based, minimal code changes
- Good documentation
- Integrates with Pydantic AI

**Cons:**
- Requires Postgres (we currently use SQLite)
- New dependency
- Learning curve

**Effort:** 1-2 days to integrate, but requires Postgres setup

**References:**
- [DBOS GitHub](https://github.com/dbos-inc/dbos-transact-py)
- [DBOS Documentation](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
- [DBOS vs Temporal Comparison](https://www.dbos.dev/blog/durable-execution-coding-comparison)

### Option C: Hybrid Approach

Start with custom orchestrator (simple, SQLite-based), design it so we can swap in DBOS later if needed:

1. Define phases as decorated functions
2. Use our own checkpoint storage initially
3. Abstract the persistence layer
4. Migrate to DBOS when we need Postgres anyway

**Recommendation:** Option C - Start simple, evolve as needed.

---

## Execution Plan

### Phase 1: Cleanup (Stream C) - 1 day

1. Remove `core/formatting/` directory
2. Remove `config/formatter_thresholds/` directory
3. Update `core/__init__.py`
4. Remove `sf-hamilton` from dependencies
5. Check pint/pandas usage
6. Run tests

### Phase 2: Pipeline Foundation - 2 days

1. Create `pipeline/` module structure
2. Define Phase protocol
3. Create checkpoint model
4. Build basic orchestrator

### Phase 3: Phase Migration - 2 days

1. Migrate import phase
2. Migrate typing phase
3. Migrate statistics phase
4. Test parallel execution
5. Migrate remaining phases

### Phase 4: CLI + Integration - 1 day

1. Create CLI interface
2. Add status/reset commands
3. Integration tests

---

## Success Criteria

- [ ] No dead code in codebase
- [ ] All 414 tests pass
- [ ] Pipeline runs end-to-end with checkpointing
- [ ] Parallel phases execute correctly
- [ ] Clear documentation for running pipeline
