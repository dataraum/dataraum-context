# Pipeline Orchestrator Plan

## Problem

Current state: 17 separate scripts with manual execution order, no parallelization, no checkpointing.

```
run_phase1_import.py      → must run first
run_phase2_typing.py      → needs phase 1
run_phase3_statistics.py  → needs phase 2
run_phase3b_statistical_quality.py → needs phase 3
run_phase4_relationships.py → needs phase 2
run_phase4b_correlations.py → needs phase 3
run_phase5_semantic.py    → needs phase 3
run_phase6_correlation.py → needs phase 3
run_phase7_temporal.py    → needs phase 2
run_phase7_slicing.py     → needs phase 7
run_phase8_slice_analysis.py → needs phases 7, 8
run_phase9_quality_summary.py → needs many
run_phase10_topology.py   → needs relationships
run_phase11_business_cycles.py → needs phase 7
run_phase12_validation.py → needs semantic
run_graph_agent.py        → needs all analysis
```

## Solution: Simple DAG Orchestrator

### Design Principles

1. **Single DAG definition** - One place to see the whole pipeline
2. **Checkpoint-based** - Skip completed phases on restart
3. **Parallel execution** - Run independent phases concurrently
4. **Testable phases** - Each phase is a callable with typed inputs/outputs
5. **No external dependencies** - Pure Python, uses existing SQLite for state

### DAG Definition

```yaml
# config/pipeline.yaml
phases:
  import:
    function: dataraum_context.pipeline.phases.import_csv
    inputs: []
    outputs: [raw_tables]

  typing:
    function: dataraum_context.pipeline.phases.run_typing
    inputs: [raw_tables]
    outputs: [typed_tables]

  statistics:
    function: dataraum_context.pipeline.phases.run_statistics
    inputs: [typed_tables]
    outputs: [statistical_profiles]

  statistical_quality:
    function: dataraum_context.pipeline.phases.run_statistical_quality
    inputs: [statistical_profiles]
    outputs: [quality_metrics]

  relationships:
    function: dataraum_context.pipeline.phases.run_relationships
    inputs: [typed_tables]
    outputs: [relationships]

  correlations:
    function: dataraum_context.pipeline.phases.run_correlations
    inputs: [statistical_profiles]
    outputs: [correlations]

  semantic:
    function: dataraum_context.pipeline.phases.run_semantic
    inputs: [statistical_profiles]
    outputs: [semantic_annotations]

  temporal:
    function: dataraum_context.pipeline.phases.run_temporal
    inputs: [typed_tables]
    outputs: [temporal_profiles]

  slicing:
    function: dataraum_context.pipeline.phases.run_slicing
    inputs: [temporal_profiles]
    outputs: [slices]

  topology:
    function: dataraum_context.pipeline.phases.run_topology
    inputs: [relationships, correlations]
    outputs: [topology_metrics]

  business_cycles:
    function: dataraum_context.pipeline.phases.run_business_cycles
    inputs: [temporal_profiles]
    outputs: [cycle_analysis]

  validation:
    function: dataraum_context.pipeline.phases.run_validation
    inputs: [semantic_annotations]
    outputs: [validation_results]

  entropy:
    function: dataraum_context.pipeline.phases.run_entropy
    inputs: [statistical_profiles, semantic_annotations, relationships, correlations]
    outputs: [entropy_context]
```

### Dependency Graph (Parallelization)

```
                    ┌─────────┐
                    │ import  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ typing  │
                    └────┬────┘
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │temporal │    │statistics │   │relationships│
    └────┬────┘    └─────┬─────┘   └─────┬─────┘
         │               │               │
    ┌────▼────┐    ┌─────┼─────┐         │
    │ slicing │    │     │     │         │
    └────┬────┘    │     │     │         │
         │    ┌────▼──┐ ┌▼────┐│    ┌────▼────┐
    ┌────▼────┤quality│ │corr ││    │topology │
    │ cycles  └───────┘ └──┬──┘│    └─────────┘
    └─────────┐            │   │
              │       ┌────▼───┴┐
              │       │semantic │
              │       └────┬────┘
              │            │
              │       ┌────▼─────┐
              │       │validation│
              │       └──────────┘
              │
         ┌────▼────────────────────────────┐
         │           entropy               │
         └─────────────────────────────────┘
```

After `typing`, these can run **in parallel**:
- `temporal` → `slicing` → `business_cycles`
- `statistics` → `quality`, `correlations`, `semantic` → `validation`
- `relationships` → `topology`

### Implementation

#### Phase Interface

```python
# src/dataraum_context/pipeline/base.py
from dataclasses import dataclass
from typing import Protocol, Any
from sqlalchemy.ext.asyncio import AsyncSession
import duckdb

@dataclass
class PhaseContext:
    """Context passed to each phase."""
    session: AsyncSession
    duckdb_conn: duckdb.DuckDBPyConnection
    source_id: str
    table_ids: list[str]

@dataclass
class PhaseResult:
    """Result from a phase execution."""
    success: bool
    outputs: dict[str, Any]  # Named outputs
    duration_seconds: float
    error: str | None = None

class Phase(Protocol):
    """Protocol for pipeline phases."""

    async def run(self, ctx: PhaseContext) -> PhaseResult:
        """Execute the phase."""
        ...

    @property
    def name(self) -> str:
        """Phase identifier."""
        ...
```

#### Checkpoint Storage

```python
# src/dataraum_context/pipeline/checkpoints.py
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from dataraum_context.storage.base import Base

class PipelineCheckpoint(Base):
    """Tracks completed pipeline phases."""

    __tablename__ = "pipeline_checkpoints"

    checkpoint_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    phase_name: Mapped[str] = mapped_column(String, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(nullable=False)
    outputs: Mapped[dict] = mapped_column(JSON)  # What was produced
    input_hash: Mapped[str] = mapped_column(String)  # Hash of inputs for invalidation
```

#### Orchestrator

```python
# src/dataraum_context/pipeline/orchestrator.py
import asyncio
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    max_parallel: int = 4
    fail_fast: bool = True
    skip_completed: bool = True

class Orchestrator:
    """Simple DAG orchestrator with parallel execution."""

    def __init__(self, dag: dict, config: PipelineConfig = None):
        self.dag = dag
        self.config = config or PipelineConfig()
        self._completed: set[str] = set()
        self._running: set[str] = set()
        self._failed: set[str] = set()

    async def run(self, ctx: PhaseContext) -> dict[str, PhaseResult]:
        """Run the pipeline, respecting dependencies."""
        results = {}

        while not self._is_complete():
            # Find phases ready to run (dependencies met, not running)
            ready = self._get_ready_phases()

            if not ready:
                if self._running:
                    # Wait for running phases
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Deadlock or all failed
                    break

            # Run ready phases in parallel (up to max_parallel)
            batch = list(ready)[:self.config.max_parallel - len(self._running)]

            tasks = [self._run_phase(name, ctx) for name in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[name] = PhaseResult(False, {}, 0, str(result))
                    self._failed.add(name)
                    if self.config.fail_fast:
                        return results
                else:
                    results[name] = result
                    if result.success:
                        self._completed.add(name)
                    else:
                        self._failed.add(name)

        return results

    def _get_ready_phases(self) -> set[str]:
        """Get phases whose dependencies are met."""
        ready = set()
        for name, phase in self.dag.items():
            if name in self._completed or name in self._running or name in self._failed:
                continue
            deps = phase.get('inputs', [])
            if all(d in self._completed for d in deps):
                ready.add(name)
        return ready
```

### Usage

```python
# Run full pipeline
from dataraum_context.pipeline import run_pipeline

results = await run_pipeline(
    source_id="finance_example",
    tables=["Master_txn_table", "customer_table", ...],
    skip_completed=True,  # Resume from checkpoint
)

# Run specific phases
results = await run_pipeline(
    source_id="finance_example",
    phases=["statistics", "semantic"],  # Only these + dependencies
)

# Run with different config
results = await run_pipeline(
    source_id="finance_example",
    max_parallel=2,  # Limit parallelism
)
```

### CLI

```bash
# Run full pipeline
uv run python -m dataraum_context.pipeline run --source finance_example

# Run specific phase (+ dependencies)
uv run python -m dataraum_context.pipeline run --source finance_example --phase semantic

# Show pipeline status
uv run python -m dataraum_context.pipeline status --source finance_example

# Reset checkpoints
uv run python -m dataraum_context.pipeline reset --source finance_example

# Visualize DAG
uv run python -m dataraum_context.pipeline graph
```

---

## Parallel Work Streams

Given this plan, we can work on multiple streams in parallel:

### Stream A: Pipeline Orchestrator (New)
1. Create `pipeline/` module structure
2. Define Phase protocol and PhaseContext
3. Create PipelineCheckpoint model
4. Build Orchestrator with parallel execution
5. Migrate existing scripts to Phase implementations
6. Create CLI interface

### Stream B: API Endpoints (Backlog 3.3)
1. Create `api/routes/entropy.py`
2. Create `api/routes/pipeline.py` (status, trigger)
3. Wire up FastAPI app
4. Add OpenAPI documentation

### Stream C: Configuration Extraction (Backlog 2.5.1-2.5.2)
1. Extract entropy thresholds to YAML
2. Extract detector multipliers
3. Update code to read from config

### Stream D: Frontend Foundation (Backlog 4.4)
1. Create `ui/` directory structure
2. Set up bun + vite + react/svelte
3. Create entropy dashboard component
4. Create pipeline status component

---

## Implementation Order

```
Week 1:
├── Stream A: Phase protocol, checkpoint model, basic orchestrator
├── Stream B: FastAPI routes skeleton
└── Stream C: threshold.yaml extraction

Week 2:
├── Stream A: Migrate 3-4 phases, test parallel execution
├── Stream B: Entropy endpoints, pipeline status endpoint
└── Stream D: UI scaffold, API client

Week 3:
├── Stream A: Migrate remaining phases, CLI
├── Stream B: Remaining endpoints
└── Stream D: Entropy dashboard, pipeline viewer
```

---

## Benefits

1. **Testability**: Each phase is a function with typed inputs/outputs
2. **Visibility**: Pipeline status in DB, queryable via API
3. **Parallelism**: 3-4x faster full runs
4. **Resumability**: Restart from any checkpoint
5. **Simplicity**: No external workflow engine, pure Python
6. **Maintainability**: One DAG definition, not 17 scripts
