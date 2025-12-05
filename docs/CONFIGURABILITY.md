# Configurability Architecture

## Executive Summary

The DataRaum Context Engine has strong foundations for configurability but currently implements only a fraction of its potential. This document analyzes three dimensions of configurability: **Sources**, **Domains**, and **Pipeline Flow**, with recommendations based on industry best practices.

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [A) Sources: Loader Plugin Architecture](#a-sources-loader-plugin-architecture)
3. [B) Domains: Ontology Framework](#b-domains-ontology-framework)
4. [C) Pipeline Flow: Step Registry](#c-pipeline-flow-step-registry)
5. [D) Workflow Engine Abstraction](#d-workflow-engine-abstraction)
6. [E) Step Packages: Installable Components](#e-step-packages-installable-components)
7. [F) Additional Abstraction Layers](#f-additional-abstraction-layers)
8. [Dependency Injection](#dependency-injection-recommendation)
9. [Package Architecture](#package-architecture)
10. [Documentation Gaps](#documentation-gaps-to-address)
11. [Implementation Priorities](#implementation-priorities)
12. [References](#references)

---

## Current State Assessment

### What Already Exists

| Dimension | Current Implementation | Configurability Level |
|-----------|----------------------|----------------------|
| **Sources** | `LoaderBase` ABC, only CSVLoader | Low - Pattern exists, single implementation |
| **Domains** | YAML ontologies, only financial_reporting | Medium - Structure exists, one example |
| **Pipeline** | Hardcoded 8-stage sequence in `run_pipeline()` | Low - All stages mandatory, fixed order |
| **Configuration** | Pydantic Settings + YAML files | High - Well-structured config hierarchy |

### Existing Extension Points

The codebase already has solid abstractions:
- **`Result[T]`** monad for error handling
- **`LoaderBase`** ABC for source loaders
- **`SourceConfig`** flexible source definition
- **YAML configuration** hierarchy in `config/`
- **5-pillar metadata model** (statistical, semantic, topological, temporal, domain)

---

## A) Sources: Loader Plugin Architecture

### Current State

```
staging/
├── base.py          # LoaderBase ABC, TypeSystemStrength enum
└── loaders/
    └── csv.py       # CSVLoader (only implementation)
```

The `LoaderBase` defines:
- `type_system_strength` property (UNTYPED, WEAK, STRONG)
- `load()` async method → `Result[StagingResult]`
- `get_schema()` async method → `Result[list[ColumnInfo]]`

### Recommendation: Entry Point Plugin System

Based on [Stevedore](https://github.com/openstack/stevedore) and Python entry points, implement a loader registry:

```python
# staging/registry.py
from stevedore import driver

class LoaderRegistry:
    """Registry for source loaders using entry points."""

    NAMESPACE = "dataraum.loaders"

    def __init__(self):
        self._cache: dict[str, type[LoaderBase]] = {}

    def get_loader(self, source_type: str) -> LoaderBase:
        """Get loader instance by source type."""
        if source_type not in self._cache:
            mgr = driver.DriverManager(
                namespace=self.NAMESPACE,
                name=source_type,
                invoke_on_load=False,
            )
            self._cache[source_type] = mgr.driver
        return self._cache[source_type]()

    def list_loaders(self) -> list[str]:
        """List available loader types."""
        from stevedore import ExtensionManager
        mgr = ExtensionManager(namespace=self.NAMESPACE)
        return [ext.name for ext in mgr]
```

**pyproject.toml Entry Points:**
```toml
[project.entry-points."dataraum.loaders"]
csv = "dataraum_context.staging.loaders.csv:CSVLoader"
parquet = "dataraum_context.staging.loaders.parquet:ParquetLoader"
postgres = "dataraum_context.staging.loaders.postgres:PostgresLoader"
excel = "dataraum_context.staging.loaders.excel:ExcelLoader"
api = "dataraum_context.staging.loaders.api:APILoader"
```

### Proposed Loader Hierarchy

```
UNTYPED Sources (VARCHAR-first):
├── CSVLoader         ✓ Exists
├── JSONLoader
├── ExcelLoader
└── APILoader         (REST/GraphQL)

WEAK Sources (advisory types):
├── SQLiteLoader
├── MySQLLoader
└── MongoDBLoader

STRONG Sources (enforced types):
├── ParquetLoader
├── PostgresLoader
└── DuckDBLoader      (for .duckdb files)
```

### Benefits

- **Third-party extensibility**: Users can add loaders via `pip install dataraum-loader-snowflake`
- **Clean separation**: Core package doesn't depend on all database drivers
- **Lazy loading**: Only import what's needed

---

## B) Domains: Ontology Framework

### Current State

Ontologies are YAML files in `config/ontologies/` with:
- **Concepts**: Business terms with indicators/patterns
- **Metrics**: Computable formulas with required concepts
- **Quality Rules**: Domain-specific validation rules
- **Semantic Hints**: Pattern → type/role mappings

Only `financial_reporting.yaml` exists as an example.

### Recommendation: Hierarchical Ontology Structure

**1. Base Ontology with Inheritance:**

```yaml
# config/ontologies/base.yaml (shared concepts)
name: base
version: "1.0.0"
concepts:
  - name: identifier
    indicators: ["*_id", "*_key", "*_code"]
    typical_role: key
  - name: timestamp
    indicators: ["*_at", "*_date", "*_time", "created*", "updated*"]
    typical_role: timestamp

# config/ontologies/retail.yaml
name: retail
extends: base  # Inherit base concepts
version: "1.0.0"
concepts:
  - name: customer
    indicators: ["customer*", "buyer*", "client*"]
    entity_type: dimension
  - name: product
    indicators: ["product*", "item*", "sku*"]
    entity_type: dimension
  - name: revenue
    indicators: ["revenue", "sales", "amount", "total"]
    typical_role: measure
    temporal_behavior: additive
```

**2. Ontology Composition Loader:**

```python
# context/ontology.py
class OntologyLoader:
    """Load and compose ontologies with inheritance."""

    def load(self, name: str) -> Ontology:
        """Load ontology, resolving inheritance chain."""
        config = self._load_yaml(f"config/ontologies/{name}.yaml")

        if "extends" in config:
            base = self.load(config["extends"])
            return self._merge(base, config)

        return Ontology.model_validate(config)
```

### Proposed Domain Ontologies

| Domain | Key Concepts | Use Case |
|--------|-------------|----------|
| `base` | identifier, timestamp, quantity, percentage | Universal concepts |
| `financial_reporting` | revenue, expense, asset, liability, account | Finance/Accounting |
| `retail` | customer, product, order, inventory, price | E-commerce |
| `marketing` | campaign, channel, impression, conversion | Marketing analytics |
| `operations` | resource, capacity, throughput, utilization | Operations |
| `healthcare` | patient, diagnosis, procedure, provider | Healthcare analytics |
| `iot` | device, sensor, reading, threshold, alert | Sensor/IoT data |

---

## C) Pipeline Flow: Step Registry

### Current State (Hardcoded)

```python
# dataflows/pipeline.py - run_pipeline()
async def run_pipeline(...):
    # Stage 1: Staging (MANDATORY)
    staging_result = await loader.load(...)

    # Stage 2-4: Profiling (MANDATORY)
    for table in tables:
        schema_result = await profile_schema(...)
        type_result = await resolve_types(...)
        stats_result = await profile_statistics(...)

    # Stage 5: Semantic (CRITICAL - fails pipeline)
    semantic_result = await enrich_semantic(...)

    # Stage 6-8: Non-critical enrichment
    topology_result = await enrich_topology(...)
    temporal_result = await enrich_temporal(...)
    cross_table_result = await compute_cross_table_multicollinearity(...)

    return PipelineResult(...)
```

### Industry Patterns for Dynamic Pipeline Composition

Based on research into Hamilton, Dagster, and Prefect:

**Two-Level Orchestration Pattern:**
- **Micro-orchestration** (Hamilton): Lightweight, in-process DAG for step composition
- **Macro-orchestration** (Dagster/Prefect): Full platform for scheduling, monitoring, retries

### Recommendation: Step Registry with Protocol

**1. Step Protocol Definition:**

```python
# dataflows/steps/protocol.py
from typing import Protocol, Literal, runtime_checkable

@runtime_checkable
class PipelineStep(Protocol):
    """Protocol for pipeline steps."""

    @property
    def name(self) -> str:
        """Step identifier."""
        ...

    @property
    def scope(self) -> Literal["source", "table", "column"]:
        """What level this step operates on."""
        ...

    @property
    def phase(self) -> Literal["staging", "profiling", "enrichment", "quality"]:
        """Pipeline phase."""
        ...

    @property
    def dependencies(self) -> list[str]:
        """Steps that must run before this one."""
        ...

    @property
    def criticality(self) -> Literal["critical", "required", "optional"]:
        """How failures are handled."""
        ...

    async def execute(
        self,
        context: StepContext,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
    ) -> Result[StepOutput]:
        """Execute the step."""
        ...
```

**2. Step Registry:**

```python
# dataflows/steps/registry.py

class StepRegistry:
    """Registry of pipeline steps with dependency resolution."""

    def __init__(self):
        self._steps: dict[str, type[PipelineStep]] = {}

    def register(self, step_cls: type[PipelineStep]) -> None:
        """Register a step class."""
        instance = step_cls()
        self._steps[instance.name] = step_cls

    def build_execution_order(
        self,
        requested: list[str] | None = None,
    ) -> list[PipelineStep]:
        """Build topologically sorted execution order."""
        if requested is None:
            requested = list(self._steps.keys())

        # Topological sort based on dependencies
        return self._topo_sort(requested)

    def get_steps_for_scope(
        self,
        scope: Literal["source", "table", "column"],
    ) -> list[PipelineStep]:
        """Get all steps for a given scope."""
        return [
            step_cls() for step_cls in self._steps.values()
            if step_cls().scope == scope
        ]

# Global registry
registry = StepRegistry()

# Decorator for registration
def pipeline_step(cls: type[PipelineStep]) -> type[PipelineStep]:
    """Decorator to register a pipeline step."""
    registry.register(cls)
    return cls
```

**3. Step Implementations:**

```python
# dataflows/steps/profiling.py

@pipeline_step
class SchemaProfilingStep:
    """Pattern detection and type candidate generation."""

    name = "schema_profiling"
    scope = "table"
    phase = "profiling"
    dependencies = ["staging"]
    criticality = "required"

    async def execute(
        self,
        context: StepContext,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
    ) -> Result[SchemaProfile]:
        return await profile_schema(
            context.table_id, duckdb_conn, session
        )


@pipeline_step
class SemanticEnrichmentStep:
    """LLM-based semantic analysis."""

    name = "semantic_enrichment"
    scope = "table"
    phase = "enrichment"
    dependencies = ["type_resolution", "statistics_profiling"]
    criticality = "critical"  # Pipeline fails if this fails

    async def execute(
        self,
        context: StepContext,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
    ) -> Result[SemanticEnrichmentResult]:
        return await enrich_semantic(
            session=session,
            llm_service=context.llm_service,
            table_ids=[context.table_id],
            ontology=context.ontology,
        )
```

**4. Configurable Pipeline Executor:**

```python
# dataflows/executor.py

class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""

    # Step selection
    steps: list[str] | Literal["all"] = "all"
    skip_steps: list[str] = []

    # Behavior
    fail_fast: bool = False  # Stop on first error
    parallel_enrichment: bool = True  # Run enrichment steps in parallel

    # Scope
    tables: list[str] | None = None  # None = all tables

    # Domain
    ontology: str = "general"


class PipelineExecutor:
    """Configurable pipeline executor."""

    def __init__(
        self,
        registry: StepRegistry,
        config: PipelineConfig,
    ):
        self.registry = registry
        self.config = config

    async def execute(
        self,
        source: str | list[str],
        source_name: str,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
        llm_service: LLMService,
    ) -> Result[PipelineResult]:
        """Execute pipeline with configured steps."""

        # Build execution plan
        if self.config.steps == "all":
            steps = self.registry.build_execution_order()
        else:
            steps = self.registry.build_execution_order(self.config.steps)

        # Filter out skipped steps
        steps = [s for s in steps if s.name not in self.config.skip_steps]

        # Group by phase for execution
        by_phase = self._group_by_phase(steps)

        # Execute phases
        results = {}
        for phase in ["staging", "profiling", "enrichment", "quality"]:
            phase_steps = by_phase.get(phase, [])

            if self.config.parallel_enrichment and phase == "enrichment":
                phase_results = await self._run_parallel(phase_steps, ...)
            else:
                phase_results = await self._run_sequential(phase_steps, ...)

            results.update(phase_results)

        return Result.ok(PipelineResult.from_step_results(results))
```

### Pipeline Configuration Examples

```yaml
# config/pipelines/default.yaml
name: default
description: Full pipeline with all steps

steps: all
skip_steps: []
fail_fast: false
parallel_enrichment: true
ontology: general
```

```yaml
# config/pipelines/quick_profile.yaml
name: quick_profile
description: Fast profiling without enrichment

steps:
  - staging
  - schema_profiling
  - type_resolution
  - statistics_profiling
skip_steps:
  - semantic_enrichment
  - topology_enrichment
  - temporal_enrichment
  - cross_table_multicollinearity
fail_fast: true
parallel_enrichment: false
ontology: null
```

```yaml
# config/pipelines/financial.yaml
name: financial
description: Full pipeline with financial ontology

steps: all
skip_steps: []
fail_fast: false
parallel_enrichment: true
ontology: financial_reporting
```

---

## D) Workflow Engine Abstraction

### The Challenge

The framework needs to support two distinct deployment modes:

1. **Local/OSS Mode**: Simple, in-process execution for development and open-source users
2. **Production Mode**: Durable execution on cloud platforms (e.g., [Cloudflare Python Workflows](https://blog.cloudflare.com/python-workflows/), [Temporal](https://temporal.io/), [Prefect](https://www.prefect.io/))

The core package should NOT depend on any specific workflow engine. Production engines should be pluggable without modifying the open-source codebase.

### Industry Context

| Engine | Type | Best For | Python Support |
|--------|------|----------|----------------|
| **In-process** | Micro-orchestrator | Local dev, testing | Native |
| **Hamilton** | Micro-orchestrator | Data pipelines, lineage | Native |
| **Cloudflare Workflows** | Durable execution | Serverless, edge | Beta (Nov 2025) |
| **Temporal** | Durable execution | Enterprise, multi-language | SDK |
| **Prefect** | Workflow orchestration | Data teams, Python-native | Native |

### Recommendation: Workflow Engine Protocol

**1. Define Abstract Workflow Interface:**

```python
# core/workflow/protocol.py
from typing import Protocol, TypeVar, Generic
from abc import abstractmethod

T = TypeVar("T")
R = TypeVar("R")

class WorkflowEngine(Protocol):
    """Abstract workflow engine interface."""

    @abstractmethod
    async def execute_pipeline(
        self,
        pipeline_config: PipelineConfig,
        context: ExecutionContext,
    ) -> Result[PipelineResult]:
        """Execute a complete pipeline."""
        ...

    @abstractmethod
    async def execute_step(
        self,
        step: PipelineStep,
        context: StepContext,
    ) -> Result[StepOutput]:
        """Execute a single step with retry/persistence."""
        ...

    @abstractmethod
    async def checkpoint(
        self,
        checkpoint_id: str,
        state: dict,
    ) -> None:
        """Persist checkpoint for resume capability."""
        ...

    @abstractmethod
    async def resume(
        self,
        checkpoint_id: str,
    ) -> ExecutionContext:
        """Resume from a checkpoint."""
        ...


class StepExecutor(Protocol[T, R]):
    """Protocol for step execution with retry semantics."""

    @abstractmethod
    async def run(
        self,
        input: T,
        retry_count: int = 0,
    ) -> R:
        """Execute step, may be retried by engine."""
        ...
```

**2. In-Process Engine (OSS Default):**

```python
# core/workflow/engines/local.py

class LocalWorkflowEngine:
    """Simple in-process workflow engine for local development.

    No external dependencies. Included in core package.
    Checkpoints stored in SQLite metadata database.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def execute_pipeline(
        self,
        pipeline_config: PipelineConfig,
        context: ExecutionContext,
    ) -> Result[PipelineResult]:
        """Execute pipeline steps sequentially in-process."""
        results = {}

        for step in self._build_execution_order(pipeline_config):
            try:
                step_result = await self.execute_step(step, context)
                if not step_result.success and step.criticality == "critical":
                    return Result.fail(f"Critical step {step.name} failed")
                results[step.name] = step_result
            except Exception as e:
                await self.checkpoint(context.run_id, results)
                raise

        return Result.ok(PipelineResult.from_step_results(results))

    async def checkpoint(self, checkpoint_id: str, state: dict) -> None:
        """Store checkpoint in metadata database."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            state=json.dumps(state),
            created_at=datetime.now(UTC),
        )
        self.session.add(checkpoint)
        await self.session.commit()
```

**3. Cloudflare Workflows Engine (Separate Package):**

```python
# dataraum-workflow-cloudflare/src/dataraum_workflow_cloudflare/engine.py

from cloudflare.workflows import Workflow, step

class CloudflareWorkflowEngine:
    """Cloudflare Workflows integration for durable execution.

    Install: pip install dataraum-workflow-cloudflare

    Features:
    - Automatic retries with exponential backoff
    - Persistent state across failures
    - Runs on Cloudflare edge network
    """

    async def execute_pipeline(
        self,
        pipeline_config: PipelineConfig,
        context: ExecutionContext,
    ) -> Result[PipelineResult]:
        """Dispatch to Cloudflare Workflow."""
        # Cloudflare handles durability, retries, state
        workflow = DataraumPipelineWorkflow()
        return await workflow.run(pipeline_config, context)


class DataraumPipelineWorkflow(Workflow):
    """Cloudflare Workflow definition for pipeline execution."""

    @step
    async def staging(self, context: ExecutionContext) -> StagingResult:
        """Durable staging step."""
        # Cloudflare persists result, retries on failure
        return await execute_staging(context)

    @step
    async def profiling(self, staging_result: StagingResult) -> ProfilingResult:
        """Durable profiling step."""
        return await execute_profiling(staging_result)

    # ... additional steps
```

**4. Engine Registration via Entry Points:**

```toml
# Core package - pyproject.toml
[project.entry-points."dataraum.workflow_engines"]
local = "dataraum_context.core.workflow.engines.local:LocalWorkflowEngine"

# Cloudflare package - pyproject.toml
[project.entry-points."dataraum.workflow_engines"]
cloudflare = "dataraum_workflow_cloudflare:CloudflareWorkflowEngine"

# Temporal package - pyproject.toml
[project.entry-points."dataraum.workflow_engines"]
temporal = "dataraum_workflow_temporal:TemporalWorkflowEngine"
```

**5. Engine Selection:**

```python
# core/workflow/factory.py

def get_workflow_engine(engine_name: str = "local") -> WorkflowEngine:
    """Get workflow engine by name.

    Default: 'local' (in-process, no dependencies)
    Production: 'cloudflare', 'temporal', 'prefect' (separate packages)
    """
    from stevedore import driver

    mgr = driver.DriverManager(
        namespace="dataraum.workflow_engines",
        name=engine_name,
        invoke_on_load=True,
    )
    return mgr.driver
```

### Configuration

```yaml
# config/workflow.yaml
engine: local  # or 'cloudflare', 'temporal', 'prefect'

# Engine-specific settings
local:
  checkpoint_table: workflow_checkpoints
  max_retries: 3

cloudflare:
  account_id: ${CLOUDFLARE_ACCOUNT_ID}
  api_token: ${CLOUDFLARE_API_TOKEN}

temporal:
  host: localhost:7233
  namespace: dataraum
  task_queue: pipeline-tasks
```

---

## E) Step Packages: Installable Components

### The Vision

Pipeline steps should be installable as separate packages, enabling:

- **Modular installation**: Only install what you need
- **Version independence**: Update steps independently
- **Community contributions**: Third-party enrichment steps
- **Tiered offerings**: Basic (OSS) vs Advanced (commercial)

### Namespace Package Architecture

Use Python namespace packages under `dataraum_steps.*`:

```
dataraum-context/              # Core package
├── src/dataraum_context/
│   └── ...

dataraum-steps-quality-financial/   # Financial quality rules
├── src/dataraum_steps/
│   └── quality/
│       └── financial/
│           ├── __init__.py
│           └── rules.py

dataraum-steps-statistics-advanced/  # Advanced statistics
├── src/dataraum_steps/
│   └── statistics/
│       └── advanced/
│           ├── __init__.py
│           └── bayesian.py
```

### Step Package Examples

**1. Quality Domain Packages:**

```python
# dataraum-steps-quality-financial/src/dataraum_steps/quality/financial/__init__.py

from dataraum_context.dataflows.steps.protocol import pipeline_step, PipelineStep

@pipeline_step
class FinancialQualityStep:
    """Financial domain quality rules.

    Install: pip install dataraum-steps-quality-financial

    Adds rules for:
    - Revenue/expense validation
    - Account balance checks
    - Fiscal period completeness
    - GAAP compliance hints
    """

    name = "quality_financial"
    scope = "table"
    phase = "quality"
    dependencies = ["semantic_enrichment"]
    criticality = "optional"

    async def execute(self, context: StepContext, ...) -> Result[QualityResult]:
        # Financial-specific quality checks
        rules = [
            RevenueNonNegativeRule(),
            BalanceSheetBalancesRule(),
            FiscalPeriodCompletenessRule(),
        ]
        return await apply_quality_rules(context, rules)
```

**2. Statistics Packages (Basic vs Advanced):**

```python
# Core package: Basic statistics (always included)
# dataraum_context/profiling/statistics.py

class BasicStatisticsStep:
    """Basic statistics included in core.

    - Mean, median, stddev
    - Min, max, percentiles
    - Null counts, cardinality
    """
    name = "statistics_basic"
    # ...


# Separate package: Advanced statistics
# dataraum-steps-statistics-advanced/src/dataraum_steps/statistics/advanced/__init__.py

@pipeline_step
class AdvancedStatisticsStep:
    """Advanced statistical analysis.

    Install: pip install dataraum-steps-statistics-advanced

    Adds:
    - Bayesian inference
    - Distribution fitting
    - Anomaly detection (Isolation Forest, LOF)
    - Time series decomposition
    - Multivariate analysis
    """

    name = "statistics_advanced"
    scope = "table"
    phase = "profiling"
    dependencies = ["statistics_basic"]  # Extends basic
    criticality = "optional"

    async def execute(self, context: StepContext, ...) -> Result[AdvancedStatsResult]:
        # Requires scipy, scikit-learn (heavy dependencies)
        from scipy import stats
        from sklearn.ensemble import IsolationForest
        # ...
```

**3. LLM Provider Packages:**

```python
# dataraum-llm-anthropic/src/dataraum_llm/anthropic/__init__.py

from dataraum_context.llm.protocol import LLMProvider

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration.

    Install: pip install dataraum-llm-anthropic
    """

    name = "anthropic"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        # ...


# Entry point registration
[project.entry-points."dataraum.llm_providers"]
anthropic = "dataraum_llm.anthropic:AnthropicProvider"
```

### Package Discovery

```python
# core/plugins.py

from stevedore import ExtensionManager

def discover_step_packages() -> dict[str, list[PipelineStep]]:
    """Discover all installed step packages."""
    mgr = ExtensionManager(
        namespace="dataraum.pipeline_steps",
        invoke_on_load=False,
        propagate_map_exceptions=True,
    )

    steps_by_phase = defaultdict(list)
    for ext in mgr:
        step = ext.plugin()
        steps_by_phase[step.phase].append(step)

    return steps_by_phase


def list_available_packages() -> dict[str, str]:
    """List all available step packages."""
    return {
        "quality": [
            "dataraum-steps-quality-financial",
            "dataraum-steps-quality-healthcare",
            "dataraum-steps-quality-retail",
        ],
        "statistics": [
            "dataraum-steps-statistics-advanced",
            "dataraum-steps-statistics-timeseries",
        ],
        "enrichment": [
            "dataraum-steps-enrichment-nlp",
            "dataraum-steps-enrichment-geo",
        ],
    }
```

---

## F) Additional Abstraction Layers

Beyond the core dimensions, consider these additional abstraction opportunities:

### F.1) Storage Backend Abstraction

**Data Compute Engine:**

```python
# Currently: DuckDB only
# Could support: Polars, Spark, BigQuery, Snowflake

class ComputeEngine(Protocol):
    """Abstract compute engine interface."""

    async def execute_query(self, sql: str) -> pa.Table:
        ...

    async def create_table(self, name: str, schema: pa.Schema) -> None:
        ...

    async def load_file(self, path: str, table_name: str) -> int:
        ...
```

**Metadata Storage:**

```python
# Currently: SQLAlchemy (SQLite/PostgreSQL)
# Could support: Cloud databases, object storage

class MetadataStore(Protocol):
    """Abstract metadata storage interface."""

    async def save_profile(self, profile: ColumnProfile) -> None:
        ...

    async def get_profiles(self, table_id: str) -> list[ColumnProfile]:
        ...
```

### F.2) Output/Delivery Abstraction

```python
# Currently: MCP Server, FastAPI
# Could support: File export, webhooks, streaming

class ContextDelivery(Protocol):
    """Abstract context delivery interface."""

    async def deliver(
        self,
        context: ContextDocument,
        format: Literal["json", "markdown", "yaml"],
    ) -> None:
        ...

# Implementations:
# - MCPDelivery: MCP server tools
# - RESTDelivery: FastAPI endpoints
# - FileDelivery: Export to files
# - WebhookDelivery: POST to URL
# - StreamDelivery: SSE/WebSocket
```

### F.3) Caching Backend Abstraction

```python
# Currently: LLM cache in SQLite
# Could support: Redis, Memcached, cloud caching

class CacheBackend(Protocol):
    """Abstract caching interface."""

    async def get(self, key: str) -> Any | None:
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ...

    async def invalidate(self, pattern: str) -> int:
        ...
```

### F.4) Observability Abstraction

```python
# Could integrate: OpenTelemetry, custom metrics

class ObservabilityProvider(Protocol):
    """Abstract observability interface."""

    def trace_step(self, step_name: str) -> ContextManager:
        ...

    def record_metric(self, name: str, value: float, tags: dict) -> None:
        ...

    def log_event(self, event: str, context: dict) -> None:
        ...
```

### F.5) Authentication Abstraction

```python
# Could support: API keys, OAuth, OIDC, custom

class AuthProvider(Protocol):
    """Abstract authentication interface."""

    async def authenticate(self, credentials: Any) -> User | None:
        ...

    async def authorize(self, user: User, action: str, resource: str) -> bool:
        ...
```

### Abstraction Priority Matrix

| Abstraction | Current State | Priority | Reason |
|-------------|--------------|----------|--------|
| Workflow Engine | Hardcoded | **P0** | Core flexibility need |
| Step Packages | Monolithic | **P0** | Modularity requirement |
| Source Loaders | ABC exists | **P1** | Pattern exists, expand |
| LLM Providers | Partial | **P1** | Already somewhat pluggable |
| Compute Engine | DuckDB only | **P2** | DuckDB covers most cases |
| Output Delivery | MCP + REST | **P2** | Current approach works |
| Caching | SQLite | **P3** | Works for most cases |
| Observability | None | **P3** | Nice to have |
| Auth | None | **P3** | API-level concern |

---

## Dependency Injection Recommendation

Based on research, **[Lagom](https://github.com/meadsteve/lagom)** is the best fit for this codebase:

| Framework | Pros | Cons | Fit |
|-----------|------|------|-----|
| **Lagom** | Type-based autowiring, minimal boilerplate, async support | Smaller community | ⭐⭐⭐⭐⭐ Best fit |
| **python-dependency-injector** | Feature-rich, well-documented | C extension, heavier | ⭐⭐⭐ Overkill |
| **Punq** | Simple, lightweight | Less active | ⭐⭐⭐⭐ Good alternative |

**Why Lagom:**
- Type-based auto-wiring with zero configuration
- Strong mypy integration
- Minimal changes to existing code
- Support for async Python
- Works well with FastAPI

**Example Integration:**

```python
# core/container.py
from lagom import Container

def create_container() -> Container:
    """Create DI container with all dependencies."""
    container = Container()

    # Register services
    container[Settings] = Settings()
    container[LoaderRegistry] = LoaderRegistry()
    container[StepRegistry] = StepRegistry()
    container[LLMService] = lambda c: create_llm_service(c[Settings])

    return container

# Usage in API
@router.post("/pipeline")
async def run_pipeline(
    request: PipelineRequest,
    container: Container = Depends(get_container),
):
    executor = container[PipelineExecutor]
    return await executor.execute(request)
```

---

## Package Architecture

### Monorepo vs Multi-Repo

For a highly modular framework, consider a **monorepo with multiple publishable packages**:

```
dataraum/
├── packages/
│   ├── core/                    # dataraum-context (main package)
│   │   ├── pyproject.toml
│   │   └── src/dataraum_context/
│   │
│   ├── workflow-local/          # dataraum-workflow-local (included in core)
│   │   └── ...
│   │
│   ├── workflow-cloudflare/     # dataraum-workflow-cloudflare
│   │   ├── pyproject.toml
│   │   └── src/dataraum_workflow_cloudflare/
│   │
│   ├── workflow-temporal/       # dataraum-workflow-temporal
│   │   └── ...
│   │
│   ├── llm-anthropic/           # dataraum-llm-anthropic
│   │   └── ...
│   │
│   ├── llm-openai/              # dataraum-llm-openai
│   │   └── ...
│   │
│   ├── loader-parquet/          # dataraum-loader-parquet
│   │   └── ...
│   │
│   ├── steps-quality-financial/ # dataraum-steps-quality-financial
│   │   └── ...
│   │
│   └── steps-statistics-advanced/
│       └── ...
│
├── pyproject.toml               # Workspace configuration
└── uv.lock                      # Unified lockfile
```

### Core Package Extras

The core package should define extras for common bundles:

```toml
# packages/core/pyproject.toml

[project]
name = "dataraum-context"
dependencies = [
    "duckdb>=1.0.0",
    "pyarrow>=15.0.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "stevedore>=5.0.0",  # Plugin discovery
]

[project.optional-dependencies]
# LLM providers
anthropic = ["dataraum-llm-anthropic"]
openai = ["dataraum-llm-openai"]
llm = ["dataraum-llm-anthropic", "dataraum-llm-openai"]

# Workflow engines
cloudflare = ["dataraum-workflow-cloudflare"]
temporal = ["dataraum-workflow-temporal"]
prefect = ["dataraum-workflow-prefect"]

# Loaders
parquet = ["dataraum-loader-parquet"]
postgres = ["dataraum-loader-postgres"]
all-loaders = ["dataraum-loader-parquet", "dataraum-loader-postgres", "dataraum-loader-excel"]

# Domain packages
financial = ["dataraum-steps-quality-financial"]
healthcare = ["dataraum-steps-quality-healthcare"]
retail = ["dataraum-steps-quality-retail"]

# Statistics
stats-advanced = ["dataraum-steps-statistics-advanced"]

# Everything
all = [
    "dataraum-context[llm]",
    "dataraum-context[all-loaders]",
    "dataraum-context[stats-advanced]",
]
```

### Installation Patterns

```bash
# Minimal (CSV only, no LLM, local workflow)
pip install dataraum-context

# With Anthropic LLM
pip install dataraum-context[anthropic]

# Production with Cloudflare
pip install dataraum-context[anthropic,cloudflare]

# Financial domain with advanced stats
pip install dataraum-context[anthropic,financial,stats-advanced]

# Everything
pip install dataraum-context[all]
```

### Entry Point Namespaces

All plugins discovered via standardized namespaces:

| Namespace | Purpose | Example |
|-----------|---------|---------|
| `dataraum.loaders` | Source loaders | `csv`, `parquet`, `postgres` |
| `dataraum.workflow_engines` | Workflow executors | `local`, `cloudflare`, `temporal` |
| `dataraum.llm_providers` | LLM integrations | `anthropic`, `openai`, `local` |
| `dataraum.pipeline_steps` | Pipeline steps | `quality_financial`, `stats_advanced` |
| `dataraum.ontologies` | Bundled ontologies | `financial_reporting`, `retail` |
| `dataraum.output_formats` | Output formatters | `json`, `markdown`, `mcp` |

### Version Compatibility

Define compatibility constraints between packages:

```toml
# dataraum-steps-quality-financial/pyproject.toml

[project]
name = "dataraum-steps-quality-financial"
version = "1.2.0"
dependencies = [
    "dataraum-context>=0.5.0,<1.0.0",  # Compatible core versions
]
```

---

## Documentation Gaps to Address

### 1. Extension Guide (NEW)

Create `docs/EXTENDING.md`:

```markdown
# Extending DataRaum Context Engine

## Adding a New Loader
1. Inherit from `LoaderBase`
2. Implement required methods
3. Register via entry point

## Adding a New Ontology
1. Create YAML in `config/ontologies/`
2. Define concepts, metrics, rules
3. Reference in pipeline config

## Adding a New Pipeline Step
1. Implement `PipelineStep` protocol
2. Decorate with `@pipeline_step`
3. Dependencies automatically resolved
```

### 2. Configuration Reference (NEW)

Create `docs/CONFIGURATION.md`:

```markdown
# Configuration Reference

## Environment Variables
- `DATARAUM_DATABASE_URL`
- `DATARAUM_DUCKDB_PATH`
- ...

## YAML Configuration Files
### config/llm.yaml
### config/ontologies/*.yaml
### config/pipelines/*.yaml
### config/patterns/*.yaml
```

### 3. Update ARCHITECTURE.md

Add sections for:
- Plugin architecture (loaders, steps)
- Ontology inheritance
- Pipeline configuration

---

## Implementation Priorities

### Priority Matrix

| Priority | Area | Change | Effort | Impact |
|----------|------|--------|--------|--------|
| **P0** | Workflow | Define `WorkflowEngine` protocol | Low | Critical |
| **P0** | Workflow | Implement `LocalWorkflowEngine` (OSS default) | Medium | Critical |
| **P0** | Pipeline | Add `PipelineConfig` for step selection | Low | High |
| **P0** | Pipeline | Define `PipelineStep` protocol | Low | High |
| **P1** | Pipeline | Implement `StepRegistry` with discovery | Medium | High |
| **P1** | Sources | Add entry point plugin system | Medium | High |
| **P1** | Packages | Restructure as monorepo with workspaces | High | High |
| **P2** | Domains | Add ontology inheritance (`extends:`) | Low | Medium |
| **P2** | Domains | Create 3-5 core ontologies | Medium | Medium |
| **P2** | Steps | Extract quality domains to separate packages | Medium | Medium |
| **P3** | DI | Integrate Lagom for service management | Medium | Medium |
| **P3** | Workflow | Create `dataraum-workflow-cloudflare` package | Medium | Medium |
| **P3** | Docs | Create EXTENDING.md and CONFIGURATION.md | Low | Medium |

### Phase 1: Core Abstractions (Foundation)

1. **Workflow Engine Protocol** - Abstract interface for execution engines
2. **Local Engine** - In-process executor with SQLite checkpoints (no deps)
3. **Pipeline Step Protocol** - Standard interface for all pipeline steps
4. **Step Registry** - Discovery and dependency resolution

### Phase 2: Plugin Infrastructure

1. **Entry Point System** - Stevedore-based plugin discovery
2. **Namespace Packages** - `dataraum_steps.*` for installable components
3. **Loader Plugins** - Parquet, PostgreSQL as separate packages
4. **LLM Provider Plugins** - Anthropic, OpenAI as separate packages

### Phase 3: Package Ecosystem

1. **Monorepo Structure** - UV workspaces for multi-package development
2. **Quality Domain Packages** - Financial, Healthcare, Retail as extras
3. **Advanced Statistics Package** - Heavy deps (scipy, sklearn) optional
4. **Production Workflow Engines** - Cloudflare, Temporal packages

### Phase 4: Polish & Ecosystem

1. **Ontology Inheritance** - Base ontology with domain extensions
2. **Package Marketplace** - Documentation for community contributions
3. **Version Compatibility** - SemVer constraints between packages

---

## References

### Dependency Injection
- [Lagom - Type-based autowiring DI](https://github.com/meadsteve/lagom)
- [python-dependency-injector](https://github.com/ets-labs/python-dependency-injector)
- [Punq - Simple IoC container](https://pypi.org/project/punq/)
- [Cosmic Python - DI & Bootstrapping](https://www.cosmicpython.com/book/chapter_13_dependency_injection.html)

### Workflow Orchestration
- [Cloudflare Python Workflows (Beta)](https://blog.cloudflare.com/python-workflows/)
- [Cloudflare Workflows GA](https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/)
- [Temporal - Workflow Engine Principles](https://temporal.io/blog/workflow-engine-principles)
- [Prefect - Workflow Orchestration](https://www.prefect.io/)
- [Declarative data orchestration: Dagster & Hamilton](https://blog.dagworks.io/p/declarative-data-orchestration-dagster)
- [Dagster vs Prefect comparison](https://dagster.io/vs/dagster-vs-prefect)
- [Orchestration Showdown - ZenML Blog](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow)

### Plugin Architecture
- [Stevedore - Dynamic plugins for Python](https://github.com/openstack/stevedore)
- [Creating Plugins with Stevedore](https://docs.openstack.org/stevedore/latest/user/tutorial/creating_plugins.html)
- [Python Entry Points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [Creating and Discovering Plugins](https://packaging.python.org/guides/creating-and-discovering-plugins/)

### Python Packaging
- [Namespace Packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)
- [Optional Dependencies (Extras)](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
- [PEP 508 - Dependency Specification](https://peps.python.org/pep-0508/)

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-05 | 0.2.0 | Added workflow engine abstraction, step packages, package architecture |
| 2025-12-05 | 0.1.0 | Initial reflection document |
