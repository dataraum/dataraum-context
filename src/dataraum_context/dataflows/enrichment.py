"""Hamilton dataflow for enrichment pipeline.

This module defines the enrichment pipeline as a Hamilton DAG:
    semantic → topology → cross_table_multicollinearity
               ↓              ↑
            temporal ─────────┘

Each node is a pure async function that declares its dependencies via parameters.
Hamilton automatically constructs the DAG and executes in topological order.

Usage:
    from hamilton import driver
    from hamilton.experimental import h_async
    from dataraum_context.dataflows import enrichment

    # Create driver
    dr = driver.Builder().with_modules(enrichment).with_adapter(h_async.AsyncDriver()).build()

    # Execute
    result = await dr.execute(
        ["semantic_enrichment", "topology_enrichment", "temporal_enrichment"],
        inputs={"table_ids": ["t1", "t2"], "llm_service": llm, ...}
    )
"""

from typing import Any

import duckdb
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.enrichment.cross_table_multicollinearity import (
    compute_cross_table_multicollinearity,
)
from dataraum_context.enrichment.semantic import enrich_semantic
from dataraum_context.enrichment.temporal import enrich_temporal
from dataraum_context.enrichment.topology import enrich_topology
from dataraum_context.llm import LLMService
from dataraum_context.profiling.models import CrossTableMulticollinearityAnalysis

# ============================================================================
# DAG Nodes (Hamilton Functions)
# ============================================================================


async def semantic_enrichment(
    table_ids: list[str],
    llm_service: LLMService,
    session: AsyncSession,
    ontology: str = "general",
) -> Result[Any]:
    """Step 1: Semantic enrichment via LLM.

    Analyzes column roles, entities, and semantic relationships.

    Args:
        table_ids: Tables to enrich
        llm_service: LLM service for semantic analysis
        session: Database session
        ontology: Ontology to use (default: "general")

    Returns:
        Result containing SemanticEnrichmentResult

    Dependencies: None (entry point)
    """
    return await enrich_semantic(
        session=session,
        llm_service=llm_service,
        table_ids=table_ids,
        ontology=ontology,
    )


async def topology_enrichment(
    semantic_enrichment: Result[Any],  # Depends on semantic
    table_ids: list[str],
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
) -> Result[Any]:
    """Step 2: Topology enrichment via TDA.

    Detects FK relationships and structural topology.

    Args:
        semantic_enrichment: Result from semantic enrichment (dependency)
        table_ids: Tables to enrich
        duckdb_conn: DuckDB connection
        session: Database session

    Returns:
        Result containing TopologyEnrichmentResult

    Dependencies: semantic_enrichment (for semantic context)
    """
    # Check if semantic succeeded before proceeding
    if not semantic_enrichment.success:
        return Result.fail(f"Topology skipped: semantic failed ({semantic_enrichment.error})")

    return await enrich_topology(
        session=session,
        duckdb_conn=duckdb_conn,
        table_ids=table_ids,
    )


async def temporal_enrichment(
    semantic_enrichment: Result[Any],  # Depends on semantic
    table_ids: list[str],
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
) -> Result[Any]:
    """Step 3: Temporal enrichment for time columns.

    Analyzes seasonality, trends, change points, and completeness.

    Args:
        semantic_enrichment: Result from semantic enrichment (dependency)
        table_ids: Tables to enrich
        duckdb_conn: DuckDB connection
        session: Database session

    Returns:
        Result containing TemporalEnrichmentResult

    Dependencies: semantic_enrichment (for time column identification)
    """
    # Check if semantic succeeded before proceeding
    if not semantic_enrichment.success:
        return Result.fail(f"Temporal skipped: semantic failed ({semantic_enrichment.error})")

    return await enrich_temporal(
        session=session,
        duckdb_conn=duckdb_conn,
        table_ids=table_ids,
    )


async def cross_table_multicollinearity(
    topology_enrichment: Result[Any],  # Depends on topology
    temporal_enrichment: Result[Any],  # Depends on temporal
    table_ids: list[str],
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    enable_cross_table_multicollinearity: bool = True,
) -> Result[CrossTableMulticollinearityAnalysis | None]:
    """Step 4: Cross-table multicollinearity analysis (conditional).

    Builds unified correlation matrix and detects dependencies across tables.

    Args:
        topology_enrichment: Result from topology enrichment (dependency)
        temporal_enrichment: Result from temporal enrichment (dependency)
        table_ids: Tables to analyze
        duckdb_conn: DuckDB connection
        session: Database session
        enable_cross_table_multicollinearity: Whether to run this step

    Returns:
        Result containing CrossTableMulticollinearityAnalysis or None if disabled

    Dependencies: topology_enrichment, temporal_enrichment (for relationships)
    """
    # Skip if disabled
    if not enable_cross_table_multicollinearity:
        return Result.ok(None, warnings=["Cross-table multicollinearity disabled"])

    # Skip if less than 2 tables
    if len(table_ids) < 2:
        return Result.ok(None, warnings=["Cross-table analysis requires ≥2 tables"])

    # Check dependencies
    if not topology_enrichment.success:
        return Result.fail(
            f"Cross-table multicollinearity skipped: topology failed ({topology_enrichment.error})"
        )

    if not temporal_enrichment.success:
        return Result.fail(
            f"Cross-table multicollinearity skipped: temporal failed ({temporal_enrichment.error})"
        )

    # Run analysis
    result = await compute_cross_table_multicollinearity(
        table_ids=table_ids,
        duckdb_conn=duckdb_conn,
        session=session,
    )
    return result  # type: ignore[return-value]


# ============================================================================
# Utility: Direct Execution (without Hamilton)
# ============================================================================


async def run_enrichment_pipeline_direct(
    table_ids: list[str],
    llm_service: LLMService,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    ontology: str = "general",
    include_cross_table: bool = True,
) -> dict[str, Any]:
    """Run enrichment pipeline directly (without Hamilton).

    This function executes the same pipeline as the Hamilton DAG but
    without workflow orchestration. Useful for:
    - Testing comparison with Hamilton
    - Quick prototyping
    - Environments where Hamilton isn't available

    Args:
        table_ids: Tables to enrich
        llm_service: LLM service
        duckdb_conn: DuckDB connection
        session: Database session
        ontology: Ontology to use
        include_cross_table: Whether to run cross-table analysis

    Returns:
        Dictionary with results from each step
    """
    results = {}

    # Step 1: Semantic
    semantic_result = await semantic_enrichment(
        table_ids=table_ids,
        llm_service=llm_service,
        session=session,
        ontology=ontology,
    )
    results["semantic_enrichment"] = semantic_result

    # Step 2: Topology (depends on semantic)
    topology_result = await topology_enrichment(
        semantic_enrichment=semantic_result,
        table_ids=table_ids,
        duckdb_conn=duckdb_conn,
        session=session,
    )
    results["topology_enrichment"] = topology_result

    # Step 3: Temporal (depends on semantic)
    temporal_result = await temporal_enrichment(
        semantic_enrichment=semantic_result,
        table_ids=table_ids,
        duckdb_conn=duckdb_conn,
        session=session,
    )
    results["temporal_enrichment"] = temporal_result

    # Step 4: Cross-table multicollinearity (depends on topology + temporal)
    if include_cross_table:
        cross_table_result = await cross_table_multicollinearity(
            topology_enrichment=topology_result,
            temporal_enrichment=temporal_result,
            table_ids=table_ids,
            duckdb_conn=duckdb_conn,
            session=session,
            enable_cross_table_multicollinearity=True,
        )
        results["cross_table_multicollinearity"] = cross_table_result

    return results
