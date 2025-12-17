"""Correlation analysis processor.

Main orchestrator that runs all correlation analyses:

Within-table analysis (analyze_correlations):
- Numeric correlations (Pearson, Spearman)
- Categorical associations (Cramér's V)
- Functional dependencies (A → B)
- Derived columns

Cross-table quality analysis (analyze_cross_table_quality):
- Cross-table correlations between joined data
- Redundant/derived column detection
- Multicollinearity (VDP) analysis
- Requires confirmed relationships from semantic agent
"""

import time
from datetime import UTC, datetime

import duckdb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.analysis.correlation.categorical import compute_categorical_associations
from dataraum_context.analysis.correlation.cross_table import analyze_relationship_quality
from dataraum_context.analysis.correlation.derived_columns import detect_derived_columns
from dataraum_context.analysis.correlation.functional_dependency import (
    detect_functional_dependencies,
)
from dataraum_context.analysis.correlation.models import (
    CorrelationAnalysisResult,
    CrossTableQualityResult,
    EnrichedRelationship,
)
from dataraum_context.analysis.correlation.numeric import compute_numeric_correlations
from dataraum_context.analysis.relationships.db_models import Relationship
from dataraum_context.core.models.base import Cardinality, RelationshipType, Result
from dataraum_context.storage import Column, Table


async def analyze_correlations(
    table_id: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
) -> Result[CorrelationAnalysisResult]:
    """Run complete correlation analysis on a table.

    This orchestrates all correlation analyses:
    - Numeric correlations
    - Categorical associations
    - Functional dependencies
    - Derived columns

    Args:
        table_id: Table ID to analyze
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session

    Returns:
        Result containing CorrelationAnalysisResult
    """
    start_time = time.time()

    try:
        # Get table
        table = await session.get(Table, str(table_id))
        if not table:
            return Result.fail(f"Table not found: {table_id}")

        # Run all analyses
        numeric_corr_result = await compute_numeric_correlations(table, duckdb_conn, session)
        numeric_correlations = numeric_corr_result.unwrap() if numeric_corr_result.success else []

        categorical_assoc_result = await compute_categorical_associations(
            table, duckdb_conn, session
        )
        categorical_associations = (
            categorical_assoc_result.unwrap() if categorical_assoc_result.success else []
        )

        fd_result = await detect_functional_dependencies(table, duckdb_conn, session)
        functional_dependencies = fd_result.unwrap() if fd_result.success else []

        derived_result = await detect_derived_columns(table, duckdb_conn, session)
        derived_columns = derived_result.unwrap() if derived_result.success else []

        # Summary stats
        stmt = select(Column).where(Column.table_id == table_id)
        result = await session.execute(stmt)
        total_columns = len(result.scalars().all())
        total_pairs = (total_columns * (total_columns - 1)) // 2

        significant_correlations = sum(1 for c in numeric_correlations if c.is_significant)
        strong_correlations = sum(
            1
            for c in numeric_correlations
            if max(abs(c.pearson_r or 0), abs(c.spearman_rho or 0)) > 0.7
        )

        duration = time.time() - start_time
        computed_at = datetime.now(UTC)

        analysis_result = CorrelationAnalysisResult(
            table_id=table_id,
            table_name=table.table_name,
            numeric_correlations=numeric_correlations,
            categorical_associations=categorical_associations,
            functional_dependencies=functional_dependencies,
            derived_columns=derived_columns,
            total_column_pairs=total_pairs,
            significant_correlations=significant_correlations,
            strong_correlations=strong_correlations,
            duration_seconds=duration,
            computed_at=computed_at,
        )

        return Result.ok(analysis_result)

    except Exception as e:
        return Result.fail(f"Correlation analysis failed: {e}")


async def analyze_cross_table_quality(
    relationship: Relationship,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    min_correlation: float = 0.3,
    redundancy_threshold: float = 0.99,
) -> Result[CrossTableQualityResult]:
    """Analyze quality of a confirmed cross-table relationship.

    This runs after relationships are confirmed by semantic analysis.
    It joins the tables and analyzes:
    - Cross-table correlations (unexpected relationships)
    - Redundant columns (r ≈ 1.0 within same table)
    - Derived columns (one column computed from another)
    - Multicollinearity (VDP-based dependency groups)

    Args:
        relationship: Confirmed Relationship DB model
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session
        min_correlation: Minimum |r| to report (default: 0.3)
        redundancy_threshold: Threshold for redundancy detection (default: 0.99)

    Returns:
        Result containing CrossTableQualityResult
    """
    try:
        # Load table metadata to get DuckDB paths
        from_table = await session.get(Table, relationship.from_table_id)
        to_table = await session.get(Table, relationship.to_table_id)
        from_column = await session.get(Column, relationship.from_column_id)
        to_column = await session.get(Column, relationship.to_column_id)

        if not all([from_table, to_table, from_column, to_column]):
            return Result.fail("Could not load relationship metadata")

        # Type narrowing - all objects are non-None after the check above
        assert from_table is not None
        assert to_table is not None
        assert from_column is not None
        assert to_column is not None

        # Verify DuckDB paths exist
        if not from_table.duckdb_path or not to_table.duckdb_path:
            return Result.fail("Tables missing DuckDB paths")

        # Convert cardinality string to enum (if present)
        cardinality = None
        if relationship.cardinality:
            try:
                cardinality = Cardinality(relationship.cardinality)
            except ValueError:
                pass  # Keep None if value doesn't match enum

        # Build EnrichedRelationship for the cross_table analyzer
        enriched = EnrichedRelationship(
            relationship_id=relationship.relationship_id,
            from_table=from_table.table_name,
            from_column=from_column.column_name,
            from_column_id=relationship.from_column_id,
            from_table_id=relationship.from_table_id,
            to_table=to_table.table_name,
            to_column=to_column.column_name,
            to_column_id=relationship.to_column_id,
            to_table_id=relationship.to_table_id,
            relationship_type=RelationshipType(relationship.relationship_type),
            cardinality=cardinality,
            confidence=relationship.confidence,
            detection_method=relationship.detection_method or "unknown",
            evidence=relationship.evidence or {},
        )

        # Run cross-table quality analysis
        result = analyze_relationship_quality(
            relationship=enriched,
            duckdb_conn=duckdb_conn,
            from_table_path=from_table.duckdb_path,
            to_table_path=to_table.duckdb_path,
            min_correlation=min_correlation,
            redundancy_threshold=redundancy_threshold,
        )

        if result is None:
            return Result.fail("Cross-table analysis returned no results (insufficient data)")

        return Result.ok(result)

    except Exception as e:
        return Result.fail(f"Cross-table quality analysis failed: {e}")
