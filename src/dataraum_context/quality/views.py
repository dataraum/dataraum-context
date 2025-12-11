"""Quality views - DuckDB views for System 2 metrics (Phase 7).

Creates DuckDB views that join all quality metrics from metadata tables.
These views provide a unified query interface for:
- AI context generation
- Quality dashboards and reports
- LLM filtering agent (Phase 8)
- Trend analysis over time

Key Design:
- Views join SQLAlchemy metadata tables
- DuckDB can query SQLite/PostgreSQL via ATTACH
- Views are virtual (no data duplication)
- Metadata stays in SQLAlchemy (single source of truth)
"""

import logging
from typing import Any

import duckdb

from dataraum_context.core.models.base import Result

logger = logging.getLogger(__name__)


async def create_quality_views(duckdb_conn: duckdb.DuckDBPyConnection) -> Result[None]:
    """Create DuckDB views joining all quality metrics.

    This creates two main views:
    1. column_quality_assessment - All column-level quality metrics
    2. table_quality_assessment - Aggregated table-level metrics

    Args:
        duckdb_conn: DuckDB connection (must have metadata DB attached)

    Returns:
        Result indicating success/failure

    Example:
        >>> await create_quality_views(duckdb_conn)
        >>> # Query column quality
        >>> result = duckdb_conn.execute('''
        ...     SELECT column_name, null_ratio, benford_compliant
        ...     FROM column_quality_assessment
        ...     WHERE table_id = 'table-123'
        ... ''').fetchdf()

    Note:
        Requires metadata database to be attached to DuckDB connection.
        Use: duckdb_conn.execute("ATTACH 'metadata.db' AS metadata")
    """
    try:
        # View 1: Column Quality Assessment
        # Joins all column-level quality metrics
        create_column_view_sql = """
        CREATE OR REPLACE VIEW column_quality_assessment AS
        SELECT
            -- Column identity
            c.column_id,
            c.column_name,
            c.table_id,
            c.resolved_type,

            -- Profiling metrics (from statistical_profiles)
            sp.null_ratio,
            sp.cardinality_ratio,
            sp.mean,
            sp.stddev,
            sp.distinct_count,
            sp.total_count,

            -- Statistical quality metrics
            sqm.quality_score AS statistical_quality_score,
            sqm.benford_compliant,
            sqm.iqr_outlier_ratio,
            sqm.isolation_forest_anomaly_ratio,

            -- Temporal quality metrics
            tqm.completeness_ratio AS temporal_completeness,
            tqm.has_seasonality,
            tqm.has_trend,
            tqm.is_stale,
            tqm.temporal_quality_score,

            -- Multicollinearity metrics
            mm.vif,
            mm.has_multicollinearity,

            -- Semantic annotations
            sa.semantic_role,
            sa.entity_type,
            sa.confidence AS semantic_confidence

        FROM metadata.columns c

        -- Profiling (should always exist)
        LEFT JOIN metadata.statistical_profiles sp
            ON c.column_id = sp.column_id

        -- Statistical quality (exists for numeric columns)
        LEFT JOIN metadata.statistical_quality_metrics sqm
            ON c.column_id = sqm.column_id

        -- Temporal quality (exists for time columns)
        LEFT JOIN metadata.temporal_quality_metrics tqm
            ON c.column_id = tqm.column_id

        -- Multicollinearity (exists when VIF computed)
        LEFT JOIN metadata.multicollinearity_metrics mm
            ON c.column_id = mm.column_id

        -- Semantic annotations (exists if semantic enrichment ran)
        LEFT JOIN metadata.semantic_annotations sa
            ON c.column_id = sa.column_id

        ORDER BY c.table_id, c.column_name
        """

        duckdb_conn.execute(create_column_view_sql)
        logger.info("Created column_quality_assessment view")

        # View 2: Table Quality Assessment
        # Aggregates column metrics + table-level metrics
        create_table_view_sql = """
        CREATE OR REPLACE VIEW table_quality_assessment AS
        SELECT
            -- Table identity
            t.table_id,
            t.table_name,
            t.source_id,

            -- Row/column counts
            t.row_count,
            t.column_count,

            -- Aggregated column quality
            AVG(cqa.null_ratio) AS avg_null_ratio,
            AVG(cqa.cardinality_ratio) AS avg_cardinality_ratio,
            AVG(cqa.statistical_quality_score) AS avg_statistical_quality,
            AVG(cqa.temporal_quality_score) AS avg_temporal_quality,

            -- Statistical quality issues
            SUM(CASE WHEN cqa.benford_compliant = FALSE THEN 1 ELSE 0 END) AS benford_violations,
            AVG(cqa.iqr_outlier_ratio) AS avg_outlier_ratio,

            -- Temporal quality issues
            SUM(CASE WHEN cqa.has_seasonality = TRUE THEN 1 ELSE 0 END) AS seasonal_columns,
            SUM(CASE WHEN cqa.has_trend = TRUE THEN 1 ELSE 0 END) AS trending_columns,
            SUM(CASE WHEN cqa.is_stale = TRUE THEN 1 ELSE 0 END) AS stale_columns,

            -- Multicollinearity issues
            SUM(CASE WHEN cqa.has_multicollinearity = TRUE THEN 1 ELSE 0 END) AS multicollinear_columns,
            AVG(cqa.vif) AS avg_vif,

            -- Topological quality (table-level)
            topm.betti_0 AS connected_components,
            topm.betti_1 AS cycles_count,
            topm.has_cycles,
            topm.persistence_score AS topological_persistence,

            -- Domain quality (table-level, financial only)
            dqm.financial_quality_score,
            dqm.double_entry_balanced,

            -- Quality synthesis (computed scores)
            qs.overall_quality_score,
            qs.total_issues,
            qs.critical_issues

        FROM metadata.tables t

        -- Column quality (aggregated)
        LEFT JOIN column_quality_assessment cqa
            ON t.table_id = cqa.table_id

        -- Topological quality (table-level)
        LEFT JOIN metadata.topological_quality_metrics topm
            ON t.table_id = topm.table_id

        -- Domain quality (table-level)
        LEFT JOIN metadata.domain_quality_metrics dqm
            ON t.table_id = dqm.table_id

        -- Quality synthesis (table-level)
        LEFT JOIN metadata.quality_synthesis_results qs
            ON t.table_id = qs.table_id

        GROUP BY
            t.table_id, t.table_name, t.source_id,
            t.row_count, t.column_count,
            topm.betti_0, topm.betti_1, topm.has_cycles, topm.persistence_score,
            dqm.financial_quality_score, dqm.double_entry_balanced,
            qs.overall_quality_score, qs.total_issues, qs.critical_issues

        ORDER BY t.table_name
        """

        duckdb_conn.execute(create_table_view_sql)
        logger.info("Created table_quality_assessment view")

        return Result.ok(None)

    except Exception as e:
        logger.error(f"Failed to create quality views: {e}")
        return Result.fail(f"Quality view creation failed: {e}")


async def query_column_quality(
    duckdb_conn: duckdb.DuckDBPyConnection,
    table_id: str | None = None,
    min_quality_score: float | None = None,
) -> Result[list[dict[str, Any]]]:
    """Query column quality assessment view with filters.

    Args:
        duckdb_conn: DuckDB connection
        table_id: Filter by table ID (optional)
        min_quality_score: Filter by minimum quality score (optional)

    Returns:
        Result containing list of column quality records

    Example:
        >>> result = await query_column_quality(duckdb_conn, table_id="table-123", min_quality_score=0.7)
        >>> for col in result.value:
        ...     print(f"{col['column_name']}: {col['statistical_quality_score']}")
    """
    try:
        query = "SELECT * FROM column_quality_assessment WHERE 1=1"

        if table_id:
            query += f" AND table_id = '{table_id}'"

        if min_quality_score is not None:
            query += f" AND statistical_quality_score >= {min_quality_score}"

        result = duckdb_conn.execute(query).fetchdf()
        records: list[dict[str, Any]] = result.to_dict(orient="records")  # type: ignore

        return Result.ok(records)

    except Exception as e:
        logger.error(f"Failed to query column quality: {e}")
        return Result.fail(f"Column quality query failed: {e}")


async def query_table_quality(
    duckdb_conn: duckdb.DuckDBPyConnection,
    source_id: str | None = None,
    has_critical_issues: bool | None = None,
) -> Result[list[dict[str, Any]]]:
    """Query table quality assessment view with filters.

    Args:
        duckdb_conn: DuckDB connection
        source_id: Filter by source ID (optional)
        has_critical_issues: Filter tables with critical issues (optional)

    Returns:
        Result containing list of table quality records

    Example:
        >>> result = await query_table_quality(duckdb_conn, has_critical_issues=True)
        >>> for table in result.value:
        ...     print(f"{table['table_name']}: {table['critical_issues']} critical issues")
    """
    try:
        query = "SELECT * FROM table_quality_assessment WHERE 1=1"

        if source_id:
            query += f" AND source_id = '{source_id}'"

        if has_critical_issues is not None:
            if has_critical_issues:
                query += " AND critical_issues > 0"
            else:
                query += " AND (critical_issues IS NULL OR critical_issues = 0)"

        result = duckdb_conn.execute(query).fetchdf()
        records: list[dict[str, Any]] = result.to_dict(orient="records")  # type: ignore

        return Result.ok(records)

    except Exception as e:
        logger.error(f"Failed to query table quality: {e}")
        return Result.fail(f"Table quality query failed: {e}")
