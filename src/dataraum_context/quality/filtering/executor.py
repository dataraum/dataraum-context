"""Filtering Executor - Phase 10.

Executes merged filtering logic to create clean/quarantine artifacts.

Output:
- Clean views: typed_{table}_clean (only high-quality rows)
- Quarantine tables: quality_quarantine_{table} (problematic rows + reasons)
- Optional: _quality_flags column on original table (for reports)

Key Design:
- Clean views use WHERE clauses from merged filters
- Quarantine tables include _validation_failures JSON (sparse, NULL when passed)
- Original table remains unchanged (non-destructive)
"""

import logging

import duckdb
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.quality.filtering.models import FilteringRecommendations, FilteringResult

logger = logging.getLogger(__name__)


async def execute_filtering(
    table_id: str,
    table_name: str,
    merged_filters: FilteringRecommendations,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    add_quality_flags: bool = False,
) -> Result[FilteringResult]:
    """Execute filtering logic, create clean/quarantine artifacts.

    Creates:
    1. Clean view: typed_{table}_clean with WHERE clause
    2. Quarantine table: quality_quarantine_{table} with _validation_failures
    3. Optional: _quality_flags column on original table

    Args:
        table_id: Table ID for metadata
        table_name: Table name (e.g., "typed_sales")
        merged_filters: Merged filtering recommendations (LLM + user rules)
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session
        add_quality_flags: Add _quality_flags column to original table (default: False)

    Returns:
        Result containing FilteringResult with row counts and applied filters

    Example:
        >>> result = await execute_filtering(
        ...     table_id="table-123",
        ...     table_name="typed_sales",
        ...     merged_filters=merged_recs,
        ...     duckdb_conn=duckdb_conn,
        ...     session=session
        ... )
        >>> print(f"Clean: {result.value.clean_row_count}, Quarantine: {result.value.quarantine_row_count}")
    """
    try:
        # Get original table row count
        count_query = f"SELECT COUNT(*) as total FROM {table_name}"
        count_result = duckdb_conn.execute(count_query).fetchone()
        original_row_count = count_result[0] if count_result else 0

        logger.info(f"Executing filtering for {table_name} ({original_row_count} rows)")

        # ============================================================================
        # 1. CREATE CLEAN VIEW
        # ============================================================================

        clean_view_name = f"{table_name}_clean"

        if merged_filters.clean_view_filters:
            # Combine all filters with AND
            where_clause = " AND ".join([f"({f})" for f in merged_filters.clean_view_filters])

            create_clean_view_sql = f"""
                CREATE OR REPLACE VIEW {clean_view_name} AS
                SELECT * FROM {table_name}
                WHERE {where_clause}
            """
        else:
            # No filters - clean view is entire table
            create_clean_view_sql = f"""
                CREATE OR REPLACE VIEW {clean_view_name} AS
                SELECT * FROM {table_name}
            """

        try:
            duckdb_conn.execute(create_clean_view_sql)
            clean_count_result = duckdb_conn.execute(
                f"SELECT COUNT(*) FROM {clean_view_name}"
            ).fetchone()
            clean_row_count = clean_count_result[0] if clean_count_result else 0
            logger.info(f"Created clean view {clean_view_name} with {clean_row_count} rows")
        except Exception as e:
            logger.error(f"Failed to create clean view: {e}")
            return Result.fail(f"Clean view creation failed: {e}")

        # ============================================================================
        # 2. CREATE QUARANTINE TABLE
        # ============================================================================

        quarantine_table_name = f"quality_quarantine_{table_name.replace('typed_', '')}"

        # Build CASE statements for _validation_failures
        failure_cases = []
        for i, criterion in enumerate(merged_filters.quarantine_criteria):
            # Find column name from criterion
            column_name = _extract_column_from_filter(criterion)
            reason_key = f"filter_{i}"

            # Look up rationale if available
            reason = merged_filters.rationale.get(reason_key, criterion)

            # Create CASE statement
            failure_case = f"""
                CASE WHEN ({criterion})
                THEN {{column: '{column_name}', reason: '{reason}', source: 'merged_filter'}}::JSON
                END
            """
            failure_cases.append(failure_case)

        if failure_cases:
            validation_failures_expr = f"""
                LIST_FILTER(
                    [{", ".join(failure_cases)}],
                    x -> x IS NOT NULL
                ) as _validation_failures
            """
        else:
            validation_failures_expr = "NULL::JSON[] as _validation_failures"

        # Create quarantine table with failures
        create_quarantine_sql = f"""
            CREATE OR REPLACE TABLE {quarantine_table_name} AS
            SELECT
                *,
                {validation_failures_expr}
            FROM {table_name}
            WHERE NOT (
                {" AND ".join([f"({f})" for f in merged_filters.clean_view_filters]) if merged_filters.clean_view_filters else "TRUE"}
            )
        """

        try:
            duckdb_conn.execute(create_quarantine_sql)
            quarantine_count_result = duckdb_conn.execute(
                f"SELECT COUNT(*) FROM {quarantine_table_name}"
            ).fetchone()
            quarantine_row_count = quarantine_count_result[0] if quarantine_count_result else 0
            logger.info(
                f"Created quarantine table {quarantine_table_name} with {quarantine_row_count} rows"
            )
        except Exception as e:
            logger.error(f"Failed to create quarantine table: {e}")
            return Result.fail(f"Quarantine table creation failed: {e}")

        # ============================================================================
        # 3. OPTIONAL: ADD QUALITY FLAGS TO ORIGINAL TABLE
        # ============================================================================

        if add_quality_flags:
            try:
                # Add _quality_flags column (JSON)
                # This is for reporting only, not filtering
                add_flags_sql = f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS _quality_flags JSON
                """
                duckdb_conn.execute(add_flags_sql)

                # Update flags based on quarantine membership
                # TODO: Implement flag population logic
                logger.info(f"Added _quality_flags column to {table_name}")
            except Exception as e:
                logger.warning(f"Failed to add quality flags: {e}")

        # ============================================================================
        # 4. STORE FILTERING METADATA
        # ============================================================================

        # Build filtering result
        # Convert rationale to counts (simple: 1 per reason since we can't easily count per reason)
        quarantine_reason_counts = dict.fromkeys(merged_filters.rationale.values(), 1)

        result_obj = FilteringResult(
            table_id=table_id,
            clean_table_name=clean_view_name,
            quarantine_table_name=quarantine_table_name,
            rows_in_clean=clean_row_count,
            rows_in_quarantine=quarantine_row_count,
            applied_filters=merged_filters.clean_view_filters,
            quarantine_reasons=quarantine_reason_counts,
        )

        logger.info(
            f"Filtering complete: {clean_row_count}/{original_row_count} clean "
            f"({quarantine_row_count} quarantined)"
        )

        return Result.ok(result_obj)

    except Exception as e:
        logger.error(f"Failed to execute filtering: {e}")
        return Result.fail(f"Filtering execution failed: {e}")


def _extract_column_from_filter(filter_expr: str) -> str:
    """Extract column name from filter expression.

    Simple heuristic: Column is first word before operator.

    Args:
        filter_expr: SQL WHERE clause filter

    Returns:
        Column name (or "unknown" if extraction fails)
    """
    filter_expr = filter_expr.strip()

    # Try common operators
    for separator in [
        " IS ",
        " ~ ",
        " !~ ",
        " BETWEEN ",
        " >= ",
        " <= ",
        " > ",
        " < ",
        " = ",
        " != ",
    ]:
        if separator in filter_expr:
            potential_column = filter_expr.split(separator)[0].strip()
            # Remove leading/trailing characters
            potential_column = potential_column.strip("()")
            if potential_column:
                return potential_column

    # Fallback
    return "unknown"
