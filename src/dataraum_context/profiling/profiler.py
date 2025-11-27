"""Main profiling orchestrator."""

import time
from uuid import UUID

import duckdb
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models import ProfileResult, Result
from dataraum_context.profiling.statistical import compute_statistical_profile
from dataraum_context.profiling.type_inference import infer_type_candidates
from dataraum_context.storage.models import Table


async def profile_table(
    table_id: UUID,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
) -> Result[ProfileResult]:
    """Profile a table to generate statistical metadata and type candidates.

    This is the main entry point for profiling. It:
    1. Computes statistical profiles (counts, nulls, distributions)
    2. Detects patterns and infers type candidates
    3. Stores results in metadata database

    Args:
        table_id: Table ID to profile
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session

    Returns:
        Result containing ProfileResult or error
    """
    start_time = time.time()

    try:
        # Get table from metadata
        table = await session.get(Table, str(table_id))
        if not table:
            return Result.fail(f"Table not found: {table_id}")

        if not table.duckdb_path:
            return Result.fail(f"Table has no DuckDB path: {table_id}")

        # Step 1: Statistical profiling
        stats_result = await compute_statistical_profile(
            table=table,
            duckdb_conn=duckdb_conn,
            session=session,
        )

        if not stats_result.success:
            return Result.fail(f"Statistical profiling failed: {stats_result.error}")

        # Step 2: Type inference (only for 'raw' layer tables)
        type_candidates = []
        if table.layer == "raw":
            type_result = await infer_type_candidates(
                table=table,
                duckdb_conn=duckdb_conn,
                session=session,
            )

            if not type_result.success:
                return Result.fail(f"Type inference failed: {type_result.error}")

            type_candidates = type_result.value

        # Update table's last_profiled_at
        from datetime import UTC, datetime

        table.last_profiled_at = datetime.now(UTC)
        await session.commit()

        # Build result
        duration = time.time() - start_time

        result = ProfileResult(
            profiles=stats_result.value,
            type_candidates=type_candidates,
            duration_seconds=duration,
        )

        return Result.ok(result)

    except Exception as e:
        return Result.fail(f"Profiling failed: {e}")
