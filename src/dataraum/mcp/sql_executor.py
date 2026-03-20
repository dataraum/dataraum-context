"""Direct SQL execution with quality metadata and snippet integration.

Core logic for the run_sql MCP tool. Converts caller-provided SQL (raw or
structured steps) into execute_sql_steps() calls, enriches results with
per-column quality metadata, and integrates with the snippet library.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataraum.query.execution import ExecutionResult, SQLStep, execute_sql_steps

if TYPE_CHECKING:
    import duckdb
    from sqlalchemy.orm import Session

_log = logging.getLogger(__name__)

# Hard ceiling on row limit to prevent enormous responses.
MAX_ROW_LIMIT = 10_000
DEFAULT_ROW_LIMIT = 100


def run_sql(
    cursor: duckdb.DuckDBPyConnection,
    *,
    session: Session | None = None,
    table_ids: list[str] | None = None,
    steps: list[dict[str, Any]] | None = None,
    sql: str | None = None,
    limit: int = DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Execute SQL and return results as a structured dict.

    Args:
        cursor: DuckDB connection.
        session: SQLAlchemy session (needed for quality metadata).
        table_ids: Typed table IDs (needed for quality metadata).
        steps: Structured SQL steps (list of dicts with step_id, sql, description,
            optional column_mappings).
        sql: Raw SQL convenience mode. Mutually exclusive with steps.
        limit: Max rows to return. Capped at MAX_ROW_LIMIT.

    Returns:
        Dict with columns, rows, row_count, truncated, steps_executed.
        On error, returns dict with "error" key.
    """
    # --- Validate input ---
    if steps is not None and sql is not None:
        return {"error": "Provide either 'steps' or 'sql', not both."}
    if steps is None and sql is None:
        return {"error": "Provide either 'steps' or 'sql'."}

    # Clamp limit
    effective_limit = min(max(1, limit), MAX_ROW_LIMIT)

    # --- Build SQLStep list + final_sql ---
    sql_steps: list[SQLStep]
    final_sql: str

    if sql is not None:
        # Convenience mode: wrap raw SQL as single step
        sql_steps = [SQLStep(step_id="query", sql=sql, description="Raw SQL query")]
        final_sql = "SELECT * FROM query"
    else:
        assert steps is not None
        sql_steps = [
            SQLStep(
                step_id=s["step_id"],
                sql=s["sql"],
                description=s.get("description", ""),
            )
            for s in steps
        ]
        # Final SQL selects from the last step
        last_step_id = sql_steps[-1].step_id
        final_sql = f"SELECT * FROM {last_step_id}"

    # --- Execute ---
    result = execute_sql_steps(
        steps=sql_steps,
        final_sql=final_sql,
        duckdb_conn=cursor,
        repair_fn=None,
        return_table=True,
    )

    if not result.success or not result.value:
        return {"error": str(result.error)}

    exec_result: ExecutionResult = result.value
    columns = exec_result.columns or []
    all_rows = exec_result.rows or []
    total_rows = len(all_rows)
    sliced_rows = all_rows[:effective_limit]

    # Convert to list-of-dicts
    rows_as_dicts = [dict(zip(columns, row)) for row in sliced_rows]

    # --- Quality metadata (best-effort) ---
    column_quality: dict[str, Any] | None = None
    quality_caveat: str | None = None
    if session is not None and table_ids:
        # Merge column_mappings from all steps
        merged_mappings: dict[str, str] = {}
        if steps is not None:
            for s in steps:
                mappings = s.get("column_mappings")
                if mappings:
                    merged_mappings.update(mappings)

        try:
            column_quality, quality_caveat = _build_column_quality(
                session, table_ids, columns, merged_mappings
            )
        except Exception:
            _log.debug("Quality metadata lookup failed", exc_info=True)

    from dataraum.mcp.formatters import format_run_sql_result

    return format_run_sql_result(
        columns=columns,
        rows=rows_as_dicts,
        step_results=exec_result.step_results,
        limit=effective_limit,
        total_rows=total_rows,
        column_quality=column_quality,
        quality_caveat=quality_caveat,
    )


def _build_column_quality(
    session: Session,
    table_ids: list[str],
    output_columns: list[str],
    column_mappings: dict[str, str],
) -> tuple[dict[str, Any | None], str | None]:
    """Look up quality metadata for output columns.

    Args:
        session: SQLAlchemy session.
        table_ids: Typed table IDs for this pipeline run.
        output_columns: Column names in the SQL result.
        column_mappings: Maps output col name → source col name (e.g. revenue → Betrag).

    Returns:
        (column_quality dict, caveat string or None)
    """
    from sqlalchemy import select

    from dataraum.analysis.quality_summary.db_models import ColumnQualityReport
    from dataraum.pipeline.db_models import PhaseLog
    from dataraum.storage import Column as ColumnModel, Table

    # Build lookup: (table_name, column_name) → quality info
    # First load all quality reports for these tables
    tables = session.execute(
        select(Table).where(Table.table_id.in_(table_ids))
    ).scalars().all()
    table_names = {t.table_id: t.table_name for t in tables}

    # Get all column IDs for these tables
    all_columns = session.execute(
        select(ColumnModel).where(ColumnModel.table_id.in_(table_ids))
    ).scalars().all()

    # Map (table_name, column_name) → column_id
    col_id_map: dict[tuple[str, str], str] = {}
    for col in all_columns:
        tname = table_names.get(col.table_id, "")
        col_id_map[(tname, col.column_name)] = col.column_id

    # Load quality reports indexed by source_column_id
    quality_by_col_id: dict[str, ColumnQualityReport] = {}
    col_ids = list(col_id_map.values())
    if col_ids:
        reports = session.execute(
            select(ColumnQualityReport).where(
                ColumnQualityReport.source_column_id.in_(col_ids)
            )
        ).scalars().all()
        for r in reports:
            quality_by_col_id[r.source_column_id] = r

    # Build readiness from entropy network context
    readiness_by_col: dict[str, str] = {}
    try:
        from dataraum.entropy.views.network_context import build_for_network
        from dataraum.entropy.views.query_context import network_to_column_summaries

        network_ctx = build_for_network(session, table_ids)
        if network_ctx and network_ctx.total_columns > 0:
            col_summaries = network_to_column_summaries(network_ctx)
            for key, summary in col_summaries.items():
                readiness_by_col[key] = summary.readiness
    except Exception:
        _log.debug("Entropy readiness lookup failed", exc_info=True)

    # Check if entropy phase has completed
    caveat: str | None = None
    try:
        source_ids = {t.source_id for t in tables}
        for source_id in source_ids:
            entropy_log = session.execute(
                select(PhaseLog).where(
                    PhaseLog.source_id == source_id,
                    PhaseLog.phase_name == "entropy",
                    PhaseLog.status == "completed",
                ).limit(1)
            ).scalar_one_or_none()
            if not entropy_log:
                caveat = "Quality metadata incomplete: entropy phase has not run yet"
                break
    except Exception:
        pass

    # For each output column, resolve to source and look up quality
    column_quality: dict[str, Any | None] = {}
    for out_col in output_columns:
        # Resolve source column via mappings or direct match
        source_col = column_mappings.get(out_col, out_col)

        # Try to find the source column across all tables
        found = False
        for tname in table_names.values():
            col_id = col_id_map.get((tname, source_col))
            if col_id is None:
                continue

            entry: dict[str, Any] = {"source_column": f"{tname}.{source_col}"}

            report = quality_by_col_id.get(col_id)
            if report:
                entry["quality_grade"] = report.quality_grade
                entry["quality_score"] = round(report.overall_quality_score, 2)

            readiness_key = f"{tname}.{source_col}"
            if readiness_key in readiness_by_col:
                entry["readiness"] = readiness_by_col[readiness_key]

            column_quality[out_col] = entry
            found = True
            break

        if not found:
            column_quality[out_col] = None

    return column_quality, caveat
