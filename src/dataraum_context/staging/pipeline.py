"""Staging pipeline for CSV files.

Provides two entry points:
- stage_csv(): Single file → load, profile, resolve, statistics
- stage_csv_directory(): Directory → batch processing per stage

Both return MultiTablePipelineResult for a unified API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models import SourceConfig
from dataraum_context.core.models.base import Result
from dataraum_context.profiling.models import (
    SchemaProfileResult,
    StatisticsProfileResult,
    TypeResolutionResult,
)
from dataraum_context.profiling.profiler import profile_schema, profile_statistics
from dataraum_context.profiling.type_resolution import resolve_types
from dataraum_context.staging.loaders.csv import CSVLoader
from dataraum_context.staging.models import StagedTable, StagingResult
from dataraum_context.storage.models_v2.core import Table

# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TablePipelineResult:
    """Result for a single table in the pipeline."""

    staged_table: StagedTable
    schema_profile_result: SchemaProfileResult | None = None
    type_resolution_result: TypeResolutionResult | None = None
    statistics_profile_result: StatisticsProfileResult | None = None
    error: str | None = None

    @property
    def table_name(self) -> str:
        return self.staged_table.table_name

    @property
    def raw_table_name(self) -> str:
        return self.staged_table.raw_table_name

    @property
    def typed_table_name(self) -> str:
        if self.type_resolution_result:
            return self.type_resolution_result.typed_table_name
        return ""


@dataclass
class MultiTablePipelineResult:
    """Result of the staging pipeline (single or multi-table)."""

    source_id: str
    staging_result: StagingResult
    table_results: list[TablePipelineResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return len(self.table_results)

    @property
    def successful_tables(self) -> list[TablePipelineResult]:
        return [t for t in self.table_results if t.error is None]

    @property
    def failed_tables(self) -> list[TablePipelineResult]:
        return [t for t in self.table_results if t.error is not None]

    @property
    def total_rows(self) -> int:
        return self.staging_result.total_rows

    # Convenience for single-table results
    @property
    def table(self) -> TablePipelineResult | None:
        """Get first table result (convenience for single-file pipelines)."""
        return self.table_results[0] if self.table_results else None


# =============================================================================
# Pipeline Functions
# =============================================================================


async def stage_csv(
    file_path: str,
    table_name: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    auto_resolve_types: bool = True,
    auto_profile_statistics: bool = True,
    min_confidence: float = 0.85,
) -> Result[MultiTablePipelineResult]:
    """Stage a single CSV file through the complete pipeline.

    Args:
        file_path: Path to the CSV file
        table_name: Name for the table
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session
        auto_resolve_types: Whether to resolve types automatically
        auto_profile_statistics: Whether to profile statistics
        min_confidence: Minimum confidence for type resolution

    Returns:
        Result containing MultiTablePipelineResult with single table
    """
    # Load CSV
    loader = CSVLoader()
    load_result = await loader.load(
        source_config=SourceConfig(name=table_name, source_type="csv", path=file_path),
        duckdb_conn=duckdb_conn,
        session=session,
    )

    if not load_result.success:
        return Result.fail(f"CSV loading failed: {load_result.error}")

    staging_result = load_result.unwrap()
    if not staging_result.tables:
        return Result.fail("No tables created during staging")

    # Run profiling stages
    table_results, warnings = await _run_profiling_stages(
        staged_tables=staging_result.tables,
        duckdb_conn=duckdb_conn,
        session=session,
        auto_resolve_types=auto_resolve_types,
        auto_profile_statistics=auto_profile_statistics,
        min_confidence=min_confidence,
    )

    return Result.ok(
        MultiTablePipelineResult(
            source_id=staging_result.source_id,
            staging_result=staging_result,
            table_results=table_results,
            warnings=warnings,
        ),
        warnings=warnings,
    )


async def stage_csv_directory(
    directory_path: str,
    source_name: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    file_pattern: str = "*.csv",
    auto_resolve_types: bool = True,
    auto_profile_statistics: bool = True,
    min_confidence: float = 0.85,
) -> Result[MultiTablePipelineResult]:
    """Stage all CSV files from a directory with batch-per-stage processing.

    This batch approach is crucial for downstream relationship detection
    (TDA topology) which needs all tables loaded before analysis.

    Args:
        directory_path: Path to directory containing CSV files
        source_name: Name for the source (dataset name)
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session
        file_pattern: Glob pattern for CSV files (default: "*.csv")
        auto_resolve_types: Whether to resolve types automatically
        auto_profile_statistics: Whether to profile statistics
        min_confidence: Minimum confidence for type resolution

    Returns:
        Result containing MultiTablePipelineResult
    """
    # Load all CSV files
    loader = CSVLoader()
    load_result = await loader.load_directory(
        directory_path=directory_path,
        source_name=source_name,
        duckdb_conn=duckdb_conn,
        session=session,
        file_pattern=file_pattern,
    )

    if not load_result.success:
        return Result.fail(f"CSV directory loading failed: {load_result.error}")

    staging_result = load_result.unwrap()
    if not staging_result.tables:
        return Result.fail("No tables created during staging")

    # Run profiling stages
    table_results, warnings = await _run_profiling_stages(
        staged_tables=staging_result.tables,
        duckdb_conn=duckdb_conn,
        session=session,
        auto_resolve_types=auto_resolve_types,
        auto_profile_statistics=auto_profile_statistics,
        min_confidence=min_confidence,
    )
    warnings = load_result.warnings + warnings

    return Result.ok(
        MultiTablePipelineResult(
            source_id=staging_result.source_id,
            staging_result=staging_result,
            table_results=table_results,
            warnings=warnings,
        ),
        warnings=warnings,
    )


# =============================================================================
# Internal Helpers
# =============================================================================


async def _run_profiling_stages(
    staged_tables: list[StagedTable],
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    auto_resolve_types: bool,
    auto_profile_statistics: bool,
    min_confidence: float,
) -> tuple[list[TablePipelineResult], list[str]]:
    """Run profiling stages 2-4 for a list of tables.

    Stages:
    2. Schema profiling (type discovery)
    3. Type resolution (if enabled)
    4. Statistics profiling (if enabled)

    Returns:
        Tuple of (table_results, warnings)
    """
    warnings: list[str] = []
    results: dict[str, TablePipelineResult] = {
        t.table_id: TablePipelineResult(staged_table=t) for t in staged_tables
    }

    # Stage 2: Schema profiling
    for table_id, result in results.items():
        schema_result = await profile_schema(table_id, duckdb_conn, session)
        if schema_result.success:
            result.schema_profile_result = schema_result.unwrap()
        else:
            result.error = f"Schema profiling failed: {schema_result.error}"
            warnings.append(
                f"Schema profiling failed for {result.table_name}: {schema_result.error}"
            )

    # Stage 3: Type resolution
    typed_table_ids: dict[str, str] = {}
    if auto_resolve_types:
        for table_id, result in results.items():
            if result.error:
                continue
            type_result = await resolve_types(table_id, duckdb_conn, session, min_confidence)
            if type_result.success:
                result.type_resolution_result = type_result.unwrap()
                typed_id = await _get_typed_table_id(
                    result.type_resolution_result.typed_table_name, session
                )
                if typed_id:
                    typed_table_ids[table_id] = typed_id
            else:
                result.error = f"Type resolution failed: {type_result.error}"
                warnings.append(
                    f"Type resolution failed for {result.table_name}: {type_result.error}"
                )

    # Stage 4: Statistics profiling
    if auto_resolve_types and auto_profile_statistics:
        for raw_id, typed_id in typed_table_ids.items():
            result = results[raw_id]
            if result.error:
                continue
            stats_result = await profile_statistics(
                typed_id, duckdb_conn, session, include_correlations=True
            )
            if stats_result.success:
                result.statistics_profile_result = stats_result.unwrap()
            else:
                warnings.append(
                    f"Statistics profiling failed for {result.table_name}: {stats_result.error}"
                )

    return list(results.values()), warnings


async def _get_typed_table_id(typed_table_name: str, session: AsyncSession) -> str | None:
    """Get the table ID for a typed table by DuckDB path."""
    stmt = select(Table).where(Table.duckdb_path == typed_table_name)
    result = await session.execute(stmt)
    table = result.scalar_one_or_none()
    return table.table_id if table else None
