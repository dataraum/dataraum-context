"""Validation Agent - LLM-powered SQL generation for validation checks.

This agent generates SQL queries for validation checks using semantic
annotations for column resolution, with caching for reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import duckdb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.analysis.validation.config import load_all_validation_specs
from dataraum_context.analysis.validation.db_models import ValidationSQLCache
from dataraum_context.analysis.validation.models import (
    GeneratedSQL,
    ValidationResult,
    ValidationRunResult,
    ValidationSeverity,
    ValidationSpec,
    ValidationStatus,
)
from dataraum_context.analysis.validation.resolver import get_table_schema_for_llm, resolve_columns
from dataraum_context.core.models.base import Result
from dataraum_context.llm.features._base import LLMFeature

if TYPE_CHECKING:
    from dataraum_context.llm.cache import LLMCache
    from dataraum_context.llm.config import LLMConfig
    from dataraum_context.llm.prompts import PromptRenderer
    from dataraum_context.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# System prompt for SQL generation
SQL_GENERATION_SYSTEM = """You are an expert SQL developer specializing in data validation.
Your task is to generate DuckDB SQL queries that implement validation checks.

Rules:
1. Use only the columns and tables provided in the schema
2. Use DuckDB SQL syntax
3. The query should return validation results that can be evaluated
4. For balance checks: return the values being compared
5. For constraint checks: return violating rows
6. For aggregate checks: return the aggregated values
7. Always use proper quoting for column names with special characters
8. Return a single SQL query (no multiple statements)
"""

SQL_GENERATION_TEMPLATE = """Generate a DuckDB SQL query for this validation check.

## Validation Spec
Name: {spec_name}
Description: {spec_description}
Check Type: {check_type}
Parameters: {parameters}

{sql_hints}

{expected_outcome}

## Table Schema
Table: {table_name} (DuckDB path: {duckdb_path})

Columns:
{columns_info}

## Resolved Column Mappings
{column_mappings}

## Instructions
Generate a SQL query that implements this validation. The query should:
1. Use the actual column names from the mappings above
2. Return results that can be evaluated (e.g., totals to compare, violating rows)
3. Include relevant context in the output columns

Return ONLY a JSON object with this structure:
{{
  "sql": "SELECT ...",
  "explanation": "Brief explanation of what this query checks"
}}
"""


class ValidationAgent(LLMFeature):
    """LLM-powered validation agent.

    Generates SQL for validation checks using semantic annotations
    for column resolution. Caches generated SQL for reuse.
    """

    def __init__(
        self,
        config: LLMConfig,
        provider: LLMProvider,
        prompt_renderer: PromptRenderer,
        cache: LLMCache,
    ) -> None:
        """Initialize validation agent.

        Args:
            config: LLM configuration
            provider: LLM provider instance
            prompt_renderer: Prompt template renderer
            cache: Response cache
        """
        super().__init__(config, provider, prompt_renderer, cache)

    async def run_validations(
        self,
        session: AsyncSession,
        duckdb_conn: duckdb.DuckDBPyConnection,
        table_id: str,
        validation_ids: list[str] | None = None,
        category: str | None = None,
    ) -> Result[ValidationRunResult]:
        """Run validation checks on a table.

        Args:
            session: Database session
            duckdb_conn: DuckDB connection for executing SQL
            table_id: Table to validate
            validation_ids: Specific validations to run (None = all applicable)
            category: Filter by category (e.g., 'financial')

        Returns:
            Result containing ValidationRunResult
        """
        run_id = str(uuid4())
        started_at = datetime.utcnow()
        results: list[ValidationResult] = []

        # Get table info for results
        schema = await get_table_schema_for_llm(session, table_id)
        if "error" in schema:
            return Result.fail(str(schema["error"]))

        table_name = schema["table_name"]

        # Determine which validations to run
        specs = self._get_applicable_specs(validation_ids, category)

        if not specs:
            return Result.ok(
                ValidationRunResult(
                    run_id=run_id,
                    table_id=table_id,
                    table_name=table_name,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    total_checks=0,
                )
            )

        # Run each validation
        for spec in specs:
            result = await self._run_single_validation(
                session=session,
                duckdb_conn=duckdb_conn,
                table_id=table_id,
                spec=spec,
                schema=schema,
            )
            results.append(result)

        # Summarize results
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == ValidationStatus.ERROR)

        has_critical = any(
            r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.CRITICAL
            for r in results
        )

        overall = ValidationStatus.PASSED
        if failed > 0 or errors > 0:
            overall = ValidationStatus.FAILED

        run_result = ValidationRunResult(
            run_id=run_id,
            table_id=table_id,
            table_name=table_name,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            results=results,
            total_checks=len(results),
            passed_checks=passed,
            failed_checks=failed,
            skipped_checks=skipped,
            error_checks=errors,
            overall_status=overall,
            has_critical_failures=has_critical,
        )

        return Result.ok(run_result)

    def _get_applicable_specs(
        self,
        validation_ids: list[str] | None,
        category: str | None,
    ) -> list[ValidationSpec]:
        """Get validation specs to run.

        Args:
            validation_ids: Specific IDs to run
            category: Category filter

        Returns:
            List of ValidationSpecs
        """
        all_specs = load_all_validation_specs()

        if validation_ids:
            return [all_specs[vid] for vid in validation_ids if vid in all_specs]

        if category:
            return [s for s in all_specs.values() if s.category == category]

        return list(all_specs.values())

    async def _run_single_validation(
        self,
        session: AsyncSession,
        duckdb_conn: duckdb.DuckDBPyConnection,
        table_id: str,
        spec: ValidationSpec,
        schema: dict[str, Any],
    ) -> ValidationResult:
        """Run a single validation check.

        Args:
            session: Database session
            duckdb_conn: DuckDB connection
            table_id: Table being validated
            spec: Validation spec to run
            schema: Table schema from resolver

        Returns:
            ValidationResult
        """
        table_name = schema["table_name"]

        # 1. Resolve columns using semantic annotations
        resolution = await resolve_columns(session, table_id, spec)

        if not resolution.all_resolved:
            # Fail with clear error about unresolved columns
            return ValidationResult(
                validation_id=spec.validation_id,
                spec_name=spec.name,
                status=ValidationStatus.SKIPPED,
                severity=spec.severity,
                table_id=table_id,
                table_name=table_name,
                passed=False,
                message=resolution.error_message or "Column resolution failed",
                details={"unresolved_columns": resolution.unresolved},
            )

        # 2. Get or generate SQL
        sql_result = await self._get_or_generate_sql(
            session=session,
            spec=spec,
            schema=schema,
            resolution_map={alias: col.column_name for alias, col in resolution.resolved.items()},
        )

        if not sql_result.success or not sql_result.value:
            return ValidationResult(
                validation_id=spec.validation_id,
                spec_name=spec.name,
                status=ValidationStatus.ERROR,
                severity=spec.severity,
                table_id=table_id,
                table_name=table_name,
                passed=False,
                message=sql_result.error or "SQL generation failed",
            )

        generated = sql_result.value

        # 3. Execute SQL
        try:
            result_df = duckdb_conn.execute(generated.sql_query).df()
            result_rows: list[dict[str, Any]] = result_df.to_dict(orient="records")  # type: ignore[assignment]
            row_count = len(result_rows)

            # 4. Evaluate results based on check type
            passed, message, details = self._evaluate_result(
                spec=spec,
                result_rows=result_rows,
                row_count=row_count,
            )

            return ValidationResult(
                validation_id=spec.validation_id,
                spec_name=spec.name,
                status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
                severity=spec.severity,
                table_id=table_id,
                table_name=table_name,
                passed=passed,
                message=message,
                details=details,
                sql_used=generated.sql_query,
                result_rows=result_rows[:10],  # Limit stored rows
                row_count=row_count,
                columns_resolved={
                    alias: col.column_name for alias, col in resolution.resolved.items()
                },
            )

        except Exception as e:
            logger.error(f"SQL execution failed for {spec.validation_id}: {e}")
            return ValidationResult(
                validation_id=spec.validation_id,
                spec_name=spec.name,
                status=ValidationStatus.ERROR,
                severity=spec.severity,
                table_id=table_id,
                table_name=table_name,
                passed=False,
                message=f"SQL execution error: {e}",
                sql_used=generated.sql_query,
            )

    async def _get_or_generate_sql(
        self,
        session: AsyncSession,
        spec: ValidationSpec,
        schema: dict[str, Any],
        resolution_map: dict[str, str],
    ) -> Result[GeneratedSQL]:
        """Get SQL from cache or generate via LLM.

        Args:
            session: Database session
            spec: Validation spec
            schema: Table schema
            resolution_map: Resolved column mappings

        Returns:
            Result containing GeneratedSQL
        """
        # Build cache key from spec + schema + resolution
        cache_key = self._build_cache_key(spec, schema, resolution_map)

        # Check cache first
        cached = await self._get_cached_sql(session, cache_key)
        if cached:
            logger.debug(f"Using cached SQL for {spec.validation_id}")
            return Result.ok(cached)

        # Generate via LLM
        sql_result = await self._generate_sql(session, spec, schema, resolution_map)

        if sql_result.success and sql_result.value:
            # Cache the generated SQL
            await self._cache_sql(session, cache_key, sql_result.value)

        return sql_result

    def _build_cache_key(
        self,
        spec: ValidationSpec,
        schema: dict[str, Any],
        resolution_map: dict[str, str],
    ) -> str:
        """Build a cache key for SQL lookup.

        Args:
            spec: Validation spec
            schema: Table schema
            resolution_map: Resolved column mappings

        Returns:
            Cache key string
        """
        key_data = {
            "validation_id": spec.validation_id,
            "version": spec.version,
            "table_name": schema["table_name"],
            "columns": sorted(resolution_map.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    async def _get_cached_sql(
        self,
        session: AsyncSession,
        cache_key: str,
    ) -> GeneratedSQL | None:
        """Get cached SQL if available.

        Args:
            session: Database session
            cache_key: Cache key

        Returns:
            GeneratedSQL or None if not cached
        """
        query = select(ValidationSQLCache).where(ValidationSQLCache.cache_key == cache_key)
        result = await session.execute(query)
        cached = result.scalar_one_or_none()

        if cached:
            return GeneratedSQL(
                validation_id=cached.validation_id,
                sql_query=cached.sql_query,
                explanation=cached.explanation,
                generated_at=cached.generated_at,
                model_used=cached.model_used,
                prompt_hash=cache_key,
            )

        return None

    async def _cache_sql(
        self,
        session: AsyncSession,
        cache_key: str,
        generated: GeneratedSQL,
    ) -> None:
        """Cache generated SQL.

        Args:
            session: Database session
            cache_key: Cache key
            generated: Generated SQL to cache
        """
        cache_entry = ValidationSQLCache(
            cache_key=cache_key,
            validation_id=generated.validation_id,
            sql_query=generated.sql_query,
            explanation=generated.explanation,
            generated_at=generated.generated_at,
            model_used=generated.model_used,
        )
        session.add(cache_entry)
        await session.commit()

    async def _generate_sql(
        self,
        session: AsyncSession,
        spec: ValidationSpec,
        schema: dict[str, Any],
        resolution_map: dict[str, str],
    ) -> Result[GeneratedSQL]:
        """Generate SQL via LLM.

        Args:
            session: Database session
            spec: Validation spec
            schema: Table schema
            resolution_map: Resolved column mappings

        Returns:
            Result containing GeneratedSQL
        """
        # Build columns info
        columns_info = []
        for col in schema.get("columns", []):
            col_line = f"  - {col['column_name']} ({col.get('data_type', 'unknown')})"
            if "semantic" in col:
                sem = col["semantic"]
                if sem.get("entity_type"):
                    col_line += f" [entity: {sem['entity_type']}]"
                if sem.get("role"):
                    col_line += f" [role: {sem['role']}]"
            columns_info.append(col_line)

        # Build column mappings
        mappings = [f'  {alias} -> "{actual}"' for alias, actual in resolution_map.items()]

        # Build prompt
        sql_hints = f"SQL Hints: {spec.sql_hints}" if spec.sql_hints else ""
        expected = f"Expected Outcome: {spec.expected_outcome}" if spec.expected_outcome else ""

        prompt = SQL_GENERATION_TEMPLATE.format(
            spec_name=spec.name,
            spec_description=spec.description,
            check_type=spec.check_type,
            parameters=json.dumps(spec.parameters) if spec.parameters else "None",
            sql_hints=sql_hints,
            expected_outcome=expected,
            table_name=schema["table_name"],
            duckdb_path=schema["duckdb_path"],
            columns_info="\n".join(columns_info),
            column_mappings="\n".join(mappings),
        )

        # Get feature config
        feature_config = self.config.features.validation
        if not feature_config or not feature_config.enabled:
            return Result.fail("Validation feature is disabled in config")

        # Call LLM
        from dataraum_context.llm.providers.base import LLMRequest

        model = self.provider.get_model_for_tier(feature_config.model_tier)

        request = LLMRequest(
            prompt=prompt,
            system=SQL_GENERATION_SYSTEM,
            max_tokens=2000,
            temperature=0.0,  # Deterministic for SQL
            response_format="json",
        )

        result = await self.provider.complete(request)
        if not result.success or not result.value:
            return Result.fail(result.error or "LLM call failed")

        # Parse response
        try:
            response_data = json.loads(result.value.content)
            sql = response_data.get("sql", "")
            explanation = response_data.get("explanation", "")

            if not sql:
                return Result.fail("LLM did not return SQL")

            generated = GeneratedSQL(
                validation_id=spec.validation_id,
                sql_query=sql,
                explanation=explanation,
                generated_at=datetime.utcnow(),
                model_used=model,
            )

            return Result.ok(generated)

        except json.JSONDecodeError as e:
            return Result.fail(f"Failed to parse LLM response: {e}")

    def _evaluate_result(
        self,
        spec: ValidationSpec,
        result_rows: list[dict[str, Any]],
        row_count: int,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate validation result based on check type.

        Args:
            spec: Validation spec
            result_rows: Query result rows
            row_count: Total row count

        Returns:
            Tuple of (passed, message, details)
        """
        check_type = spec.check_type
        params = spec.parameters

        if check_type == "balance":
            # Balance checks compare two values
            if row_count == 0:
                return (False, "No results returned", {})

            row = result_rows[0]
            tolerance = params.get("tolerance", 0.01)

            # Find the columns to compare
            value_cols = [k for k in row.keys() if "total" in k.lower() or "sum" in k.lower()]
            if len(value_cols) >= 2:
                val1 = float(row[value_cols[0]] or 0)
                val2 = float(row[value_cols[1]] or 0)
                diff = abs(val1 - val2)
                passed = diff <= tolerance
                return (
                    passed,
                    f"Balance check: {value_cols[0]}={val1:.2f}, {value_cols[1]}={val2:.2f}, diff={diff:.2f}",
                    {"values": row, "difference": diff, "tolerance": tolerance},
                )

            # Fall back to checking if difference column is within tolerance
            if "difference" in row or "diff" in row:
                diff = abs(float(row.get("difference", row.get("diff", 0)) or 0))
                passed = diff <= tolerance
                return (
                    passed,
                    f"Balance difference: {diff:.2f} (tolerance: {tolerance})",
                    {"difference": diff, "tolerance": tolerance},
                )

            return (True, "Balance check passed", {"row": row})

        elif check_type == "constraint":
            # Constraint checks return violating rows
            if row_count == 0:
                return (True, "No constraint violations found", {})
            return (
                False,
                f"Found {row_count} constraint violations",
                {"violation_count": row_count},
            )

        elif check_type == "comparison":
            # Comparison checks (e.g., Assets = Liabilities + Equity)
            if row_count == 0:
                return (False, "No results returned", {})

            row = result_rows[0]
            tolerance = params.get("tolerance", 0.01)

            # Check for an equation_holds or is_valid column
            if "equation_holds" in row:
                passed = bool(row["equation_holds"])
                return (passed, f"Equation check: {'passed' if passed else 'failed'}", row)

            if "is_valid" in row:
                passed = bool(row["is_valid"])
                return (passed, f"Comparison check: {'passed' if passed else 'failed'}", row)

            # Check for difference column
            if "difference" in row:
                diff = abs(float(row["difference"] or 0))
                passed = diff <= tolerance
                return (
                    passed,
                    f"Comparison difference: {diff:.2f}",
                    {"difference": diff},
                )

            return (True, "Comparison check completed", row)

        elif check_type == "aggregate":
            # Aggregate checks return summary values
            if row_count == 0:
                return (False, "No results returned", {})

            row = result_rows[0]
            return (True, "Aggregate check completed", row)

        else:
            # Custom check - assume passing if any results
            return (
                row_count > 0,
                f"Custom check returned {row_count} rows",
                {"row_count": row_count},
            )


__all__ = ["ValidationAgent"]
