"""LLM Filtering Agent - The Bridge (Phase 8).

This module bridges System 2 (quality measurement) and System 1 (data filtering).

Flow:
1. Query quality metrics from DuckDB views (System 2)
2. Analyze metrics with LLM → generate filtering SQL
3. Return FilteringRecommendations for merger with user rules (Phase 9)

Key Design:
- LLM analyzes ALL quality pillars (statistical, topological, temporal, domain)
- Generates executable DuckDB SQL WHERE clauses
- Provides clear rationale for each recommendation
- User rules (Phase 9) can override/extend recommendations
"""

import json
import logging
from typing import Any

import duckdb
import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.llm import LLMService
from dataraum_context.llm.providers.base import LLMRequest
from dataraum_context.quality.filtering.models import FilteringRecommendations
from dataraum_context.storage.models_v2.core import Table

logger = logging.getLogger(__name__)

# Cache for loaded prompts
_PROMPTS_CACHE: dict[str, Any] | None = None


def _load_prompts() -> dict[str, Any]:
    """Load filtering analysis prompts from YAML config.

    Returns:
        Dictionary of prompt templates

    Raises:
        FileNotFoundError: If prompts file not found
        yaml.YAMLError: If prompts file is invalid
    """
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    from pathlib import Path

    # Try common locations
    prompt_paths = [
        Path("config/prompts/filtering_analysis.yaml"),
        Path.cwd() / "config/prompts/filtering_analysis.yaml",
    ]

    for prompt_path in prompt_paths:
        if prompt_path.exists():
            with open(prompt_path) as f:
                _PROMPTS_CACHE = yaml.safe_load(f)
                return _PROMPTS_CACHE

    raise FileNotFoundError("filtering_analysis.yaml prompts file not found")


async def analyze_quality_for_filtering(
    table_id: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    llm_service: LLMService | None,
) -> Result[FilteringRecommendations]:
    """Analyze quality metrics and generate filtering recommendations.

    This is the LLM bridge between System 2 (measurement) and System 1 (filtering).

    Args:
        table_id: Table to analyze
        duckdb_conn: DuckDB connection (with metadata attached)
        session: SQLAlchemy session for metadata queries
        llm_service: LLM service (optional - returns empty recommendations if None)

    Returns:
        Result containing FilteringRecommendations:
        - clean_view_filters: SQL WHERE clauses for clean data
        - quarantine_criteria: SQL conditions for problematic rows
        - column_exclusions: Columns to exclude from default queries
        - rationale: Explanation for each recommendation

    Example:
        >>> recs = await analyze_quality_for_filtering(table_id, duckdb_conn, session, llm_service)
        >>> if recs.success:
        ...     for filter in recs.value.clean_view_filters:
        ...         print(f"Filter: {filter}")
    """
    try:
        # Get table metadata
        from sqlalchemy import select

        stmt = select(Table).where(Table.table_id == table_id)
        result = await session.execute(stmt)
        table = result.scalar_one_or_none()

        if not table:
            return Result.fail(f"Table {table_id} not found")

        # If no LLM, return empty recommendations
        if llm_service is None:
            logger.warning("LLM service unavailable, returning empty recommendations")
            return Result.ok(
                FilteringRecommendations(
                    clean_view_filters=[],
                    quarantine_criteria=[],
                    column_exclusions=[],
                    rationale={},
                )
            )

        # Query column quality metrics from DuckDB view
        column_query = f"""
            SELECT
                column_name,
                resolved_type,
                null_ratio,
                cardinality_ratio,
                statistical_quality_score,
                benford_compliant,
                iqr_outlier_ratio,
                temporal_completeness,
                has_seasonality,
                is_stale,
                vif,
                has_multicollinearity,
                semantic_role
            FROM column_quality_assessment
            WHERE table_id = '{table_id}'
            ORDER BY column_name
        """

        try:
            column_metrics_df = duckdb_conn.execute(column_query).fetchdf()
            column_metrics = column_metrics_df.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Failed to query column quality view: {e}")
            column_metrics = []

        # Query table quality metrics from DuckDB view
        table_query = f"""
            SELECT
                avg_null_ratio,
                avg_statistical_quality,
                benford_violations,
                avg_outlier_ratio,
                stale_columns,
                multicollinear_columns,
                avg_vif,
                cycles_count,
                has_cycles,
                overall_quality_score,
                total_issues,
                critical_issues
            FROM table_quality_assessment
            WHERE table_id = '{table_id}'
        """

        try:
            table_metrics_df = duckdb_conn.execute(table_query).fetchdf()
            table_metrics = (
                table_metrics_df.to_dict(orient="records")[0] if len(table_metrics_df) > 0 else {}
            )
        except Exception as e:
            logger.warning(f"Failed to query table quality view: {e}")
            table_metrics = {}

        # Build problematic columns summary
        problematic_columns = []
        for col in column_metrics:
            issues = []
            if col.get("null_ratio", 0) > 0.3:
                issues.append(f"High nulls: {col['null_ratio']:.1%}")
            if col.get("benford_compliant") is False:
                issues.append("Benford violation")
            if col.get("iqr_outlier_ratio", 0) > 0.1:
                issues.append(f"Outliers: {col['iqr_outlier_ratio']:.1%}")
            if col.get("is_stale") is True:
                issues.append("Stale data")
            if col.get("has_multicollinearity") is True:
                issues.append(f"Multicollinear (VIF={col.get('vif', 0):.1f})")

            if issues:
                problematic_columns.append(f"- {col['column_name']}: {', '.join(issues)}")

        # Load prompts
        try:
            prompts = _load_prompts()
            filtering_prompt = prompts.get("filtering_analysis", {})
        except Exception as e:
            logger.warning(f"Failed to load prompts: {e}")
            return Result.ok(
                FilteringRecommendations(
                    clean_view_filters=[],
                    quarantine_criteria=[],
                    column_exclusions=[],
                    rationale={"error": f"Prompts unavailable: {e}"},
                )
            )

        # Format context for LLM
        system_prompt = filtering_prompt.get("system", "")
        user_template = filtering_prompt.get("user", "")

        user_prompt = user_template.format(
            table_name=table.table_name,
            row_count=table.row_count or 0,
            column_count=len(column_metrics),
            overall_quality_score=table_metrics.get("overall_quality_score", 1.0) or 1.0,
            critical_issues=table_metrics.get("critical_issues", 0) or 0,
            total_issues=table_metrics.get("total_issues", 0) or 0,
            benford_violations=table_metrics.get("benford_violations", 0) or 0,
            multicollinear_columns=table_metrics.get("multicollinear_columns", 0) or 0,
            stale_columns=table_metrics.get("stale_columns", 0) or 0,
            has_cycles=table_metrics.get("has_cycles", False) or False,
            column_metrics_json=json.dumps(column_metrics, indent=2, default=str),
            problematic_columns_summary=(
                "\n".join(problematic_columns)
                if problematic_columns
                else "No significant issues detected"
            ),
        )

        # Call LLM
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        request = LLMRequest(
            prompt=full_prompt,
            max_tokens=1500,  # Filtering recommendations need space
            temperature=0.3,  # Lower temperature for precise SQL generation
            response_format="json",
        )

        response_result = await llm_service.provider.complete(request)

        if not response_result.success or not response_result.value:
            logger.error(f"LLM request failed: {response_result.error}")
            return Result.ok(
                FilteringRecommendations(
                    clean_view_filters=[],
                    quarantine_criteria=[],
                    column_exclusions=[],
                    rationale={"error": "LLM request failed"},
                )
            )

        # Parse JSON response
        response_text = response_result.value.content.strip()
        try:
            recommendations_dict = json.loads(response_text)

            # Validate structure
            recommendations = FilteringRecommendations(
                clean_view_filters=recommendations_dict.get("clean_view_filters", []),
                quarantine_criteria=recommendations_dict.get("quarantine_criteria", []),
                column_exclusions=recommendations_dict.get("column_exclusions", []),
                rationale=recommendations_dict.get("rationale", {}),
            )

            logger.info(
                f"Generated {len(recommendations.clean_view_filters)} clean filters, "
                f"{len(recommendations.quarantine_criteria)} quarantine criteria for {table.table_name}"
            )

            return Result.ok(recommendations)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return Result.fail(f"Invalid JSON response from LLM: {e}")

    except Exception as e:
        logger.error(f"Failed to analyze quality for filtering: {e}")
        return Result.fail(f"Filtering analysis failed: {e}")
