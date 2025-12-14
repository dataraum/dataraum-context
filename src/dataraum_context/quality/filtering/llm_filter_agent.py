"""LLM Filtering Agent - The Bridge (Phase 8).

This module bridges quality context (System 2) and data filtering (System 1).

Flow:
1. Format quality context from metadata tables
2. Analyze context with LLM → generate filtering SQL
3. Return FilteringRecommendations for merger with user rules (Phase 9)

Key Design:
- LLM analyzes quality context from all pillars (statistical, topological, temporal, domain)
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
from dataraum_context.quality.context import format_table_quality_context
from dataraum_context.quality.filtering.models import FilteringRecommendations

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
    """Analyze quality context and generate filtering recommendations.

    This is the LLM bridge between quality context and data filtering.

    Args:
        table_id: Table to analyze
        duckdb_conn: DuckDB connection (for row counts)
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
        # Get quality context using the new context formatter
        table_context = await format_table_quality_context(table_id, session, duckdb_conn)

        if not table_context:
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

        # Build column metrics from context
        column_metrics = []
        for col in table_context.columns:
            col_info = {
                "column_name": col.column_name,
                "null_ratio": col.null_ratio,
                "cardinality_ratio": col.cardinality_ratio,
                "outlier_ratio": col.outlier_ratio,
                "parse_success_rate": col.parse_success_rate,
                "benford_compliant": col.benford_compliant,
                "is_stale": col.is_stale,
                "data_freshness_days": col.data_freshness_days,
                "has_seasonality": col.has_seasonality,
                "has_trend": col.has_trend,
                "flags": col.flags,
                "issue_count": len(col.issues),
            }
            column_metrics.append(col_info)

        # Count issues by type
        all_issues = table_context.issues + [
            issue for col in table_context.columns for issue in col.issues
        ]
        critical_issues = sum(1 for i in all_issues if i.severity.value == "critical")
        total_issues = len(all_issues)

        # Count specific issue types
        benford_violations = sum(
            1 for col in table_context.columns if col.benford_compliant is False
        )
        stale_columns = sum(1 for col in table_context.columns if col.is_stale is True)

        # Build problematic columns summary from context
        problematic_columns = []
        for col in table_context.columns:
            if col.flags or col.issues:
                issue_parts = []
                if col.null_ratio and col.null_ratio > 0.3:
                    issue_parts.append(f"High nulls: {col.null_ratio:.1%}")
                if col.benford_compliant is False:
                    issue_parts.append("Benford violation")
                if col.outlier_ratio and col.outlier_ratio > 0.1:
                    issue_parts.append(f"Outliers: {col.outlier_ratio:.1%}")
                if col.is_stale is True:
                    issue_parts.append("Stale data")
                if "high_nulls" in col.flags:
                    if not any("null" in p.lower() for p in issue_parts):
                        issue_parts.append("High nulls (flagged)")
                if "high_outliers" in col.flags:
                    if not any("outlier" in p.lower() for p in issue_parts):
                        issue_parts.append("High outliers (flagged)")

                if issue_parts:
                    problematic_columns.append(f"- {col.column_name}: {', '.join(issue_parts)}")

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
            table_name=table_context.table_name,
            row_count=table_context.row_count or 0,
            column_count=table_context.column_count,
            critical_issues=critical_issues,
            total_issues=total_issues,
            benford_violations=benford_violations,
            stale_columns=stale_columns,
            has_cycles=table_context.betti_1 is not None and table_context.betti_1 > 0,
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
                f"{len(recommendations.quarantine_criteria)} quarantine criteria "
                f"for {table_context.table_name}"
            )

            return Result.ok(recommendations)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return Result.fail(f"Invalid JSON response from LLM: {e}")

    except Exception as e:
        logger.error(f"Failed to analyze quality for filtering: {e}")
        return Result.fail(f"Filtering analysis failed: {e}")
