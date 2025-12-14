"""LLM Feature: Filter Recommendations.

Generates SQL WHERE clause recommendations based on quality issues.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.llm.features._base import LLMFeature
from dataraum_context.quality.models import DatasetQualityContext

logger = logging.getLogger(__name__)


class FilterRecommendation(BaseModel):
    """A single filter recommendation."""

    columns: list[str] = Field(description="Columns affected by this filter")
    where_clause: str = Field(description="SQL WHERE clause")
    reason: str = Field(description="Why this filter helps")
    priority: str = Field(default="medium", description="Priority: high, medium, low")


class FilterRecommendationsResult(BaseModel):
    """Result of filter recommendations generation."""

    filters: list[FilterRecommendation] = Field(default_factory=list)
    model: str | None = None


class FilterRecommendationsFeature(LLMFeature):
    """Generate SQL filter recommendations from quality issues."""

    async def generate(
        self,
        session: AsyncSession,
        quality_context: DatasetQualityContext,
        table_ids: list[str] | None = None,
    ) -> Result[FilterRecommendationsResult]:
        """Generate filter recommendations using LLM.

        Args:
            session: Database session for caching
            quality_context: Quality context with issues and metrics
            table_ids: Optional table IDs for cache key

        Returns:
            Result containing FilterRecommendationsResult
        """
        # Check if feature is enabled
        feature_config = self.config.features.filter_recommendations
        if not feature_config or not feature_config.enabled:
            logger.info("Filter recommendations feature is disabled")
            return Result.ok(FilterRecommendationsResult(filters=[]))

        # Build context for prompt
        context_json = self._build_context_json(quality_context)

        # Render prompt
        prompt, temperature = self.renderer.render(
            "filter_recommendations",
            context={"quality_context_json": context_json},
        )

        # Call LLM
        model_tier = feature_config.model_tier or "fast"
        result = await self._call_llm(
            session=session,
            feature_name="filter_recommendations",
            prompt=prompt,
            temperature=temperature,
            model_tier=model_tier,
            table_ids=table_ids,
        )

        if not result.success or not result.value:
            logger.error(f"LLM call failed: {result.error}")
            return Result.fail(result.error or "LLM call failed")

        # Parse response
        try:
            response_data = json.loads(result.value.content)
            filters = []

            for f in response_data.get("filters", []):
                filters.append(
                    FilterRecommendation(
                        columns=f.get("columns", []),
                        where_clause=f.get("where_clause", ""),
                        reason=f.get("reason", ""),
                        priority=f.get("priority", "medium"),
                    )
                )

            return Result.ok(
                FilterRecommendationsResult(
                    filters=filters,
                    model=result.value.model,
                )
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return Result.fail(f"Failed to parse response: {e}")
        except Exception as e:
            logger.error(f"Error processing filter recommendations: {e}")
            return Result.fail(f"Error processing recommendations: {e}")

    def _build_context_json(self, quality_context: DatasetQualityContext) -> str:
        """Build JSON context for the LLM prompt."""
        # Create a simplified view of the quality context
        tables_list: list[dict[str, Any]] = []
        cross_table_list: list[dict[str, Any]] = []
        context: dict[str, Any] = {
            "tables": tables_list,
            "cross_table_issues": cross_table_list,
            "summary": {
                "total_tables": quality_context.total_tables,
                "total_columns": quality_context.total_columns,
                "total_issues": quality_context.total_issues,
                "critical_issues": quality_context.critical_issue_count,
                "issues_by_severity": quality_context.issues_by_severity,
            },
        }

        # Add table summaries with issues
        for table in quality_context.tables:
            table_info: dict[str, Any] = {
                "table_name": table.table_name,
                "row_count": table.row_count,
                "flags": table.flags,
                "columns": [],
            }

            # Add columns with issues
            for col in table.columns:
                if col.issues or col.flags:
                    col_info: dict[str, Any] = {
                        "column_name": col.column_name,
                        "null_ratio": col.null_ratio,
                        "outlier_ratio": col.outlier_ratio,
                        "flags": col.flags,
                        "issues": [
                            {
                                "type": issue.issue_type,
                                "severity": issue.severity.value,
                                "description": issue.description,
                            }
                            for issue in col.issues[:3]  # Top 3 issues
                        ],
                    }
                    table_info["columns"].append(col_info)

            # Add table-level issues
            if table.issues:
                table_info["table_issues"] = [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity.value,
                        "description": issue.description,
                    }
                    for issue in table.issues[:3]
                ]

            tables_list.append(table_info)

        # Add cross-table issues
        for issue in quality_context.cross_table_issues:
            cross_table_list.append(
                {
                    "type": issue.issue_type,
                    "severity": issue.severity.value,
                    "description": issue.description,
                }
            )

        return json.dumps(context, indent=2, default=str)
