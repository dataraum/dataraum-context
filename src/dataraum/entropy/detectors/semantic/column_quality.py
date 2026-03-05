"""Column quality entropy detector.

Measures uncertainty from LLM-generated column quality reports.
Converts ColumnQualityReport scores into entropy: score = 1.0 - avg_quality_score.

Produces one EntropyObject per column, using the same logic that was previously
inline in entropy_phase._run_dimensional_entropy().

Source: quality_summary analysis (ColumnQualityReport)
"""

from __future__ import annotations

from typing import Any

from dataraum.entropy.detectors.base import DetectorContext, EntropyDetector
from dataraum.entropy.models import EntropyObject, ResolutionOption


class ColumnQualityDetector(EntropyDetector):
    """Detect entropy from column quality reports.

    Table-scoped detector that produces one EntropyObject per column
    based on LLM quality assessments across slices.
    """

    detector_id = "column_quality"
    layer = "semantic"
    dimension = "dimensional"
    sub_dimension = "column_quality"
    scope = "table"
    required_analyses = ["column_quality_reports"]
    description = "Column quality entropy from LLM quality reports"

    def load_data(self, context: DetectorContext) -> None:
        """Load column quality reports for the table."""
        if context.session is None or not context.table_id or not context.table_name:
            return

        from dataraum.entropy.detectors.loaders import load_column_quality_reports

        data = load_column_quality_reports(
            context.session,
            context.table_id,
            context.table_name,
        )
        if data is not None:
            context.analysis_results["column_quality_reports"] = data

    def detect(self, context: DetectorContext) -> list[EntropyObject]:
        """Produce one EntropyObject per column from quality reports."""
        reports_data: dict[str, Any] = context.get_analysis("column_quality_reports", {})
        if not reports_data:
            return []

        objects: list[EntropyObject] = []
        for col_name, col_data in reports_data.items():
            avg_quality_score = col_data["avg_quality_score"]
            entropy_score = 1.0 - avg_quality_score

            evidence: list[dict[str, Any]] = [
                {
                    "source": "column_quality_report",
                    "column_id": col_data["column_id"],
                    "table_id": col_data["table_id"],
                    "slices_analyzed": col_data["slices_analyzed"],
                    "avg_quality_score": avg_quality_score,
                    "grades": col_data["grades"],
                    "key_findings": col_data["key_findings"],
                    "quality_issues_count": col_data["quality_issues_count"],
                    "recommendations_count": col_data["recommendations_count"],
                }
            ]

            resolution_options = [
                ResolutionOption(
                    action="investigate_quality_issues",
                    parameters={
                        "column_name": col_name,
                        "key_findings": col_data["key_findings"],
                        "quality_issues": col_data["quality_issues"],
                        "recommendations": col_data["recommendations"],
                    },
                    effort="medium",
                    description=(
                        f"Review {col_data['quality_issues_count']} quality issues "
                        f"and {col_data['recommendations_count']} recommendations for {col_name}"
                    ),
                ),
            ]

            # Use the effective table name for the target
            effective_table_name = col_data["table_name"]
            obj = EntropyObject(
                layer=self.layer,
                dimension=self.dimension,
                sub_dimension=self.sub_dimension,
                target=f"column:{effective_table_name}.{col_name}",
                score=entropy_score,
                evidence=evidence,
                resolution_options=resolution_options,
                detector_id=self.detector_id,
                source_analysis_ids=[],
            )
            objects.append(obj)

        return objects
