"""Quality Issue Aggregation.

Aggregates quality issues from all pillars into unified format for context consumption.
This module focuses on collecting issues from various quality sources, NOT computing scores.

Architecture:
- Pillar 1 (Statistical): Benford, outliers (IQR + Isolation Forest), correlations
- Pillar 2 (Topological): Orphaned components, anomalous cycles, structural instability
- Pillar 4 (Temporal): Completeness gaps, stale data, distribution changes
- Pillar 5 (Domain): Domain-specific rule violations

The aggregation functions extract issues from stored quality metrics and convert them
to a unified QualitySynthesisIssue format with:
- Severity levels (CRITICAL, ERROR, WARNING, INFO)
- Dimension classification (for context, not scoring)
- Evidence and recommendations
"""

import logging
from typing import Any
from uuid import uuid4

from dataraum_context.quality.models import (
    QualityDimension,
    QualitySynthesisIssue,
    QualitySynthesisSeverity,
)
from dataraum_context.storage.models_v2.correlation_context import (
    ColumnCorrelation,
    FunctionalDependency,
)
from dataraum_context.storage.models_v2.domain_quality import DomainQualityMetrics
from dataraum_context.storage.models_v2.statistical_context import StatisticalQualityMetrics
from dataraum_context.storage.models_v2.temporal_context import TemporalQualityMetrics
from dataraum_context.storage.models_v2.topological_context import TopologicalQualityMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# Issue Aggregation Functions
# ============================================================================


def aggregate_statistical_issues(
    stat_quality: StatisticalQualityMetrics | None,
    column_id: str,
    column_name: str,
) -> list[QualitySynthesisIssue]:
    """Extract quality issues from statistical quality metrics.

    Args:
        stat_quality: Statistical quality metrics from DB
        column_id: Column ID
        column_name: Column name for display

    Returns:
        List of QualitySynthesisIssue for Benford violations, outliers, etc.
    """
    if not stat_quality or not stat_quality.quality_data:
        return []

    # Deserialize JSONB to Pydantic model
    from dataraum_context.profiling.models import StatisticalQualityResult

    try:
        quality_result = StatisticalQualityResult.model_validate(stat_quality.quality_data)
        issues_list = quality_result.quality_issues
    except Exception:
        # Fallback to dict access if deserialization fails
        issues_list = stat_quality.quality_data.get("quality_issues", [])

    if not issues_list:
        return []

    issues = []
    for quality_issue in issues_list:
        # Handle both dict and object formats
        if isinstance(quality_issue, dict):
            issue_type = quality_issue.get("issue_type", "unknown")
            severity_str = quality_issue.get("severity", "warning")
            description = quality_issue.get("description", "Statistical quality issue")
            evidence = quality_issue.get("evidence", {})
        else:
            issue_type = getattr(quality_issue, "issue_type", "unknown")
            severity_str = getattr(quality_issue, "severity", "warning")
            description = getattr(quality_issue, "description", "Statistical quality issue")
            evidence = getattr(quality_issue, "evidence", {})

        # Map severity
        severity_map = {
            "critical": QualitySynthesisSeverity.CRITICAL,
            "error": QualitySynthesisSeverity.ERROR,
            "warning": QualitySynthesisSeverity.WARNING,
            "info": QualitySynthesisSeverity.INFO,
        }
        severity = severity_map.get(severity_str, QualitySynthesisSeverity.WARNING)

        # Map to dimension (for context classification, not scoring)
        dimension_map = {
            "benford_violation": QualityDimension.ACCURACY,
            "outliers": QualityDimension.VALIDITY,
            "outliers_iqr": QualityDimension.VALIDITY,
            "outliers_isolation_forest": QualityDimension.VALIDITY,
            "distribution_shift": QualityDimension.CONSISTENCY,
            "multicollinearity": QualityDimension.CONSISTENCY,
        }
        dimension = dimension_map.get(issue_type, QualityDimension.VALIDITY)

        issue = QualitySynthesisIssue(
            issue_id=str(uuid4()),
            issue_type=issue_type,
            severity=severity,
            dimension=dimension,
            table_id=None,
            column_id=column_id,
            column_name=column_name,
            description=description,
            recommendation=None,
            evidence=evidence,
            source_pillar=1,  # Statistical
            source_module="statistical_quality",
            detected_at=stat_quality.computed_at,
        )
        issues.append(issue)

    return issues


def aggregate_temporal_issues(
    temp_quality: TemporalQualityMetrics | None,
    column_id: str,
    column_name: str,
) -> list[QualitySynthesisIssue]:
    """Extract quality issues from temporal quality metrics.

    Args:
        temp_quality: Temporal quality metrics from DB
        column_id: Column ID
        column_name: Column name for display

    Returns:
        List of QualitySynthesisIssue for gaps, staleness, distribution changes
    """
    if not temp_quality or not temp_quality.temporal_data:
        return []

    # Deserialize JSONB to Pydantic model
    from dataraum_context.quality.models import TemporalQualityResult

    try:
        quality_result = TemporalQualityResult.model_validate(temp_quality.temporal_data)
        issues_list = quality_result.quality_issues
    except Exception:
        # Fallback to dict access if deserialization fails
        issues_list = temp_quality.temporal_data.get("quality_issues", [])

    if not issues_list:
        return []

    issues = []
    for quality_issue in issues_list:
        # Handle both QualityIssue object and dict
        from dataraum_context.quality.models import QualityIssue as QualityIssueModel

        if isinstance(quality_issue, QualityIssueModel):
            issue_type = quality_issue.issue_type
            severity_str = quality_issue.severity
            description = quality_issue.description
            evidence = quality_issue.evidence
        else:
            issue_dict: dict[str, Any] = quality_issue  # type: ignore[assignment]
            issue_type = issue_dict.get("issue_type", "unknown")
            severity_str = issue_dict.get("severity", "warning")
            description = issue_dict.get("description", "Temporal quality issue")
            evidence = issue_dict.get("evidence", {})

        severity_map = {
            "critical": QualitySynthesisSeverity.CRITICAL,
            "error": QualitySynthesisSeverity.ERROR,
            "warning": QualitySynthesisSeverity.WARNING,
            "info": QualitySynthesisSeverity.INFO,
        }
        severity = severity_map.get(severity_str, QualitySynthesisSeverity.WARNING)

        # Map to dimension
        dimension_map = {
            "low_completeness": QualityDimension.COMPLETENESS,
            "large_gap": QualityDimension.COMPLETENESS,
            "stale_data": QualityDimension.TIMELINESS,
            "many_change_points": QualityDimension.CONSISTENCY,
            "unstable_distribution": QualityDimension.CONSISTENCY,
        }
        dimension = dimension_map.get(issue_type, QualityDimension.COMPLETENESS)

        issue = QualitySynthesisIssue(
            issue_id=str(uuid4()),
            issue_type=issue_type,
            severity=severity,
            dimension=dimension,
            table_id=None,
            column_id=column_id,
            column_name=column_name,
            description=description,
            recommendation=None,
            evidence=evidence,
            source_pillar=4,  # Temporal
            source_module="temporal_quality",
            detected_at=temp_quality.computed_at,
        )
        issues.append(issue)

    return issues


def aggregate_topological_issues(
    topo_quality: TopologicalQualityMetrics | None,
    table_id: str,
    table_name: str,
) -> list[QualitySynthesisIssue]:
    """Extract quality issues from topological quality metrics (table-level).

    Args:
        topo_quality: Topological quality metrics from DB
        table_id: Table ID
        table_name: Table name for display

    Returns:
        List of QualitySynthesisIssue for orphaned components, anomalous cycles, etc.
    """
    if not topo_quality:
        return []

    issues = []

    # Check for orphaned components
    if topo_quality.orphaned_components and topo_quality.orphaned_components > 0:
        issues.append(
            QualitySynthesisIssue(
                issue_id=str(uuid4()),
                issue_type="orphaned_components",
                severity=QualitySynthesisSeverity.WARNING,
                dimension=QualityDimension.CONSISTENCY,
                table_id=table_id,
                column_id=None,
                column_name=None,
                description=(
                    f"{topo_quality.orphaned_components} disconnected structural "
                    "components detected"
                ),
                recommendation="Investigate structural relationships and data integrity",
                evidence={"orphaned_count": topo_quality.orphaned_components},
                source_pillar=2,  # Topological
                source_module="topological_quality",
                detected_at=topo_quality.computed_at,
            )
        )

    # Check for anomalous cycles (deserialize from JSONB)
    if topo_quality.topology_data:
        try:
            from dataraum_context.quality.models import TopologicalQualityResult

            topology_result = TopologicalQualityResult.model_validate(topo_quality.topology_data)
            anomalous_cycles = topology_result.anomalous_cycles
        except Exception:
            # Fallback to dict access
            anomalous_cycles = topo_quality.topology_data.get("anomalous_cycles", [])

        if anomalous_cycles and len(anomalous_cycles) > 0:
            cycle_count = len(anomalous_cycles)
            issues.append(
                QualitySynthesisIssue(
                    issue_id=str(uuid4()),
                    issue_type="anomalous_cycles",
                    severity=QualitySynthesisSeverity.WARNING,
                    dimension=QualityDimension.CONSISTENCY,
                    table_id=table_id,
                    column_id=None,
                    column_name=None,
                    description=f"{cycle_count} anomalous relationship cycles detected",
                    recommendation="Review circular relationships for logical correctness",
                    evidence={"cycle_count": cycle_count},
                    source_pillar=2,  # Topological
                    source_module="topological_quality",
                    detected_at=topo_quality.computed_at,
                )
            )

    # Check for homological instability
    if topo_quality.homologically_stable is False:
        bottleneck_distance = None
        if topo_quality.topology_data:
            bottleneck_distance = topo_quality.topology_data.get("bottleneck_distance")

        issues.append(
            QualitySynthesisIssue(
                issue_id=str(uuid4()),
                issue_type="structural_instability",
                severity=QualitySynthesisSeverity.INFO,
                dimension=QualityDimension.CONSISTENCY,
                table_id=table_id,
                column_id=None,
                column_name=None,
                description="Structural topology has changed significantly from baseline",
                recommendation="Review for data schema changes or relationship drift",
                evidence={"bottleneck_distance": bottleneck_distance},
                source_pillar=2,  # Topological
                source_module="topological_quality",
                detected_at=topo_quality.computed_at,
            )
        )

    return issues


def aggregate_correlation_issues(
    column_id: str,
    column_name: str,
    high_correlations: list[ColumnCorrelation],
    fd_violations: list[FunctionalDependency],
) -> list[QualitySynthesisIssue]:
    """Extract quality issues from correlation metrics.

    Args:
        column_id: Column ID
        column_name: Column name for display
        high_correlations: List of high correlation records (>0.9)
        fd_violations: List of functional dependency violations

    Returns:
        List of QualitySynthesisIssue for correlations and FD violations
    """
    issues = []

    # High correlations indicate potential redundancy
    if high_correlations:
        for corr in high_correlations[:3]:  # Top 3
            other_col_id = corr.column2_id if corr.column1_id == column_id else corr.column1_id
            corr_value = corr.pearson_r if corr.pearson_r else corr.spearman_rho
            issues.append(
                QualitySynthesisIssue(
                    issue_id=str(uuid4()),
                    issue_type="high_correlation",
                    severity=QualitySynthesisSeverity.INFO,
                    dimension=QualityDimension.CONSISTENCY,
                    table_id=corr.table_id,
                    column_id=column_id,
                    column_name=column_name,
                    description=f"Very high correlation ({corr_value:.2f}) with another column",
                    recommendation="Consider if one column is derived or redundant",
                    evidence={"other_column_id": other_col_id, "correlation": corr_value},
                    source_pillar=1,  # Statistical
                    source_module="correlation",
                    detected_at=corr.computed_at,
                )
            )

    # Functional dependency violations
    if fd_violations:
        for fd in fd_violations[:3]:  # Top 3
            issues.append(
                QualitySynthesisIssue(
                    issue_id=str(uuid4()),
                    issue_type="fd_violation",
                    severity=QualitySynthesisSeverity.WARNING,
                    dimension=QualityDimension.CONSISTENCY,
                    table_id=fd.table_id,
                    column_id=column_id,
                    column_name=column_name,
                    description=f"Functional dependency violated {fd.violation_count} times",
                    recommendation="Investigate data integrity and normalization",
                    evidence={
                        "violation_count": fd.violation_count,
                        "confidence": fd.confidence,
                    },
                    source_pillar=1,  # Statistical
                    source_module="correlation",
                    detected_at=fd.computed_at,
                )
            )

    return issues


def aggregate_domain_issues(
    domain_quality: DomainQualityMetrics | None,
    table_id: str | None = None,
    column_id: str | None = None,
    column_name: str | None = None,
) -> list[QualitySynthesisIssue]:
    """Extract quality issues from domain quality metrics.

    Domain quality is typically table-level, so column_id/column_name may be None.

    Args:
        domain_quality: Domain quality metrics from DB
        table_id: Table ID (optional, extracted from domain_quality if not provided)
        column_id: Column ID (optional)
        column_name: Column name (optional)

    Returns:
        List of QualitySynthesisIssue for domain rule violations
    """
    if not domain_quality or not domain_quality.violations:
        return []

    # Handle both list and dict formats
    violations_list = domain_quality.violations
    if isinstance(violations_list, dict):
        violations_list = violations_list.get("violations", [])

    # Get table_id from domain_quality if not provided
    effective_table_id = table_id or str(domain_quality.table_id)

    issues = []
    for violation in violations_list[:5]:  # Top 5
        severity_map = {
            "critical": QualitySynthesisSeverity.CRITICAL,
            "high": QualitySynthesisSeverity.ERROR,
            "medium": QualitySynthesisSeverity.WARNING,
            "low": QualitySynthesisSeverity.INFO,
        }
        severity = severity_map.get(
            violation.get("severity", "medium"), QualitySynthesisSeverity.WARNING
        )

        issue = QualitySynthesisIssue(
            issue_id=str(uuid4()),
            issue_type="domain_rule_violation",
            severity=severity,
            dimension=QualityDimension.ACCURACY,
            table_id=effective_table_id,
            column_id=column_id,
            column_name=column_name,
            description=violation.get("description", "Domain rule violation"),
            recommendation=violation.get("recommendation"),
            evidence=violation.get("evidence", {}),
            source_pillar=5,  # Quality (domain quality)
            source_module="domain_quality",
            detected_at=domain_quality.computed_at,
        )
        issues.append(issue)

    return issues


# ============================================================================
# Public API - Convenience functions for issue collection
# ============================================================================

# Re-export with cleaner names (keep underscore versions for backward compatibility)
_aggregate_statistical_issues = aggregate_statistical_issues
_aggregate_temporal_issues = aggregate_temporal_issues
_aggregate_topological_issues = aggregate_topological_issues
_aggregate_correlation_issues = aggregate_correlation_issues
_aggregate_domain_quality_issues = aggregate_domain_issues
