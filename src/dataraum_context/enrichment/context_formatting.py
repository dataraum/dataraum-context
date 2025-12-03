"""Context formatting for LLM consumption.

This module formats metadata from various pillars into structured, interpretable
context documents that LLMs can use to make data quality and analysis decisions.

Each formatter function takes raw metadata and produces:
- Natural language interpretations
- Actionable recommendations
- Structured evidence
- Severity assessments
"""

from typing import Any

from dataraum_context.profiling.models import (
    ColumnVIF,
    DependencyGroup,
    MulticollinearityAnalysis,
)


def _format_dependency_groups(
    dependency_groups: list[DependencyGroup], column_vifs: list[ColumnVIF]
) -> list[dict[str, Any]]:
    """Format dependency groups for LLM consumption.

    Args:
        dependency_groups: List of DependencyGroup objects
        column_vifs: List of ColumnVIF objects for column name lookup

    Returns:
        List of formatted dependency group dicts
    """
    # Create column ID to name mapping
    column_id_to_name = {vif.column_id: vif.column_ref.column_name for vif in column_vifs}

    formatted_groups = []
    for i, group in enumerate(dependency_groups, start=1):
        # Map column IDs to names
        column_names = [
            column_id_to_name.get(col_id, col_id) for col_id in group.involved_column_ids
        ]

        # Get VIF values for group members
        vif_values = {
            column_id_to_name.get(vif.column_id, vif.column_id): round(vif.vif, 2)
            for vif in column_vifs
            if vif.column_id in group.involved_column_ids
        }

        formatted_groups.append(
            {
                "group_id": i,
                "severity": group.severity,
                "condition_index": round(group.condition_index, 1),
                "columns": column_names,
                "vif_values": vif_values,
                "variance_proportions": [round(vdp, 3) for vdp in group.variance_proportions],
                "interpretation": group.interpretation,
                "recommendation": _get_dependency_group_recommendation(group, column_names),
            }
        )

    return formatted_groups


def _get_dependency_group_recommendation(group: DependencyGroup, column_names: list[str]) -> str:
    """Generate actionable recommendation for a dependency group.

    Args:
        group: DependencyGroup object
        column_names: Column names in the group

    Returns:
        Recommendation string
    """
    num_cols = len(column_names)

    if group.severity == "severe":
        if num_cols == 2:
            return (
                f"Remove one of {column_names[0]} or {column_names[1]}. "
                f"These columns are nearly identical or one is derived from the other."
            )
        else:
            col_list = ", ".join(column_names[:3])
            if num_cols > 3:
                col_list += f", and {num_cols - 3} others"
            return (
                f"Review {col_list}. These {num_cols} columns form a linear combination. "
                f"Consider consolidating or removing redundant columns."
            )
    else:  # moderate
        col_list = ", ".join(column_names)
        return (
            f"Investigate relationship between {col_list}. "
            f"These may represent related metrics that could be simplified."
        )


def format_multicollinearity_for_llm(
    analysis: MulticollinearityAnalysis,
) -> dict[str, Any]:
    """Format multicollinearity analysis for LLM consumption.

    Transforms VIF, Tolerance, and Condition Index metrics into structured
    context with natural language interpretations and actionable recommendations.

    Research-based thresholds:
    - VIF = 1: No multicollinearity
    - VIF 1-5: Low to moderate (acceptable)
    - VIF > 5: High, requires attention
    - VIF > 10: Serious, often problematic

    Args:
        analysis: MulticollinearityAnalysis from profiling

    Returns:
        Dict with multicollinearity context for LLM
    """
    # Generate overall interpretation
    interpretation = _get_multicollinearity_interpretation(analysis)

    # Format table-level assessment
    table_level = None
    if analysis.condition_index:
        table_level = {
            "condition_index": analysis.condition_index.condition_index,
            "severity": analysis.condition_index.severity,
            "assessment": analysis.condition_index.interpretation,
            "problematic_dimensions": analysis.condition_index.problematic_dimensions,
        }

        # Add dependency groups if available
        if analysis.condition_index.dependency_groups:
            table_level["dependency_groups"] = _format_dependency_groups(
                analysis.condition_index.dependency_groups, analysis.column_vifs
            )

    # Format problematic columns with detailed analysis
    problematic_columns = []
    for vif in analysis.column_vifs:
        if vif.vif > 4:  # Threshold for investigation
            column_info = {
                "column": vif.column_ref.column_name,
                "vif": round(vif.vif, 2),
                "tolerance": round(vif.tolerance, 3),
                "severity": _get_vif_severity_label(vif.vif),
                "interpretation": _get_vif_interpretation(vif.vif),
                "impact": _get_vif_impact(vif.vif),
            }

            # Add correlated columns if available
            if vif.correlated_with:
                column_info["highly_correlated_with"] = vif.correlated_with[:3]

            problematic_columns.append(column_info)

    # Generate recommendations
    recommendations = _generate_multicollinearity_recommendations(analysis)

    return {
        "multicollinearity_assessment": {
            "overall_severity": analysis.overall_severity,
            "summary": interpretation,
            "num_problematic_columns": analysis.num_problematic_columns,
            "table_level": table_level,
            "problematic_columns": problematic_columns,
            "recommendations": recommendations,
            "technical_details": {
                "total_columns_analyzed": len(analysis.column_vifs),
                "analysis_method": "VIF (Variance Inflation Factor) via OLS regression",
                "table_method": "Condition Index via eigenvalue decomposition",
            },
        }
    }


def _get_vif_severity_label(vif: float) -> str:
    """Get severity label based on research thresholds.

    Thresholds:
    - VIF = 1: No multicollinearity
    - VIF 1-5: Low to moderate
    - VIF 5-10: High
    - VIF > 10: Serious

    Args:
        vif: Variance Inflation Factor value

    Returns:
        Severity label
    """
    if vif <= 1.0:
        return "none"
    elif vif <= 5.0:
        return "low_to_moderate"
    elif vif <= 10.0:
        return "high"
    else:
        return "serious"


def _get_vif_interpretation(vif: float) -> str:
    """Get natural language interpretation of VIF value.

    Args:
        vif: Variance Inflation Factor value

    Returns:
        Natural language interpretation
    """
    if vif <= 1.0:
        return f"No multicollinearity detected (VIF={vif:.1f})"
    elif vif <= 5.0:
        return f"Low to moderate multicollinearity (VIF={vif:.1f}) - generally acceptable"
    elif vif <= 10.0:
        return f"High multicollinearity (VIF={vif:.1f}) - requires attention"
    else:
        return f"Serious multicollinearity (VIF={vif:.1f}) - often considered problematic"


def _get_vif_impact(vif: float) -> str:
    """Explain the practical impact of the VIF value.

    Args:
        vif: Variance Inflation Factor value

    Returns:
        Impact description
    """
    if vif <= 1.0:
        return "Column variance is not inflated by correlation with other columns"
    elif vif <= 5.0:
        variance_inflation = (vif - 1) * 100
        return f"Column variance inflated by ~{variance_inflation:.0f}% due to correlation with other columns"
    elif vif <= 10.0:
        return f"Column is {vif:.1f}x more variable than it would be if uncorrelated - may indicate redundancy"
    else:
        redundancy = (1 - 1 / vif) * 100
        return f"Column is ~{redundancy:.0f}% redundant with other columns - likely derived or duplicate data"


def _get_multicollinearity_interpretation(
    analysis: MulticollinearityAnalysis,
) -> str:
    """Generate natural language summary of multicollinearity analysis.

    Args:
        analysis: MulticollinearityAnalysis

    Returns:
        Natural language summary
    """
    if analysis.overall_severity == "none":
        return (
            "No significant multicollinearity detected. Columns are relatively "
            "independent and each provides unique information."
        )

    if analysis.overall_severity == "moderate":
        parts = []
        if analysis.num_problematic_columns > 0:
            parts.append(
                f"{analysis.num_problematic_columns} column(s) show elevated "
                f"correlation with other columns"
            )
        if analysis.condition_index and analysis.condition_index.has_multicollinearity:
            parts.append(
                f"Table-level analysis indicates moderate structural redundancy "
                f"(Condition Index: {analysis.condition_index.condition_index:.1f})"
            )

        summary = "Moderate multicollinearity: " + ". ".join(parts) + "."
        return (
            summary
            + " This may indicate some overlap in information but is not necessarily problematic."
        )

    # Severe
    severity_desc = []
    if analysis.num_problematic_columns > 0:
        severity_desc.append(
            f"{analysis.num_problematic_columns} columns are highly redundant with other columns"
        )

    if analysis.condition_index:
        severity_desc.append(
            f"Condition Index of {analysis.condition_index.condition_index:.1f} "
            f"indicates severe structural redundancy"
        )

    return (
        "Severe multicollinearity detected. "
        + ". ".join(severity_desc)
        + ". This suggests significant overlap in information, possibly from "
        "derived columns, duplicate data, or mathematically related fields. "
        "Consider data consolidation before analysis."
    )


def _generate_multicollinearity_recommendations(
    analysis: MulticollinearityAnalysis,
) -> list[str]:
    """Generate actionable recommendations based on multicollinearity analysis.

    Args:
        analysis: MulticollinearityAnalysis

    Returns:
        List of actionable recommendations
    """
    recommendations = []

    if analysis.overall_severity == "none":
        return ["No action needed - columns are sufficiently independent for analysis"]

    # Identify severe cases (VIF > 10)
    severe_cols = [vif for vif in analysis.column_vifs if vif.vif > 10]

    if severe_cols:
        col_names = [vif.column_ref.column_name for vif in severe_cols[:3]]
        if len(severe_cols) > 3:
            col_list = f"{', '.join(col_names)}, and {len(severe_cols) - 3} others"
        else:
            col_list = ", ".join(col_names)

        recommendations.append(
            f"🔴 Critical: Investigate {col_list}. These columns show serious "
            f"redundancy (VIF > 10). Consider removing or consolidating them."
        )

    # Identify high cases (VIF 5-10)
    high_cols = [vif for vif in analysis.column_vifs if 5 < vif.vif <= 10]

    if high_cols:
        recommendations.append(
            f"🟡 Attention: {len(high_cols)} column(s) have high multicollinearity "
            f"(VIF 5-10). Review if these represent the same underlying concept."
        )

    # Check for derived columns (very high VIF with multiple correlations)
    derived_candidates = [
        vif for vif in analysis.column_vifs if vif.vif > 15 and len(vif.correlated_with) >= 2
    ]

    if derived_candidates:
        recommendations.append(
            "💡 Some columns may be calculated from others (e.g., total = sum of parts). "
            "Use derived columns for validation only, not as independent variables."
        )

    # Dependency group recommendations (VDP-based - most specific)
    if analysis.condition_index and analysis.condition_index.dependency_groups:
        # Create column ID to name mapping
        column_id_to_name = {
            vif.column_id: vif.column_ref.column_name for vif in analysis.column_vifs
        }

        for i, group in enumerate(analysis.condition_index.dependency_groups, start=1):
            column_names = [
                column_id_to_name.get(col_id, col_id) for col_id in group.involved_column_ids
            ]
            col_list = ", ".join(column_names[:3])
            if len(column_names) > 3:
                col_list += f", and {len(column_names) - 3} others"

            if group.severity == "severe":
                recommendations.append(
                    f"🔴 Dependency Group {i}: {col_list} form a severe linear dependency "
                    f"(CI={group.condition_index:.1f}). Remove redundant columns from this group."
                )
            else:
                recommendations.append(
                    f"🟡 Dependency Group {i}: {col_list} show shared variance "
                    f"(CI={group.condition_index:.1f}). Review for potential simplification."
                )

    # Table-level severity (CI thresholds: <10=none, 10-30=moderate, >30=severe)
    if analysis.condition_index and analysis.condition_index.condition_index > 30:
        recommendations.append(
            "⚠️ Severe table-level multicollinearity (Condition Index > 30) suggests "
            "high degree of linear dependency that makes regression estimates unstable. "
            "Consider a comprehensive review of column selection before statistical analysis."
        )

    # Moderate table-level
    if analysis.condition_index and 10 <= analysis.condition_index.condition_index <= 30:
        recommendations.append(
            "ℹ️ Moderate table-level multicollinearity detected (CI 10-30). This warrants "
            "investigation but may not prevent analysis. Review column relationships carefully."
        )

    # General advice for moderate cases
    if analysis.overall_severity == "moderate" and not recommendations:
        recommendations.append(
            "Consider reviewing column relationships to ensure each provides unique "
            "information. Multicollinearity doesn't invalidate data but may affect "
            "certain types of analysis (e.g., regression coefficient interpretation)."
        )

    return recommendations
