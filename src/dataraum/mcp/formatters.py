"""Formatters for LLM-optimized output.

These format data for consumption by LLMs via MCP tools.
Focus on clarity and structured information.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataraum.entropy.contracts import ContractEvaluation, ContractProfile
    from dataraum.query.models import QueryResult


def format_context_for_llm(source_name: str, context_text: str) -> str:
    """Format context document for LLM consumption."""
    return f"""# Data Context: {source_name}

{context_text}

---
Use this context to understand the data schema, relationships, and quality characteristics
when answering questions or generating queries.
"""


def format_entropy_summary(
    source_name: str,
    snapshot: Any,
    interpretations: Sequence[Any],
    table_filter: str | None = None,
) -> str:
    """Format entropy summary for LLM consumption."""
    lines = [f"# Entropy Summary: {source_name}"]

    if table_filter:
        lines.append(f"Filtered to table: {table_filter}")

    lines.append("")
    lines.append(f"## Overall Status: {snapshot.overall_readiness.upper()}")
    lines.append(f"Composite Score: {snapshot.avg_composite_score:.3f}")
    lines.append("")

    # Dimension breakdown
    lines.append("## Entropy by Dimension")
    lines.append(f"- Structural: {snapshot.avg_structural_entropy:.3f}")
    lines.append(f"- Semantic: {snapshot.avg_semantic_entropy:.3f}")
    lines.append(f"- Value: {snapshot.avg_value_entropy:.3f}")
    lines.append(f"- Computational: {snapshot.avg_computational_entropy:.3f}")
    lines.append("")

    # Issue counts
    high_entropy = [i for i in interpretations if i.composite_score > 0.2]
    investigate = [i for i in interpretations if i.readiness == "investigate"]
    blocked = [i for i in interpretations if i.readiness == "blocked"]

    lines.append("## Issue Summary")
    lines.append(f"- Total columns analyzed: {len(interpretations)}")
    lines.append(f"- High entropy (>0.2): {len(high_entropy)}")
    lines.append(f"- Needs investigation: {len(investigate)}")
    lines.append(f"- Blocked: {len(blocked)}")
    lines.append("")

    # Top issues
    if interpretations:
        lines.append("## Top High-Entropy Columns")
        for interp in interpretations[:10]:
            status_icon = {"ready": "✓", "investigate": "⚠", "blocked": "✗"}.get(
                interp.readiness, "?"
            )
            lines.append(
                f"- [{status_icon}] {interp.table_name}.{interp.column_name}: "
                f"{interp.composite_score:.3f}"
            )
            if interp.explanation:
                explanation = interp.explanation.split(".")[0]
                lines.append(f"  {explanation}")

    return "\n".join(lines)


def format_contract_evaluation(
    evaluation: ContractEvaluation,
    profile: ContractProfile,
) -> str:
    """Format contract evaluation for LLM consumption."""
    lines = [f"# Contract Evaluation: {profile.display_name}"]
    lines.append("")

    # Status
    status = "PASS" if evaluation.is_compliant else "FAIL"
    lines.append(f"## Status: {status} ({evaluation.confidence_level.label})")
    lines.append(
        f"Overall Score: {evaluation.overall_score:.2f} (threshold: {profile.overall_threshold})"
    )
    lines.append("")

    # Description
    lines.append(f"**Description:** {profile.description}")
    lines.append("")

    # Dimension scores
    lines.append("## Dimension Scores")
    for dim, threshold in profile.dimension_thresholds.items():
        score = evaluation.dimension_scores.get(dim, 0.0)
        status_icon = "✓" if score <= threshold else "✗"
        lines.append(f"- {dim}: {score:.2f} / {threshold:.2f} [{status_icon}]")
    lines.append("")

    # Violations
    if evaluation.violations:
        lines.append("## Violations")
        for v in evaluation.violations:
            if v.dimension:
                lines.append(f"- {v.dimension}: {v.actual:.2f} exceeds {v.max_allowed:.2f}")
                if v.affected_columns:
                    cols = ", ".join(v.affected_columns[:5])
                    lines.append(f"  Affected: {cols}")
            elif v.details:
                lines.append(f"- {v.details}")
        lines.append("")

    # Warnings
    if evaluation.warnings:
        lines.append("## Warnings")
        for w in evaluation.warnings:
            lines.append(f"- {w.details}")
        lines.append("")

    # Path to compliance
    if not evaluation.is_compliant and evaluation.worst_dimension:
        lines.append("## Path to Compliance")
        lines.append(
            f"Focus on: {evaluation.worst_dimension} (score: {evaluation.worst_dimension_score:.2f})"
        )
        lines.append(f"Estimated effort: {evaluation.estimated_effort_to_comply}")

    return "\n".join(lines)


def format_query_result(result: QueryResult) -> str:
    """Format query result for LLM consumption."""
    lines = ["# Query Result"]
    lines.append("")

    # Confidence
    lines.append(f"## Confidence: {result.confidence_level.label} {result.confidence_level.emoji}")
    if result.contract:
        lines.append(f"Contract: {result.contract}")
    lines.append("")

    # Answer
    lines.append("## Answer")
    lines.append(result.answer)
    lines.append("")

    # Data
    if result.data and result.columns:
        lines.append("## Data")
        lines.append(f"Columns: {', '.join(result.columns)}")
        lines.append(f"Rows: {len(result.data)}")
        lines.append("")

        # Show first few rows as markdown table
        if len(result.data) <= 20:
            # Header
            lines.append("| " + " | ".join(result.columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(result.columns)) + " |")
            # Rows
            for row in result.data[:20]:
                values = [str(row.get(c, ""))[:30] for c in result.columns]
                lines.append("| " + " | ".join(values) + " |")
        else:
            lines.append(f"(Showing first 5 of {len(result.data)} rows)")
            lines.append("| " + " | ".join(result.columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(result.columns)) + " |")
            for row in result.data[:5]:
                values = [str(row.get(c, ""))[:30] for c in result.columns]
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    # SQL
    if result.sql:
        lines.append("## Generated SQL")
        lines.append("```sql")
        lines.append(result.sql)
        lines.append("```")
        lines.append("")

    # Assumptions
    if result.assumptions:
        lines.append("## Assumptions")
        for a in result.assumptions:
            lines.append(f"- {a.assumption} ({a.basis.value})")

    return "\n".join(lines)
