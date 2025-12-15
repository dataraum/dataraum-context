"""Rules Merger - Phase 9.

Merges LLM filtering recommendations with user-defined YAML rules.

Priority System:
- OVERRIDE: User rule replaces any matching LLM recommendation
- EXTEND: User rule added to LLM recommendations
- SUGGEST: User rule only if no LLM recommendation for that column

Key Design:
- LLM provides intelligent defaults based on actual data quality
- Users can override with domain-specific constraints
- Source attribution maintained (llm vs user_rule)
"""

import logging

from dataraum_context.core.models.base import Result
from dataraum_context.quality.filtering.models import (
    FilteringRecommendations,
    FilteringRulesConfig,
    RulePriority,
)
from dataraum_context.storage.models_v2.core import Column

logger = logging.getLogger(__name__)


async def merge_filtering_rules(
    llm_recommendations: FilteringRecommendations,
    user_rules: FilteringRulesConfig,
    columns: list[Column],
) -> Result[FilteringRecommendations]:
    """Merge LLM recommendations with user-defined rules.

    Priority handling:
    - OVERRIDE: User rule replaces any matching LLM recommendation
    - EXTEND: User rule added to LLM recommendations
    - SUGGEST: User rule only if no LLM recommendation for that column

    Args:
        llm_recommendations: Recommendations from LLM analysis
        user_rules: User-defined rules from YAML
        columns: Column metadata for matching rules

    Returns:
        Result containing merged FilteringRecommendations with source attribution

    Example:
        >>> llm_recs = FilteringRecommendations(
        ...     clean_view_filters=["price > 0"],
        ...     quarantine_criteria=["price <= 0"],
        ...     column_exclusions=[],
        ...     rationale={"price": "LLM detected negative values"}
        ... )
        >>> user_rules = load_filtering_rules("config/filtering/default.yaml")
        >>> merged = await merge_filtering_rules(llm_recs, user_rules, columns)
        >>> # User OVERRIDE rule "price > 0 AND price < 1000000" replaces LLM "price > 0"
    """
    try:
        # Start with LLM recommendations
        merged_clean_filters = list(llm_recommendations.clean_view_filters)
        merged_quarantine_criteria = list(llm_recommendations.quarantine_criteria)
        merged_column_exclusions = list(llm_recommendations.column_exclusions)
        merged_rationale = dict(llm_recommendations.rationale)

        # Track which columns have been handled
        columns_with_llm_filters = _extract_column_names_from_filters(merged_clean_filters)

        # Process user rules by priority
        override_rules = [
            r for r in user_rules.filtering_rules if r.priority == RulePriority.OVERRIDE
        ]
        extend_rules = [r for r in user_rules.filtering_rules if r.priority == RulePriority.EXTEND]
        suggest_rules = [
            r for r in user_rules.filtering_rules if r.priority == RulePriority.SUGGEST
        ]

        # 1. OVERRIDE rules - Replace LLM recommendations
        for rule in override_rules:
            # Find matching columns
            matching_columns = [
                col
                for col in columns
                if rule.matches_column(col.column_name, col.resolved_type, None)
            ]

            for col in matching_columns:
                # Remove any LLM filters for this column
                merged_clean_filters = [
                    f
                    for f in merged_clean_filters
                    if not _filter_references_column(f, col.column_name)
                ]
                merged_quarantine_criteria = [
                    c
                    for c in merged_quarantine_criteria
                    if not _filter_references_column(c, col.column_name)
                ]

                # Add user rule filter (skip if rule.filter is None)
                if rule.filter is None:
                    continue
                user_filter = _apply_template_variables(rule.filter, col)
                merged_clean_filters.append(user_filter)

                # Generate inverse for quarantine
                quarantine_filter = _invert_filter(user_filter)
                if quarantine_filter:
                    merged_quarantine_criteria.append(quarantine_filter)

                # Update rationale
                rationale_key = f"{col.column_name}_override"
                merged_rationale[rationale_key] = (
                    f"User OVERRIDE rule '{rule.name}': {rule.filter} (replaced LLM recommendation)"
                )

                logger.debug(f"Applied OVERRIDE rule '{rule.name}' to column {col.column_name}")

        # 2. EXTEND rules - Add to LLM recommendations
        for rule in extend_rules:
            matching_columns = [
                col
                for col in columns
                if rule.matches_column(col.column_name, col.resolved_type, None)
            ]

            for col in matching_columns:
                # Skip if rule.filter is None
                if rule.filter is None:
                    continue
                user_filter = _apply_template_variables(rule.filter, col)

                # Check if this exact filter already exists (avoid duplicates)
                if user_filter not in merged_clean_filters:
                    merged_clean_filters.append(user_filter)

                    # Generate inverse for quarantine
                    quarantine_filter = _invert_filter(user_filter)
                    if quarantine_filter and quarantine_filter not in merged_quarantine_criteria:
                        merged_quarantine_criteria.append(quarantine_filter)

                    # Update rationale
                    rationale_key = f"{col.column_name}_extend"
                    merged_rationale[rationale_key] = (
                        f"User EXTEND rule '{rule.name}': {rule.filter} (added to LLM recommendations)"
                    )

                    logger.debug(f"Applied EXTEND rule '{rule.name}' to column {col.column_name}")

        # 3. SUGGEST rules - Only if no LLM recommendation for that column
        for rule in suggest_rules:
            matching_columns = [
                col
                for col in columns
                if rule.matches_column(col.column_name, col.resolved_type, None)
            ]

            for col in matching_columns:
                # Only apply if column doesn't already have filters
                if col.column_name not in columns_with_llm_filters:
                    # Skip if rule.filter is None
                    if rule.filter is None:
                        continue
                    user_filter = _apply_template_variables(rule.filter, col)
                    merged_clean_filters.append(user_filter)

                    # Generate inverse for quarantine
                    quarantine_filter = _invert_filter(user_filter)
                    if quarantine_filter:
                        merged_quarantine_criteria.append(quarantine_filter)

                    # Update rationale
                    rationale_key = f"{col.column_name}_suggest"
                    merged_rationale[rationale_key] = (
                        f"User SUGGEST rule '{rule.name}': {rule.filter}"
                    )

                    logger.debug(f"Applied SUGGEST rule '{rule.name}' to column {col.column_name}")

        # Create merged recommendations - preserve all structured fields from LLM
        merged = FilteringRecommendations(
            # Structured filters from LLM (preserved as-is)
            scope_filters=list(llm_recommendations.scope_filters),
            quality_filters=list(llm_recommendations.quality_filters),
            flags=list(llm_recommendations.flags),
            calculation_impacts=list(llm_recommendations.calculation_impacts),
            # Merged legacy fields (LLM + user rules)
            clean_view_filters=merged_clean_filters,
            quarantine_criteria=merged_quarantine_criteria,
            column_exclusions=merged_column_exclusions,
            rationale=merged_rationale,
            # Metadata (preserved from LLM, updated source)
            source="merged",
            confidence=llm_recommendations.confidence,
            requires_acknowledgment=llm_recommendations.requires_acknowledgment,
            business_cycles=list(llm_recommendations.business_cycles),
        )

        logger.info(
            f"Merged filtering rules: {len(merged.clean_view_filters)} clean filters "
            f"({len(llm_recommendations.clean_view_filters)} LLM + "
            f"{len(merged.clean_view_filters) - len(llm_recommendations.clean_view_filters)} user), "
            f"{len(merged.scope_filters)} scope, {len(merged.quality_filters)} quality, "
            f"{len(merged.flags)} flags preserved"
        )

        return Result.ok(merged)

    except Exception as e:
        logger.error(f"Failed to merge filtering rules: {e}")
        return Result.fail(f"Rules merge failed: {e}")


def _extract_column_names_from_filters(filters: list[str]) -> set[str]:
    """Extract column names referenced in filter expressions.

    Simple heuristic: Assumes column name is first word before operator.

    Args:
        filters: List of SQL WHERE clause filters

    Returns:
        Set of column names found in filters
    """
    column_names = set()
    for filter_expr in filters:
        # Simple extraction: "column_name OPERATOR ..."
        # Split on common operators
        for separator in [" IS ", " ~ ", " BETWEEN ", " > ", " < ", " = ", " != "]:
            if separator in filter_expr:
                potential_column = filter_expr.split(separator)[0].strip()
                # Remove leading/trailing characters like parentheses
                potential_column = potential_column.strip("()")
                column_names.add(potential_column)
                break
    return column_names


def _filter_references_column(filter_expr: str, column_name: str) -> bool:
    """Check if a filter expression references a specific column.

    Args:
        filter_expr: SQL WHERE clause filter
        column_name: Column name to check

    Returns:
        True if filter references the column
    """
    # Simple check: column name appears in filter
    # Could be enhanced with proper SQL parsing
    return column_name in filter_expr


def _apply_template_variables(filter_template: str, column: Column) -> str:
    """Apply template variables to a filter expression.

    Supports variables like {column}, {min_value}, {max_value}, etc.

    Args:
        filter_template: Filter with template variables
        column: Column metadata for variable substitution

    Returns:
        Filter with variables substituted
    """
    # Simple substitution
    result = filter_template.replace("{column}", column.column_name)

    # TODO: Add more template variables based on column metadata
    # - {min_value}, {max_value} from statistical profiles
    # - {pattern} from detected patterns
    # - etc.

    return result


def _invert_filter(filter_expr: str) -> str | None:
    """Generate inverse filter for quarantine criteria.

    Inverts common filter patterns:
    - "col IS NOT NULL" → "col IS NULL"
    - "col > 0" → "col <= 0"
    - "col ~ 'pattern'" → "col !~ 'pattern'"
    - "col BETWEEN a AND b" → "col < a OR col > b"

    Args:
        filter_expr: SQL WHERE clause filter

    Returns:
        Inverted filter or None if inversion not supported
    """
    filter_expr = filter_expr.strip()

    # IS NOT NULL → IS NULL
    if " IS NOT NULL" in filter_expr:
        return filter_expr.replace(" IS NOT NULL", " IS NULL")

    # IS NULL → IS NOT NULL
    if " IS NULL" in filter_expr:
        return filter_expr.replace(" IS NULL", " IS NOT NULL")

    # Regex match ~ → not match !~
    if " ~ " in filter_expr:
        return filter_expr.replace(" ~ ", " !~ ")

    # Greater than > → less than or equal <=
    if " > " in filter_expr and " >= " not in filter_expr:
        return filter_expr.replace(" > ", " <= ")

    # Greater or equal >= → less than <
    if " >= " in filter_expr:
        return filter_expr.replace(" >= ", " < ")

    # Less than < → greater than or equal >=
    if " < " in filter_expr and " <= " not in filter_expr:
        return filter_expr.replace(" < ", " >= ")

    # Less or equal <= → greater than >
    if " <= " in filter_expr:
        return filter_expr.replace(" <= ", " > ")

    # BETWEEN a AND b → < a OR > b
    if " BETWEEN " in filter_expr:
        # Parse "col BETWEEN min AND max"
        parts = filter_expr.split(" BETWEEN ")
        if len(parts) == 2:
            column = parts[0].strip()
            range_parts = parts[1].split(" AND ")
            if len(range_parts) == 2:
                min_val = range_parts[0].strip()
                max_val = range_parts[1].strip()
                return f"({column} < {min_val} OR {column} > {max_val})"

    # Default: return None if inversion not supported
    logger.warning(f"Cannot invert filter: {filter_expr}")
    return None
