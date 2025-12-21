"""Semantic annotation resolver for validation checks.

Resolves column requirements to actual column names using semantic annotations.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dataraum_context.analysis.semantic.db_models import SemanticAnnotation
from dataraum_context.analysis.validation.models import (
    ColumnRequirement,
    ColumnResolutionResult,
    ResolvedColumn,
    ValidationSpec,
)
from dataraum_context.storage import Column, Table

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def resolve_columns(
    session: AsyncSession,
    table_id: str,
    spec: ValidationSpec,
) -> ColumnResolutionResult:
    """Resolve column requirements for a validation spec.

    Uses semantic annotations to find matching columns. Falls back to
    name pattern matching if semantic criteria don't match.

    Args:
        session: Database session
        table_id: Table to resolve columns for
        spec: Validation spec with column requirements

    Returns:
        ColumnResolutionResult with resolved and unresolved columns
    """
    result = ColumnResolutionResult()

    # Load table with columns and their semantic annotations
    table_query = (
        select(Table)
        .where(Table.table_id == table_id)
        .options(selectinload(Table.columns).selectinload(Column.semantic_annotation))
    )
    table_result = await session.execute(table_query)
    table = table_result.scalar_one_or_none()

    if not table:
        result.error_message = f"Table {table_id} not found"
        return result

    if not table.duckdb_path:
        result.error_message = f"Table {table_id} has no DuckDB path"
        return result

    # Build column lookup with their annotations
    columns_with_annotations: list[tuple[Column, SemanticAnnotation | None]] = []
    for col in table.columns:
        annotation = col.semantic_annotation
        columns_with_annotations.append((col, annotation))

    # Resolve each requirement
    for alias, requirement in spec.column_requirements.items():
        resolved = _resolve_single_requirement(
            requirement=requirement,
            columns_with_annotations=columns_with_annotations,
            table=table,
        )

        if resolved:
            result.resolved[alias] = resolved
            logger.debug(
                f"Resolved '{alias}' -> '{resolved.column_name}' via {resolved.matched_by}"
            )
        else:
            if requirement.required:
                result.unresolved.append(alias)
                logger.warning(
                    f"Failed to resolve required column '{alias}' for {spec.validation_id}"
                )
            else:
                logger.debug(f"Optional column '{alias}' not resolved")

    # Determine overall success
    result.all_resolved = len(result.unresolved) == 0

    if not result.all_resolved:
        result.error_message = f"Could not resolve required columns: {result.unresolved}"

    return result


def _resolve_single_requirement(
    requirement: ColumnRequirement,
    columns_with_annotations: list[tuple[Column, SemanticAnnotation | None]],
    table: Table,
) -> ResolvedColumn | None:
    """Resolve a single column requirement.

    Args:
        requirement: The column requirement to resolve
        columns_with_annotations: List of (Column, SemanticAnnotation) tuples
        table: The table being validated

    Returns:
        ResolvedColumn if found, None otherwise
    """
    best_match: ResolvedColumn | None = None
    best_confidence: float = 0.0

    for col, annotation in columns_with_annotations:
        matched_by: str | None = None
        confidence: float = 0.0

        # Try semantic matching first (higher priority)
        if requirement.has_semantic_criteria() and annotation:
            matched_by, confidence = _match_semantic(requirement, annotation)

        # Fall back to name pattern matching
        if not matched_by and requirement.name_patterns:
            matched_by, confidence = _match_name_patterns(
                requirement.name_patterns, col.column_name
            )

        if matched_by and confidence > best_confidence:
            best_confidence = confidence
            best_match = ResolvedColumn(
                column_id=col.column_id,
                column_name=col.column_name,
                table_id=table.table_id,
                table_name=table.table_name,
                duckdb_path=table.duckdb_path or f"typed_{table.table_name}",
                matched_by=matched_by,
                confidence=confidence,
                semantic_role=annotation.semantic_role if annotation else None,
                entity_type=annotation.entity_type if annotation else None,
                business_name=annotation.business_name if annotation else None,
            )

    return best_match


def _match_semantic(
    requirement: ColumnRequirement,
    annotation: SemanticAnnotation,
) -> tuple[str | None, float]:
    """Try to match a column via semantic annotations.

    Args:
        requirement: Column requirement with semantic criteria
        annotation: Semantic annotation to match against

    Returns:
        Tuple of (matched_by, confidence) or (None, 0) if no match
    """
    # Check semantic_role match
    if requirement.semantic_role:
        if (
            annotation.semantic_role
            and annotation.semantic_role.lower() == requirement.semantic_role.lower()
        ):
            return ("semantic_role", 0.9)

    # Check entity_type match (highest priority for validation)
    if requirement.entity_type:
        if (
            annotation.entity_type
            and annotation.entity_type.lower() == requirement.entity_type.lower()
        ):
            return ("entity_type", 0.95)

    # Check business_domain match
    if requirement.business_domain:
        if (
            annotation.business_domain
            and annotation.business_domain.lower() == requirement.business_domain.lower()
        ):
            return ("business_domain", 0.85)

    # Check ontology_term match (exact match required)
    if requirement.ontology_term:
        if annotation.ontology_term and annotation.ontology_term == requirement.ontology_term:
            return ("ontology_term", 1.0)

    return (None, 0.0)


def _match_name_patterns(patterns: list[str], column_name: str) -> tuple[str | None, float]:
    """Try to match a column via name patterns.

    Args:
        patterns: List of patterns to match (case-insensitive)
        column_name: Column name to match against

    Returns:
        Tuple of (matched_by, confidence) or (None, 0) if no match
    """
    column_lower = column_name.lower()

    for pattern in patterns:
        pattern_lower = pattern.lower()

        # Exact match
        if column_lower == pattern_lower:
            return ("name_pattern", 0.9)

        # Prefix match
        if column_lower.startswith(pattern_lower):
            return ("name_pattern", 0.7)

        # Contains match
        if pattern_lower in column_lower:
            return ("name_pattern", 0.5)

        # Regex match
        try:
            if re.match(pattern, column_name, re.IGNORECASE):
                return ("name_pattern", 0.6)
        except re.error:
            pass

    return (None, 0.0)


async def get_table_schema_for_llm(
    session: AsyncSession,
    table_id: str,
) -> dict[str, Any]:
    """Get table schema with semantic annotations for LLM context.

    Args:
        session: Database session
        table_id: Table ID

    Returns:
        Dict with table info and column details
    """
    table_query = (
        select(Table)
        .where(Table.table_id == table_id)
        .options(selectinload(Table.columns).selectinload(Column.semantic_annotation))
    )
    table_result = await session.execute(table_query)
    table = table_result.scalar_one_or_none()

    if not table:
        return {"error": f"Table {table_id} not found"}

    columns = []
    for col in table.columns:
        col_info: dict[str, Any] = {
            "column_name": col.column_name,
            "data_type": col.resolved_type or col.raw_type,
        }

        if col.semantic_annotation:
            ann = col.semantic_annotation
            col_info["semantic"] = {
                "role": ann.semantic_role,
                "entity_type": ann.entity_type,
                "business_name": ann.business_name,
                "domain": ann.business_domain,
            }

        columns.append(col_info)

    return {
        "table_name": table.table_name,
        "duckdb_path": table.duckdb_path or f"typed_{table.table_name}",
        "columns": columns,
    }


__all__ = [
    "resolve_columns",
    "get_table_schema_for_llm",
]
