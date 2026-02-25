"""Utility functions for the SQL Knowledge Base.

Provides:
- SQL normalization (lowercase, collapse whitespace)
- Expression normalization (commutative sorting, placeholder substitution)
- Usage type determination (exact_reuse, adapted, etc.)
- Snippet usage tracking (used by the graph agent)
"""

from __future__ import annotations

import re
from typing import Any


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, collapse whitespace.

    Args:
        sql: Raw SQL string

    Returns:
        Normalized SQL string
    """
    s = sql.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# --- Expression Normalization ---

# Regex to match a binary expression: operand OPERATOR operand
# Supports nested parenthesized expressions
_OPERATOR_PATTERN = re.compile(r"([+\-*/])")


def normalize_expression(expression: str) -> tuple[str, list[str], dict[str, str]]:
    """Normalize a mathematical expression for template matching.

    Canonicalizes commutative operations by sorting operands alphabetically.
    Replaces concrete field names with placeholders ({A}, {B}, ...).

    Rules:
    - a * b -> sort operands (a < b alphabetically)
    - a + b -> sort operands
    - a / b -> preserve order (NOT commutative)
    - a - b -> preserve order (NOT commutative)
    - Parentheses preserved for precedence

    Args:
        expression: A formula expression like "(accounts_receivable / revenue) * days_in_period"

    Returns:
        Tuple of:
        - normalized_expression: Expression with sorted commutative ops and placeholders
        - sorted_input_fields: Alphabetically sorted list of unique field names
        - input_bindings: Mapping from placeholder ({A}, {B}, ...) to field name
    """
    # Extract unique field names (identifiers that aren't numbers or operators)
    fields = _extract_fields(expression)
    sorted_fields = sorted(set(fields))

    # Create placeholder mapping
    bindings: dict[str, str] = {}
    field_to_placeholder: dict[str, str] = {}
    for i, field_name in enumerate(sorted_fields):
        placeholder = "{" + chr(65 + i) + "}"  # {A}, {B}, {C}, ...
        bindings[placeholder] = field_name
        field_to_placeholder[field_name] = placeholder

    # Replace field names with placeholders (longest first to avoid partial matches)
    normalized = expression.strip()
    for field_name in sorted(field_to_placeholder.keys(), key=len, reverse=True):
        normalized = normalized.replace(field_name, field_to_placeholder[field_name])

    # Sort commutative operations
    normalized = _sort_commutative_ops(normalized)

    return normalized, sorted_fields, bindings


def _extract_fields(expression: str) -> list[str]:
    """Extract field name identifiers from an expression.

    Matches sequences of word characters (letters, digits, underscore)
    that are not pure numbers.

    Args:
        expression: Math expression string

    Returns:
        List of field name strings (may contain duplicates)
    """
    # Match word sequences that aren't pure numbers
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression)
    return tokens


def _sort_commutative_ops(expr: str) -> str:
    """Sort operands of commutative operations (+ and *).

    Handles top-level commutative operations. For nested expressions,
    works on the string level after placeholder substitution.

    This is simple text manipulation, not full symbolic algebra.
    Handles common patterns: ratio * constant, sum of components.

    Args:
        expr: Expression with placeholders

    Returns:
        Expression with sorted commutative operands
    """
    # Handle multiplication: split on *, sort parts, rejoin
    expr = _sort_binary_commutative(expr, "*")
    # Handle addition: split on +, sort parts, rejoin
    expr = _sort_binary_commutative(expr, "+")
    return expr


def _sort_binary_commutative(expr: str, operator: str) -> str:
    """Sort operands around a commutative operator.

    Only sorts at the top level (respects parentheses).

    Args:
        expr: Expression string
        operator: The operator to sort around ("*" or "+")

    Returns:
        Expression with sorted operands for this operator
    """
    parts = _split_at_top_level(expr, operator)
    if len(parts) <= 1:
        return expr

    # Sort the parts (strip whitespace for comparison, preserve structure)
    stripped = [p.strip() for p in parts]
    sorted_parts = sorted(stripped)

    op_with_space = f" {operator} "
    return op_with_space.join(sorted_parts)


def _split_at_top_level(expr: str, operator: str) -> list[str]:
    """Split expression by operator only at top level (outside parentheses).

    Args:
        expr: Expression string
        operator: Single character operator

    Returns:
        List of parts
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == operator and depth == 0:
            # Check it's not part of ** (power operator)
            if operator == "*" and i + 1 < len(expr) and expr[i + 1] == "*":
                current.append(ch)
            else:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1

    parts.append("".join(current))
    return parts


# --- Usage Type Determination ---


def determine_usage_type(
    generated_sql: str,
    provided_snippet_sql: str | None,
) -> str:
    """Determine how a generated SQL step relates to a provided snippet.

    Uses normalized SQL equality (lowercase, collapsed whitespace).
    If the step_id matched a provided snippet, the SQL is either an
    exact reuse or an adaptation — never "provided_not_used" (that case
    is handled separately for snippets whose step_id wasn't generated).

    Args:
        generated_sql: The SQL actually generated/used
        provided_snippet_sql: The snippet SQL that was provided (None if no snippet)

    Returns:
        One of: "exact_reuse", "adapted", "newly_generated"
    """
    if provided_snippet_sql is None:
        return "newly_generated"

    if normalize_sql(generated_sql) == normalize_sql(provided_snippet_sql):
        return "exact_reuse"
    else:
        return "adapted"


def track_snippet_usage(
    library: Any,
    execution_id: str,
    execution_type: str,
    provided_snippets: dict[str, dict[str, Any]],
    generated_steps: list[dict[str, str]],
) -> None:
    """Track how provided snippets were used in an execution.

    Used by the graph agent (step_id-based matching). The query agent
    has its own deterministic tracking via snippet_id on each step.

    Classification (deterministic, via normalize_sql equality):
    - step_id matches provided snippet + normalized SQL equal → exact_reuse
    - step_id matches provided snippet + SQL differs → adapted
    - step_id not in provided snippets → newly_generated
    - Provided snippets whose step_id wasn't generated → provided_not_used

    Args:
        library: SnippetLibrary instance
        execution_id: Execution or cache key identifier
        execution_type: "query" or "graph"
        provided_snippets: Dict of step_id -> {sql, snippet_id, ...}
        generated_steps: List of dicts with step_id, sql, description
    """
    used_snippet_ids: set[str] = set()

    for gen_step in generated_steps:
        step_id = gen_step.get("step_id", "")
        provided = provided_snippets.get(step_id)

        if provided is None:
            library.record_usage(
                execution_id=execution_id,
                execution_type=execution_type,
                usage_type="newly_generated",
                step_id=step_id,
            )
        else:
            snippet_id = provided.get("snippet_id")
            provided_sql = provided.get("sql", "")
            gen_sql = gen_step.get("sql", "")
            usage_type = determine_usage_type(gen_sql, provided_sql)
            is_exact = usage_type == "exact_reuse"

            library.record_usage(
                execution_id=execution_id,
                execution_type=execution_type,
                usage_type=usage_type,
                snippet_id=snippet_id,
                match_confidence=1.0,
                sql_match_ratio=1.0 if is_exact else 0.0,
                step_id=step_id,
            )
            if snippet_id:
                used_snippet_ids.add(snippet_id)

    # Record provided_not_used for snippets not referenced by any generated step
    generated_step_ids = {s.get("step_id", "") for s in generated_steps}
    for step_id, provided in provided_snippets.items():
        if step_id not in generated_step_ids:
            snippet_id = provided.get("snippet_id")
            if snippet_id and snippet_id not in used_snippet_ids:
                library.record_usage(
                    execution_id=execution_id,
                    execution_type=execution_type,
                    usage_type="provided_not_used",
                    snippet_id=snippet_id,
                    step_id=step_id,
                )


__all__ = [
    "normalize_sql",
    "normalize_expression",
    "track_snippet_usage",
]
