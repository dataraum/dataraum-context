"""Rule evaluator for data quality validation.

Evaluates quality rules against data in DuckDB by:
1. Querying metadata from SQLAlchemy (Column, SemanticAnnotation, StatisticalProfile)
2. Matching rules to columns based on role/type/pattern
3. Generating and executing DuckDB SQL for each rule
4. Collecting pass/fail statistics and violation samples
5. Returning RuleResult objects

Architecture:
-----------
┌─────────────────────────────────────────────────────────────┐
│ RuleEvaluator                                               │
│                                                             │
│  evaluate_table(table_id, rules_config)                    │
│      ↓                                                      │
│  1. Load metadata from DB (columns, roles, stats)          │
│  2. Match rules to columns                                 │
│  3. Generate SQL for each rule                             │
│  4. Execute SQL, collect results                           │
│  5. Return TableRuleResults                                │
└─────────────────────────────────────────────────────────────┘

Rule Evaluation Strategy:
------------------------
ROLE-BASED RULES:
  - Query SemanticAnnotation for semantic_role
  - Match rules from role_based_rules.{role}
  - Example: semantic_role='key' → apply not_null, unique

TYPE-BASED RULES:
  - Use Column.resolved_type (DuckDB type)
  - Match rules from type_based_rules.{type}
  - Example: resolved_type='DOUBLE' → apply not_nan, not_inf

PATTERN-BASED RULES:
  - Match Column.column_name against regex patterns
  - Apply rules from matching pattern
  - Example: column_name='customer_email' → apply valid_email

STATISTICAL RULES:
  - Query StatisticalProfile for existing stats
  - Apply thresholds from statistical_rules
  - Example: null_ratio > 0.5 → fail null_rate_threshold

CONSISTENCY RULES:
  - Find column pairs matching patterns
  - Generate cross-column WHERE clauses
  - Example: start_date > end_date → fail date_order

SQL Generation Examples:
-----------------------
NOT_NULL:
  SELECT
    COUNT(*) as total,
    COUNT(*) - COUNT({column}) as failed,
    COUNT({column}) as passed
  FROM {table}

UNIQUE:
  SELECT
    COUNT(*) as total,
    COUNT(*) - COUNT(DISTINCT {column}) as failed,
    COUNT(DISTINCT {column}) as passed
  FROM {table}

NOT_NAN (for DOUBLE):
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN isnan({column}) THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN NOT isnan({column}) THEN 1 ELSE 0 END) as passed
  FROM {table}

VALID_EMAIL (pattern-based):
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN NOT regexp_matches({column}, '^[^@]+@[^@]+\\.[^@]+$')
        THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN regexp_matches({column}, '^[^@]+@[^@]+\\.[^@]+$')
        THEN 1 ELSE 0 END) as passed
  FROM {table}

DATE_ORDER (consistency):
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN {start_col} > {end_col} THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN {start_col} <= {end_col} THEN 1 ELSE 0 END) as passed
  FROM {table}

Violation Sampling:
------------------
For failed records, collect samples using:
  SELECT {column}, rowid as row_number
  FROM {table}
  WHERE {violation_condition}
  LIMIT 10

Implementation Status:
---------------------
[ ] Basic rules: not_null, unique, not_nan, not_inf
[ ] Type validation: valid_date, valid_timestamp, max_length
[ ] Pattern validation: valid_email, valid_url, valid_phone
[ ] Statistical: outlier_detection, cardinality_check, null_rate_threshold
[ ] Consistency: date_order, amount_sign
[ ] Aggregation: TableRuleResults, DatasetRuleResults
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import duckdb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.quality.rules.models import (
    RuleResult,
    RulesConfig,
    RuleViolation,
    TableRuleResults,
)
from dataraum_context.storage.models_v2.core import Column, Table
from dataraum_context.storage.models_v2.semantic_context import SemanticAnnotation
from dataraum_context.storage.models_v2.statistical_context import StatisticalProfile

# =============================================================================
# Data Models for Internal Use
# =============================================================================


class ColumnMetadata:
    """Complete metadata for a column needed for rule evaluation."""

    def __init__(
        self,
        column_id: str,
        table_id: str,
        column_name: str,
        resolved_type: str | None,
        semantic_role: str | None = None,
        confidence: float | None = None,
        null_ratio: float | None = None,
        distinct_count: int | None = None,
        cardinality_ratio: float | None = None,
    ):
        self.column_id = column_id
        self.table_id = table_id
        self.column_name = column_name
        self.resolved_type = resolved_type
        self.semantic_role = semantic_role
        self.confidence = confidence
        self.null_ratio = null_ratio
        self.distinct_count = distinct_count
        self.cardinality_ratio = cardinality_ratio


class TableMetadata:
    """Complete metadata for a table needed for rule evaluation."""

    def __init__(
        self,
        table_id: str,
        table_name: str,
        duckdb_path: str,
        columns: list[ColumnMetadata],
    ):
        self.table_id = table_id
        self.table_name = table_name
        self.duckdb_path = duckdb_path
        self.columns = columns


# =============================================================================
# Metadata Loading
# =============================================================================


async def load_table_metadata(
    table_id: str,
    session: AsyncSession,
) -> Result[TableMetadata]:
    """Load complete metadata for a table from the database.

    Loads:
    - Table info (name, duckdb_path)
    - All columns with types
    - Semantic annotations (roles)
    - Statistical profiles

    Args:
        table_id: ID of the table to load metadata for
        session: SQLAlchemy async session

    Returns:
        Result containing TableMetadata with all columns
    """
    # Load table info
    stmt = select(Table).where(Table.table_id == table_id)
    result = await session.execute(stmt)
    table = result.scalar_one_or_none()

    if not table:
        return Result.fail(f"Table not found: {table_id}")

    # Load columns
    columns_stmt = (
        select(Column).where(Column.table_id == table_id).order_by(Column.column_position)
    )
    columns_result = await session.execute(columns_stmt)
    columns = columns_result.scalars().all()

    if not columns:
        return Result.fail(f"No columns found for table: {table_id}")

    # Load semantic annotations (keyed by column_id)
    ann_stmt = (
        select(SemanticAnnotation)
        .join(Column, SemanticAnnotation.column_id == Column.column_id)
        .where(Column.table_id == table_id)
    )
    ann_result = await session.execute(ann_stmt)
    annotations = {ann.column_id: ann for ann in ann_result.scalars().all()}

    # Load statistical profiles (keyed by column_id)
    prof_stmt = (
        select(StatisticalProfile)
        .join(Column, StatisticalProfile.column_id == Column.column_id)
        .where(Column.table_id == table_id)
    )
    prof_result = await session.execute(prof_stmt)
    profiles = {prof.column_id: prof for prof in prof_result.scalars().all()}

    # Build ColumnMetadata objects
    column_metadata_list = []
    for col in columns:
        ann = annotations.get(col.column_id)
        prof = profiles.get(col.column_id)

        col_meta = ColumnMetadata(
            column_id=col.column_id,
            table_id=col.table_id,
            column_name=col.column_name,
            resolved_type=col.resolved_type,
            semantic_role=ann.semantic_role if ann else None,
            confidence=ann.confidence if ann else None,
            null_ratio=prof.null_ratio if prof else None,
            distinct_count=prof.distinct_count if prof else None,
            cardinality_ratio=prof.cardinality_ratio if prof else None,
        )
        column_metadata_list.append(col_meta)

    if not table.duckdb_path:
        return Result.fail(f"Table {table.table_name} has no DuckDB path set")

    table_metadata = TableMetadata(
        table_id=table.table_id,
        table_name=table.table_name,
        duckdb_path=table.duckdb_path,
        columns=column_metadata_list,
    )

    return Result.ok(table_metadata)


# =============================================================================
# Rule Matching
# =============================================================================


def match_role_based_rules(
    column: ColumnMetadata,
    rules_config: RulesConfig,
) -> list[tuple[str, Any]]:
    """Match role-based rules to a column.

    Args:
        column: Column metadata with semantic_role
        rules_config: Rules configuration

    Returns:
        List of (rule_name, rule_definition) tuples that apply
    """
    if not column.semantic_role:
        return []

    matched_rules = []
    role = column.semantic_role.lower()

    # Get rules for this role
    role_rules = getattr(rules_config.role_based_rules, role, [])
    for rule_def in role_rules:
        matched_rules.append((f"role_{role}_{rule_def.rule}", rule_def))

    return matched_rules


def match_type_based_rules(
    column: ColumnMetadata,
    rules_config: RulesConfig,
) -> list[tuple[str, Any]]:
    """Match type-based rules to a column.

    Args:
        column: Column metadata with resolved_type
        rules_config: Rules configuration

    Returns:
        List of (rule_name, rule_definition) tuples that apply
    """
    if not column.resolved_type:
        return []

    matched_rules = []
    col_type = column.resolved_type.upper()

    # Get rules for this type
    type_rules = getattr(rules_config.type_based_rules, col_type, [])
    for rule_def in type_rules:
        matched_rules.append((f"type_{col_type}_{rule_def.rule}", rule_def))

    return matched_rules


def match_pattern_based_rules(
    column: ColumnMetadata,
    rules_config: RulesConfig,
) -> list[tuple[str, Any]]:
    """Match pattern-based rules to a column.

    Args:
        column: Column metadata with column_name
        rules_config: Rules configuration

    Returns:
        List of (rule_name, rule_definition) tuples that apply
    """
    matched_rules = []

    for pattern_rule in rules_config.pattern_based_rules:
        # Match column name against pattern
        if re.match(pattern_rule.pattern, column.column_name, re.IGNORECASE):
            for rule_def in pattern_rule.rules:
                matched_rules.append((f"pattern_{rule_def.rule}", rule_def))

    return matched_rules


# =============================================================================
# SQL Generation Functions
# =============================================================================


def generate_not_null_sql(
    table_path: str,
    column_name: str,
) -> tuple[str, str]:
    """Generate SQL for not_null rule.

    Returns:
        Tuple of (count_sql, sample_sql)
    """
    count_sql = f"""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN "{column_name}" IS NULL THEN 1 ELSE 0 END) as failed,
        SUM(CASE WHEN "{column_name}" IS NOT NULL THEN 1 ELSE 0 END) as passed
    FROM {table_path}
    """

    sample_sql = f"""
    SELECT
        rowid as row_number,
        NULL as value
    FROM {table_path}
    WHERE "{column_name}" IS NULL
    LIMIT 10
    """

    return (count_sql, sample_sql)


def generate_unique_sql(
    table_path: str,
    column_name: str,
) -> tuple[str, str]:
    """Generate SQL for unique rule.

    Returns:
        Tuple of (count_sql, sample_sql)
    """
    count_sql = f"""
    SELECT
        COUNT(*) as total,
        COUNT(*) - COUNT(DISTINCT "{column_name}") as failed,
        COUNT(DISTINCT "{column_name}") as passed
    FROM {table_path}
    """

    # Sample duplicates
    sample_sql = f"""
    SELECT
        "{column_name}" as value,
        COUNT(*) as duplicate_count
    FROM {table_path}
    WHERE "{column_name}" IS NOT NULL
    GROUP BY "{column_name}"
    HAVING COUNT(*) > 1
    LIMIT 10
    """

    return (count_sql, sample_sql)


def generate_not_nan_sql(
    table_path: str,
    column_name: str,
) -> tuple[str, str]:
    """Generate SQL for not_nan rule (floating point columns).

    Returns:
        Tuple of (count_sql, sample_sql)
    """
    count_sql = f"""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN isnan("{column_name}") THEN 1 ELSE 0 END) as failed,
        SUM(CASE WHEN NOT isnan("{column_name}") THEN 1 ELSE 0 END) as passed
    FROM {table_path}
    """

    sample_sql = f"""
    SELECT
        rowid as row_number,
        "{column_name}" as value
    FROM {table_path}
    WHERE isnan("{column_name}")
    LIMIT 10
    """

    return (count_sql, sample_sql)


def generate_not_inf_sql(
    table_path: str,
    column_name: str,
) -> tuple[str, str]:
    """Generate SQL for not_inf rule (floating point columns).

    Returns:
        Tuple of (count_sql, sample_sql)
    """
    count_sql = f"""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN isinf("{column_name}") THEN 1 ELSE 0 END) as failed,
        SUM(CASE WHEN NOT isinf("{column_name}") THEN 1 ELSE 0 END) as passed
    FROM {table_path}
    """

    sample_sql = f"""
    SELECT
        rowid as row_number,
        "{column_name}" as value
    FROM {table_path}
    WHERE isinf("{column_name}")
    LIMIT 10
    """

    return (count_sql, sample_sql)


def generate_numeric_type_sql(
    table_path: str,
    column_name: str,
) -> tuple[str, str]:
    """Generate SQL for numeric_type rule (checks if column is numeric).

    Returns:
        Tuple of (count_sql, sample_sql)
    """
    count_sql = f"""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN TRY_CAST("{column_name}" AS DOUBLE) IS NULL THEN 1 ELSE 0 END) as failed,
        SUM(CASE WHEN TRY_CAST("{column_name}" AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END) as passed
    FROM {table_path}
    """

    sample_sql = f"""
    SELECT
        rowid as row_number,
        "{column_name}" as value
    FROM {table_path}
    WHERE TRY_CAST("{column_name}" AS DOUBLE) IS NULL
    LIMIT 10
    """

    return (count_sql, sample_sql)


# =============================================================================
# Rule Evaluator (Main Class)
# =============================================================================


class RuleEvaluator:
    """Evaluates quality rules against data in DuckDB.

    Uses metadata from SQLAlchemy database to determine which rules
    apply to which columns, then generates and executes DuckDB SQL
    to check data quality.
    """

    def __init__(
        self,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
    ):
        """Initialize evaluator.

        Args:
            duckdb_conn: DuckDB connection for data queries
            session: SQLAlchemy session for metadata queries
        """
        self.duckdb_conn = duckdb_conn
        self.session = session

    def _get_sql_generator(self, rule_name: str) -> Any:
        """Get SQL generator function for a rule.

        Args:
            rule_name: Name of the rule

        Returns:
            SQL generator function or None
        """
        generators = {
            "not_null": generate_not_null_sql,
            "unique": generate_unique_sql,
            "not_nan": generate_not_nan_sql,
            "not_inf": generate_not_inf_sql,
            "numeric_type": generate_numeric_type_sql,
        }
        return generators.get(rule_name)

    def _execute_rule_sql(
        self,
        count_sql: str,
        sample_sql: str,
    ) -> tuple[int, int, int, list[dict[str, Any]]]:
        """Execute rule SQL and collect results.

        Args:
            count_sql: SQL to get pass/fail counts
            sample_sql: SQL to get violation samples

        Returns:
            Tuple of (total, passed, failed, samples)
        """
        # Execute count query
        count_result = self.duckdb_conn.execute(count_sql).fetchone()
        if count_result is None:
            return (0, 0, 0, [])

        total = int(count_result[0]) if count_result[0] is not None else 0
        failed = int(count_result[1]) if count_result[1] is not None else 0
        passed = int(count_result[2]) if count_result[2] is not None else 0

        # Execute sample query
        sample_result = self.duckdb_conn.execute(sample_sql).fetchall()
        samples = []
        for row in sample_result:
            sample_dict = {}
            if len(row) >= 1:
                sample_dict["row_number"] = row[0]
            if len(row) >= 2:
                sample_dict["value"] = row[1]
            samples.append(sample_dict)

        return (total, passed, failed, samples)

    async def evaluate_table(
        self,
        table_id: str,
        rules_config: RulesConfig,
    ) -> Result[TableRuleResults]:
        """Evaluate all applicable rules against a table.

        Args:
            table_id: ID of table to evaluate
            rules_config: Rules configuration to apply

        Returns:
            Result containing TableRuleResults with all rule outcomes
        """
        start_time = time.time()

        # Load metadata
        metadata_result = await load_table_metadata(table_id, self.session)
        if not metadata_result.success:
            return Result.fail(metadata_result.error or "Failed to load metadata")

        table_meta = metadata_result.unwrap()

        # Collect all rule results
        rule_results: list[RuleResult] = []

        # Evaluate rules for each column
        for column in table_meta.columns:
            # Match role-based rules
            role_rules = match_role_based_rules(column, rules_config)

            # Match type-based rules
            type_rules = match_type_based_rules(column, rules_config)

            # Match pattern-based rules
            pattern_rules = match_pattern_based_rules(column, rules_config)

            # Combine all matched rules
            all_rules = role_rules + type_rules + pattern_rules

            # Evaluate each matched rule
            for rule_name, rule_def in all_rules:
                # Get SQL generator for this rule
                sql_generator = self._get_sql_generator(rule_def.rule)
                if not sql_generator:
                    # Skip rules we don't have generators for yet
                    continue

                try:
                    # Generate SQL
                    count_sql, sample_sql = sql_generator(
                        table_meta.duckdb_path,
                        column.column_name,
                    )

                    # Execute SQL and collect results
                    rule_start = time.time()
                    total, passed, failed, samples = self._execute_rule_sql(count_sql, sample_sql)
                    rule_time = (time.time() - rule_start) * 1000  # ms

                    # Calculate pass rate
                    pass_rate = passed / total if total > 0 else 1.0

                    # Build violation samples
                    violations = []
                    for sample in samples:
                        violation = RuleViolation(
                            column_name=column.column_name,
                            row_number=sample.get("row_number"),
                            value=sample.get("value"),
                            expected=rule_def.description,
                        )
                        violations.append(violation)

                    # Create RuleResult
                    result = RuleResult(
                        rule_id=str(uuid4()),
                        rule_name=rule_def.rule,
                        rule_type=self._infer_rule_type(rule_name),
                        severity=rule_def.severity,
                        table_id=table_id,
                        column_id=column.column_id,
                        total_records=total,
                        passed_records=passed,
                        failed_records=failed,
                        pass_rate=pass_rate,
                        failure_samples=violations,
                        max_samples=10,
                        execution_time_ms=rule_time,
                        evaluated_at=datetime.now(UTC),
                    )
                    rule_results.append(result)

                except Exception:
                    # Log but don't fail the entire evaluation
                    continue

        # Aggregate into TableRuleResults
        evaluation_time = (time.time() - start_time) * 1000  # ms

        # Calculate summary statistics
        total_rules = len(rule_results)
        rules_passed = sum(1 for r in rule_results if not r.has_failures)
        rules_failed = sum(1 for r in rule_results if r.has_failures)

        error_count = sum(1 for r in rule_results if r.severity == "error" and r.has_failures)
        warning_count = sum(1 for r in rule_results if r.severity == "warning" and r.has_failures)
        info_count = sum(1 for r in rule_results if r.severity == "info" and r.has_failures)

        avg_pass_rate = (
            sum(r.pass_rate for r in rule_results) / total_rules if total_rules > 0 else 1.0
        )
        total_violations = sum(r.failed_records for r in rule_results)

        table_results = TableRuleResults(
            table_id=table_id,
            table_name=table_meta.table_name,
            rule_results=rule_results,
            total_rules_evaluated=total_rules,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            avg_pass_rate=avg_pass_rate,
            total_violations=total_violations,
            evaluated_at=datetime.now(UTC),
            evaluation_time_ms=evaluation_time,
        )

        return Result.ok(table_results)

    def _infer_rule_type(self, rule_name: str) -> str:
        """Infer rule type from rule name prefix.

        Args:
            rule_name: Full rule name (e.g., "role_key_not_null")

        Returns:
            Rule type: "role_based", "type_based", "pattern_based", "statistical", "consistency"
        """
        if rule_name.startswith("role_"):
            return "role_based"
        elif rule_name.startswith("type_"):
            return "type_based"
        elif rule_name.startswith("pattern_"):
            return "pattern_based"
        elif rule_name.startswith("stat_"):
            return "statistical"
        elif rule_name.startswith("consistency_"):
            return "consistency"
        else:
            return "role_based"  # default


# =============================================================================
# Public API
# =============================================================================


async def evaluate_table_rules(
    table_id: str,
    rules_config: RulesConfig,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
) -> Result[TableRuleResults]:
    """Evaluate quality rules against a table.

    Convenience function that creates an evaluator and runs evaluation.

    Args:
        table_id: ID of table to evaluate
        rules_config: Rules configuration to apply
        duckdb_conn: DuckDB connection
        session: SQLAlchemy async session

    Returns:
        Result containing TableRuleResults
    """
    evaluator = RuleEvaluator(duckdb_conn, session)
    return await evaluator.evaluate_table(table_id, rules_config)
