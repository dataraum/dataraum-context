"""Pydantic models for quality rules configuration and evaluation results.

These models represent the structure of rules defined in config/rules/*.yaml
and the results of evaluating those rules against data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Rule Definition Models (Configuration)
# =============================================================================


class RuleDefinition(BaseModel):
    """A single quality rule definition.

    This is the base rule structure used across all rule types.
    """

    rule: str = Field(description="Name of the rule (e.g., 'not_null', 'unique')")
    severity: Literal["error", "warning", "info"] = Field(
        description="Severity level of rule violations"
    )
    description: str = Field(description="Human-readable description of the rule")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Optional parameters for rule evaluation"
    )


class RoleBasedRules(BaseModel):
    """Rules applied based on semantic role of columns.

    Roles are assigned during semantic enrichment (key, timestamp, measure, etc).
    """

    key: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for key/identifier columns"
    )
    timestamp: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for timestamp columns"
    )
    measure: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for measure/metric columns"
    )
    foreign_key: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for foreign key columns"
    )
    dimension: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for dimension/categorical columns"
    )


class TypeBasedRules(BaseModel):
    """Rules applied based on DuckDB data type.

    These rules are applied after type resolution in the profiling phase.
    """

    DOUBLE: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for DOUBLE columns"
    )
    FLOAT: list[RuleDefinition] = Field(default_factory=list, description="Rules for FLOAT columns")
    INTEGER: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for INTEGER columns"
    )
    BIGINT: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for BIGINT columns"
    )
    DATE: list[RuleDefinition] = Field(default_factory=list, description="Rules for DATE columns")
    TIMESTAMP: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for TIMESTAMP columns"
    )
    VARCHAR: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for VARCHAR columns"
    )
    BOOLEAN: list[RuleDefinition] = Field(
        default_factory=list, description="Rules for BOOLEAN columns"
    )


class PatternBasedRule(BaseModel):
    """A rule applied based on column name pattern.

    Pattern is a regex that matches against column names.
    """

    pattern: str = Field(description="Regex pattern to match column names (e.g., '.*_email$')")
    rules: list[RuleDefinition] = Field(description="Rules to apply if pattern matches")


class StatisticalRule(RuleDefinition):
    """A rule applied based on statistical properties.

    These rules are auto-generated from profiling statistics.
    """

    applies_to: Literal["numeric", "categorical", "all"] = Field(
        description="Type of columns this rule applies to"
    )


class ConsistencyRulePattern(BaseModel):
    """Pattern specification for cross-column consistency rules.

    Different pattern types for different consistency checks.
    """

    # Date ordering patterns
    start_column: str | None = Field(default=None, description="Regex for start date column")
    end_column: str | None = Field(default=None, description="Regex for end date column")

    # Amount/balance patterns
    debit_column: str | None = Field(default=None, description="Regex for debit amount column")
    credit_column: str | None = Field(default=None, description="Regex for credit amount column")

    # Generic column pair patterns
    column_a: str | None = Field(default=None, description="Regex for first column in pair")
    column_b: str | None = Field(default=None, description="Regex for second column in pair")


class ConsistencyRule(BaseModel):
    """A cross-column consistency rule.

    These rules check relationships between multiple columns in the same table.
    """

    rule: str = Field(description="Name of the consistency rule")
    description: str = Field(description="Human-readable description")
    pattern: ConsistencyRulePattern = Field(description="Column name patterns to match")
    expression: str = Field(
        description="SQL expression for the consistency check (uses column placeholders)"
    )
    severity: Literal["error", "warning", "info"] = Field(
        description="Severity level of violations"
    )


class CustomRuleTemplate(BaseModel):
    """Template for custom parameterized rules.

    Allows defining reusable rule templates in configuration.
    """

    description: str = Field(description="Description of what this template does")
    parameters: list[dict[str, Any]] = Field(
        default_factory=list, description="Parameter definitions"
    )
    expression: str = Field(description="SQL expression template with parameter placeholders")


class RulesConfig(BaseModel):
    """Complete quality rules configuration.

    This represents the full structure of a rules YAML file.
    """

    name: str = Field(description="Name of the rules configuration")
    version: str = Field(description="Version string (e.g., '1.0.0')")
    description: str = Field(description="Description of this rule set")

    role_based_rules: RoleBasedRules = Field(
        default_factory=RoleBasedRules, description="Rules based on semantic roles"
    )
    type_based_rules: TypeBasedRules = Field(
        default_factory=TypeBasedRules, description="Rules based on data types"
    )
    pattern_based_rules: list[PatternBasedRule] = Field(
        default_factory=list, description="Rules based on column name patterns"
    )
    statistical_rules: list[StatisticalRule] = Field(
        default_factory=list, description="Rules based on statistical properties"
    )
    consistency_rules: list[ConsistencyRule] = Field(
        default_factory=list, description="Cross-column consistency rules"
    )
    custom_rule_templates: dict[str, CustomRuleTemplate] = Field(
        default_factory=dict, description="Reusable custom rule templates"
    )


# =============================================================================
# Rule Evaluation Result Models
# =============================================================================


class RuleViolation(BaseModel):
    """A single rule violation instance.

    Represents one specific failure of a rule on a particular row/column.
    """

    column_name: str | None = Field(
        default=None, description="Name of the column (None for multi-column rules)"
    )
    row_number: int | None = Field(
        default=None, description="Row number where violation occurred (if sampled)"
    )
    value: Any = Field(default=None, description="Actual value that violated the rule")
    expected: str | None = Field(
        default=None, description="Expected value or constraint description"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (e.g., related column values for consistency rules)",
    )


class RuleResult(BaseModel):
    """Result of evaluating a single rule against data.

    Contains pass/fail statistics and sample violations.
    """

    rule_id: str = Field(description="Unique identifier for this rule evaluation (for tracking)")
    rule_name: str = Field(description="Name of the rule that was evaluated")
    rule_type: Literal[
        "role_based", "type_based", "pattern_based", "statistical", "consistency"
    ] = Field(description="Type of rule")
    severity: Literal["error", "warning", "info"] = Field(description="Severity level of this rule")

    # Scope
    table_id: str = Field(description="ID of the table this rule was evaluated against")
    column_id: str | None = Field(
        default=None, description="ID of the column (None for multi-column rules)"
    )

    # Evaluation results
    total_records: int = Field(description="Total number of records evaluated")
    passed_records: int = Field(description="Number of records that passed the rule")
    failed_records: int = Field(description="Number of records that failed the rule")
    pass_rate: float = Field(description="Ratio of passed to total (0.0 to 1.0)")

    # Sample violations (for diagnosis)
    failure_samples: list[RuleViolation] = Field(
        default_factory=list,
        description="Sample of violations (limited to prevent large payloads)",
    )
    max_samples: int = Field(default=10, description="Maximum number of samples collected")

    # Metadata
    execution_time_ms: float = Field(description="Time taken to evaluate this rule in milliseconds")
    evaluated_at: datetime = Field(description="Timestamp when rule was evaluated (UTC)")

    @property
    def has_failures(self) -> bool:
        """Check if there were any rule failures."""
        return self.failed_records > 0

    @property
    def failure_rate(self) -> float:
        """Get failure rate (inverse of pass rate)."""
        return 1.0 - self.pass_rate


class TableRuleResults(BaseModel):
    """Aggregated rule evaluation results for a single table.

    Contains all rule results and summary statistics.
    """

    table_id: str = Field(description="ID of the evaluated table")
    table_name: str = Field(description="Name of the evaluated table")

    # Individual rule results
    rule_results: list[RuleResult] = Field(
        default_factory=list, description="Results for each rule"
    )

    # Summary statistics
    total_rules_evaluated: int = Field(default=0, description="Total number of rules evaluated")
    rules_passed: int = Field(default=0, description="Number of rules fully passed")
    rules_failed: int = Field(default=0, description="Number of rules with failures")

    # Severity breakdown
    error_count: int = Field(default=0, description="Number of error-level failures")
    warning_count: int = Field(default=0, description="Number of warning-level failures")
    info_count: int = Field(default=0, description="Number of info-level issues")

    # Overall metrics
    avg_pass_rate: float = Field(default=1.0, description="Average pass rate across all rules")
    total_violations: int = Field(
        default=0, description="Total number of violations across all rules"
    )

    # Metadata
    evaluated_at: datetime = Field(description="When evaluation was performed")
    evaluation_time_ms: float = Field(default=0.0, description="Total time for all rules")

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level failures."""
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level failures."""
        return self.warning_count > 0


class DatasetRuleResults(BaseModel):
    """Aggregated rule evaluation results for an entire dataset (multiple tables).

    Contains table-level results and dataset-wide summary.
    """

    source_id: str = Field(description="ID of the data source")
    source_name: str = Field(description="Name of the data source")

    # Table-level results
    table_results: list[TableRuleResults] = Field(
        default_factory=list, description="Results for each table"
    )

    # Dataset-wide summary
    total_tables_evaluated: int = Field(default=0, description="Number of tables")
    total_rules_evaluated: int = Field(default=0, description="Total rules across all tables")
    total_violations: int = Field(default=0, description="Total violations across dataset")

    # Severity breakdown
    error_count: int = Field(default=0, description="Dataset-wide error count")
    warning_count: int = Field(default=0, description="Dataset-wide warning count")
    info_count: int = Field(default=0, description="Dataset-wide info count")

    # Metadata
    evaluated_at: datetime = Field(description="When evaluation was performed")
    evaluation_time_ms: float = Field(default=0.0, description="Total evaluation time")

    @property
    def has_errors(self) -> bool:
        """Check if any table has errors."""
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any table has warnings."""
        return self.warning_count > 0
