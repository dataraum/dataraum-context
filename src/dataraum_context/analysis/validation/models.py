"""Pydantic models for generic validation checks.

Contains data structures for validation specs, results, and SQL generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    """Severity levels for validation failures."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ColumnRequirement(BaseModel):
    """A column requirement for a validation check.

    Uses semantic annotations to resolve actual column names.
    """

    # Semantic matching criteria (at least one required)
    semantic_role: str | None = None  # 'measure', 'identifier', 'attribute'
    entity_type: str | None = None  # 'amount', 'account', 'transaction_date'
    business_domain: str | None = None  # 'finance', 'marketing'
    ontology_term: str | None = None  # Specific ontology term

    # Alternative: direct column name patterns (fallback)
    name_patterns: list[str] = Field(default_factory=list)  # e.g., ["debit", "dr"]

    # Resolution behavior
    required: bool = True
    alias: str = ""  # Alias to use in SQL (e.g., "debit_column")

    def has_semantic_criteria(self) -> bool:
        """Check if this requirement uses semantic matching."""
        return bool(
            self.semantic_role or self.entity_type or self.business_domain or self.ontology_term
        )


class ValidationSpec(BaseModel):
    """Specification for a validation check.

    Loaded from YAML configuration files.
    """

    validation_id: str
    name: str
    description: str
    category: str  # 'financial', 'data_quality', 'business_rule'
    severity: ValidationSeverity = ValidationSeverity.ERROR

    # Column requirements (resolved via semantic annotations)
    column_requirements: dict[str, ColumnRequirement] = Field(default_factory=dict)

    # Check definition
    check_type: str  # 'balance', 'comparison', 'constraint', 'aggregate'
    parameters: dict[str, Any] = Field(default_factory=dict)

    # SQL generation hints for LLM
    sql_hints: str | None = None  # Free-form hints for SQL generation
    expected_outcome: str | None = None  # What a passing result looks like

    # Metadata
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0"
    source: str = "config"


class ResolvedColumn(BaseModel):
    """A column resolved from semantic annotations."""

    column_id: str
    column_name: str
    table_id: str
    table_name: str
    duckdb_path: str

    # How it was matched
    matched_by: str  # 'semantic_role', 'entity_type', 'ontology_term', 'name_pattern'
    confidence: float = 1.0

    # Original semantic annotation
    semantic_role: str | None = None
    entity_type: str | None = None
    business_name: str | None = None


class ColumnResolutionResult(BaseModel):
    """Result of resolving column requirements."""

    resolved: dict[str, ResolvedColumn] = Field(default_factory=dict)
    unresolved: list[str] = Field(default_factory=list)
    all_resolved: bool = False
    error_message: str | None = None


class GeneratedSQL(BaseModel):
    """LLM-generated SQL for a validation check."""

    validation_id: str
    sql_query: str
    explanation: str | None = None

    # Caching metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str | None = None
    prompt_hash: str | None = None  # For cache key

    # Validation info
    is_valid: bool = True
    validation_error: str | None = None


class ValidationResult(BaseModel):
    """Result of executing a validation check."""

    validation_id: str
    spec_name: str
    status: ValidationStatus
    severity: ValidationSeverity

    # Execution details
    table_id: str
    table_name: str
    executed_at: datetime = Field(default_factory=datetime.utcnow)

    # Results
    passed: bool = False
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    # SQL execution
    sql_used: str | None = None
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0

    # Resolution info
    columns_resolved: dict[str, str] = Field(default_factory=dict)  # alias -> column_name


class ValidationRunResult(BaseModel):
    """Result of running all validations for a table."""

    run_id: str
    table_id: str
    table_name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Results
    results: list[ValidationResult] = Field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    error_checks: int = 0

    # Summary
    overall_status: ValidationStatus = ValidationStatus.PASSED
    has_critical_failures: bool = False


__all__ = [
    "ValidationSeverity",
    "ValidationStatus",
    "ColumnRequirement",
    "ValidationSpec",
    "ResolvedColumn",
    "ColumnResolutionResult",
    "GeneratedSQL",
    "ValidationResult",
    "ValidationRunResult",
]
