"""Quality layer models.

Defines data structures for quality assessment and rules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dataraum_context.core.models.base import DecisionSource, QualitySeverity


class QualityRule(BaseModel):
    """A quality rule."""

    rule_id: str

    table_name: str
    column_name: str | None = None

    rule_name: str
    rule_type: str
    rule_expression: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    severity: QualitySeverity
    source: DecisionSource
    description: str | None = None


class RuleResult(BaseModel):
    """Result of a single rule execution."""

    rule_id: str
    rule_name: str

    total_records: int
    passed_records: int
    failed_records: int
    pass_rate: float

    failure_samples: list[dict[str, Any]] = Field(default_factory=list)


class QualityScore(BaseModel):
    """Aggregate quality score."""

    scope: str  # 'table' or 'column'
    scope_id: str
    scope_name: str

    completeness: float
    validity: float
    consistency: float
    uniqueness: float
    timeliness: float

    overall: float


class Anomaly(BaseModel):
    """A detected anomaly."""

    table_name: str
    column_name: str | None = None

    anomaly_type: str
    description: str
    severity: QualitySeverity
    evidence: dict[str, Any] = Field(default_factory=dict)


# === Context Models ===
