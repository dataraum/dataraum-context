"""Quality Synthesis Models (Pillar 5).

Aggregates quality metrics from all 4 pillars into dimensional quality scores
and unified quality assessment.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QualityDimension(str, Enum):
    """Standard data quality dimensions."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    ACCURACY = "accuracy"


class QualitySeverity(str, Enum):
    """Severity levels for quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityIssue(BaseModel):
    """A single quality issue detected during assessment."""

    issue_id: str = Field(description="Unique identifier for this issue")
    issue_type: str = Field(description="Type of issue (e.g., 'benford_violation', 'large_gap')")
    severity: QualitySeverity = Field(description="Severity level")
    dimension: QualityDimension = Field(description="Which quality dimension this affects")

    # Scope
    table_id: str | None = Field(None, description="Table ID if table-level issue")
    column_id: str | None = Field(None, description="Column ID if column-level issue")
    column_name: str | None = Field(None, description="Column name for readability")

    # Description
    description: str = Field(description="Human-readable description of the issue")
    recommendation: str | None = Field(None, description="Suggested remediation")

    # Evidence
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting data (metrics, examples, etc.)"
    )

    # Source
    source_pillar: int = Field(description="Which pillar detected this (1-5)")
    source_module: str = Field(description="Which module detected this")
    detected_at: datetime = Field(description="When this was detected")


class DimensionScore(BaseModel):
    """Quality score for a single dimension."""

    dimension: QualityDimension
    score: float = Field(ge=0.0, le=1.0, description="Score from 0 (poor) to 1 (excellent)")

    # Contributing factors
    completeness_ratio: float | None = Field(None, ge=0.0, le=1.0)
    null_ratio: float | None = Field(None, ge=0.0, le=1.0)
    parse_success_rate: float | None = Field(None, ge=0.0, le=1.0)
    validation_pass_rate: float | None = Field(None, ge=0.0, le=1.0)

    # Issues affecting this dimension
    issue_count: int = Field(default=0, description="Number of issues in this dimension")
    critical_issues: int = Field(default=0, description="Number of critical issues")

    # Explanation
    explanation: str = Field(description="What this score means")


class ColumnQualityAssessment(BaseModel):
    """Quality assessment for a single column."""

    column_id: str
    column_name: str

    # Dimensional scores
    dimension_scores: list[DimensionScore] = Field(
        description="Scores for each applicable dimension"
    )

    # Overall score (weighted average of dimensions)
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")

    # Issues specific to this column
    issues: list[QualityIssue] = Field(default_factory=list)

    # Contributing pillars
    has_statistical_quality: bool = Field(default=False)
    has_temporal_quality: bool = Field(default=False)
    has_semantic_context: bool = Field(default=False)

    # Computed at
    assessed_at: datetime


class TableQualityAssessment(BaseModel):
    """Quality assessment for a table."""

    table_id: str
    table_name: str

    # Dimensional scores (table-level)
    dimension_scores: list[DimensionScore] = Field(
        description="Scores for each applicable dimension"
    )

    # Overall score
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall table quality score")

    # Column assessments
    column_assessments: list[ColumnQualityAssessment] = Field(default_factory=list)

    # Table-level issues
    issues: list[QualityIssue] = Field(default_factory=list)

    # Contributing pillars
    has_statistical_quality: bool = Field(default=False)
    has_topological_quality: bool = Field(default=False)
    has_temporal_quality: bool = Field(default=False)
    has_domain_quality: bool = Field(default=False)

    # Computed at
    assessed_at: datetime


class QualitySynthesisResult(BaseModel):
    """Complete quality synthesis for a table and its columns.

    This is the Pillar 5 output that aggregates quality from all other pillars.
    """

    table_id: str
    table_name: str

    # Table-level assessment
    table_assessment: TableQualityAssessment

    # Summary statistics
    total_columns: int
    columns_assessed: int
    total_issues: int
    critical_issues: int
    warnings: int

    # Issue breakdown by dimension
    issues_by_dimension: dict[str, int] = Field(
        default_factory=dict, description="Count of issues per dimension"
    )

    # Issue breakdown by pillar
    issues_by_pillar: dict[int, int] = Field(
        default_factory=dict,
        description="Count of issues per pillar (1=Statistical, 2=Topological, etc.)",
    )

    # Quality summary (for LLM/human consumption)
    quality_summary: str | None = Field(
        None, description="Natural language summary of quality assessment"
    )

    # Recommendations
    top_recommendations: list[str] = Field(
        default_factory=list, description="Prioritized list of recommendations"
    )

    # Metadata
    synthesis_duration_seconds: float
    synthesized_at: datetime


class DatasetQualityOverview(BaseModel):
    """Quality overview for an entire dataset (all tables)."""

    source_id: str
    source_name: str

    # Table assessments
    table_assessments: list[TableQualityAssessment] = Field(default_factory=list)

    # Overall metrics
    total_tables: int
    total_columns: int
    average_table_quality: float = Field(ge=0.0, le=1.0)
    average_column_quality: float = Field(ge=0.0, le=1.0)

    # Issues rollup
    total_issues: int
    critical_issues: int
    issues_by_dimension: dict[str, int] = Field(default_factory=dict)

    # Dataset-level summary
    dataset_summary: str | None = Field(None)

    # Computed at
    assessed_at: datetime
