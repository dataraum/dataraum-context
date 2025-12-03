"""Profiling layer models.

Defines data structures for statistical profiling and type inference."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from dataraum_context.core.models.base import (
    ColumnRef,
    DataType,
    DecisionSource,
)


class NumericStats(BaseModel):
    """Statistics for numeric columns."""

    min_value: float
    max_value: float
    mean: float
    stddev: float
    skewness: float | None = None
    kurtosis: float | None = None
    cv: float | None = None  # Coefficient of variation (stddev/mean)
    percentiles: dict[str, float | None] = Field(default_factory=dict)


class StringStats(BaseModel):
    """Statistics for string columns."""

    min_length: int
    max_length: int
    avg_length: float


class HistogramBucket(BaseModel):
    """A histogram bucket."""

    bucket_min: float | str
    bucket_max: float | str
    count: int


class ValueCount(BaseModel):
    """A value with its count."""

    value: Any
    count: int
    percentage: float


class DetectedPattern(BaseModel):
    """A detected pattern in column values."""

    name: str
    match_rate: float
    semantic_type: str | None = None


class TypeCandidate(BaseModel):
    """A candidate type for a column."""

    column_id: str
    column_ref: ColumnRef

    data_type: DataType
    confidence: float
    parse_success_rate: float
    failed_examples: list[str] = Field(default_factory=list)

    detected_pattern: str | None = None
    pattern_match_rate: float | None = None

    detected_unit: str | None = None
    unit_confidence: float | None = None


class ColumnProfile(BaseModel):
    """Statistical profile of a column."""

    column_id: str
    column_ref: ColumnRef
    profiled_at: datetime

    total_count: int
    null_count: int
    distinct_count: int

    null_ratio: float
    cardinality_ratio: float

    numeric_stats: NumericStats | None = None
    string_stats: StringStats | None = None

    histogram: list[HistogramBucket] | None = None
    top_values: list[ValueCount] | None = None
    detected_patterns: list[DetectedPattern] = Field(default_factory=list)


class ProfileResult(BaseModel):
    """Result of profiling operation."""

    profiles: list[ColumnProfile]
    type_candidates: list[TypeCandidate]
    duration_seconds: float


# === Type Resolution Models ===


class TypeDecision(BaseModel):
    """A type decision for a column."""

    column_id: str
    decided_type: DataType
    decision_source: DecisionSource = DecisionSource.AUTO
    decision_reason: str | None = None


class ColumnCastResult(BaseModel):
    """Cast result for a single column."""

    column_id: str
    column_ref: ColumnRef
    source_type: str
    target_type: DataType
    success_count: int
    failure_count: int
    success_rate: float
    failure_samples: list[str] = Field(default_factory=list)


class TypeResolutionResult(BaseModel):
    """Result of type resolution."""

    typed_table_name: str
    quarantine_table_name: str

    total_rows: int
    typed_rows: int
    quarantined_rows: int

    column_results: list[ColumnCastResult]


# === Statistical Quality Models ===


class BenfordAnalysis(BaseModel):
    """Benford's Law compliance analysis."""

    chi_square: float
    p_value: float
    is_compliant: bool  # p_value > 0.05
    digit_distribution: dict[str, float]  # {1: 0.301, 2: 0.176, ...}
    interpretation: str


class OutlierDetection(BaseModel):
    """Outlier detection results."""

    # IQR Method
    iqr_lower_fence: float
    iqr_upper_fence: float
    iqr_outlier_count: int
    iqr_outlier_ratio: float

    # Isolation Forest
    isolation_forest_score: float  # Average anomaly score
    isolation_forest_anomaly_count: int
    isolation_forest_anomaly_ratio: float

    # Sample outliers
    outlier_samples: list[dict[str, Any]] = Field(default_factory=list)  # [{value, method, score}]


class StatisticalQualityResult(BaseModel):
    """Comprehensive statistical quality assessment.

    This is the Pydantic source of truth for statistical quality metrics.
    Gets serialized to StatisticalQualityMetrics.quality_data JSONB field.
    """

    column_id: str
    column_ref: ColumnRef

    # Benford's Law (for financial/count columns)
    benford_analysis: BenfordAnalysis | None = None

    # Outlier detection
    outlier_detection: OutlierDetection | None = None

    # Overall quality assessment
    quality_score: float  # 0-1 aggregate
    quality_issues: list[dict[str, Any]] = Field(
        default_factory=list
    )  # [{issue_type, severity, description}]


class StatisticalProfilingResult(BaseModel):
    """Result of statistical profiling and quality analysis operation."""

    profiles: list[ColumnProfile] = Field(default_factory=list)
    statistical_quality: list[StatisticalQualityResult] = Field(default_factory=list)
    duration_seconds: float


# === Multicollinearity Models ===


class ColumnVIF(BaseModel):
    """Column-level multicollinearity assessment.

    VIF (Variance Inflation Factor) measures how much a column's variance
    is inflated due to correlation with other columns.
    """

    column_id: str
    column_ref: ColumnRef
    vif: float  # Variance Inflation Factor
    tolerance: float  # 1/VIF
    has_multicollinearity: bool  # VIF > 10 or Tolerance < 0.1
    severity: Literal["none", "moderate", "severe"]  # VIF: <5, 5-10, >10
    correlated_with: list[str] = Field(default_factory=list)  # Related column IDs

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of VIF score."""
        if self.vif < 5:
            return f"Low multicollinearity (VIF={self.vif:.1f})"
        elif self.vif < 10:
            return f"Moderate multicollinearity (VIF={self.vif:.1f})"
        else:
            return f"Severe multicollinearity (VIF={self.vif:.1f})"


class ConditionIndexAnalysis(BaseModel):
    """Table-level multicollinearity assessment via eigenvalue analysis.

    The Condition Index is the square root of the ratio of max to min eigenvalues
    of the correlation matrix. It provides overall multicollinearity severity.
    """

    condition_index: float  # sqrt(max(eigenvalue) / min(eigenvalue))
    eigenvalues: list[float]  # All eigenvalues from correlation matrix
    has_multicollinearity: bool  # Condition Index > 30
    severity: Literal["none", "moderate", "severe"]  # CI: <30, 30-100, >100
    problematic_dimensions: int  # Count of near-zero eigenvalues

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of Condition Index."""
        if self.condition_index < 30:
            return "No significant table-level multicollinearity"
        elif self.condition_index < 100:
            return f"Moderate table-level multicollinearity (CI={self.condition_index:.1f})"
        else:
            return f"Severe table-level multicollinearity (CI={self.condition_index:.1f})"


class MulticollinearityAnalysis(BaseModel):
    """Complete multicollinearity analysis for a table.

    Combines column-level VIF/Tolerance with table-level Condition Index
    to provide comprehensive multicollinearity assessment.
    """

    table_id: str
    table_name: str
    computed_at: datetime

    # Column-level analysis
    column_vifs: list[ColumnVIF] = Field(default_factory=list)
    num_problematic_columns: int = 0  # VIF > 10

    # Table-level analysis
    condition_index: ConditionIndexAnalysis | None = None

    # Summary flags
    has_severe_multicollinearity: bool = False
    overall_severity: Literal["none", "moderate", "severe"] = "none"

    # Quality issues detected
    quality_issues: list[dict[str, Any]] = Field(default_factory=list)
