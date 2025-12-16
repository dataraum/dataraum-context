"""Statistical Profile Models.

Pydantic models for statistical profiling data structures:
- ColumnProfile: Complete statistical profile of a column
- NumericStats: Statistics for numeric columns
- StringStats: Statistics for string columns
- HistogramBucket: Histogram bin
- ValueCount: Frequency count for top values
- DetectedPattern: Pattern detection result (used by schema profiler)
- StatisticsProfileResult: Result of statistics profiling

NOTE: Statistical quality models (BenfordAnalysis, OutlierDetection, StatisticalQualityResult)
have been moved to quality/models.py since they represent quality assessment, not statistics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from dataraum_context.core.models.base import ColumnRef


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
    """A detected pattern in column values.

    Used by the schema profiler for pattern detection on raw tables.
    Patterns are stored in SchemaProfileResult.detected_patterns (dict by column name),
    NOT in ColumnProfile which is for statistics stage only.
    """

    name: str
    match_rate: float
    semantic_type: str | None = None


class ColumnProfile(BaseModel):
    """Statistical profile of a column.

    This model is for statistics stage (typed tables) only.
    Pattern detection is done in schema stage and stored separately.
    """

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


class StatisticsProfileResult(BaseModel):
    """Result of statistics profiling (typed stage, all stats).

    Contains all row-based statistics computed on clean typed data.
    Note: correlation_result is handled separately by analysis/correlation module.
    """

    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    duration_seconds: float
