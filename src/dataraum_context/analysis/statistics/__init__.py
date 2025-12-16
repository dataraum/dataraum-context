"""Statistical profiling module.

Computes column-level statistics on typed data:
- Basic counts (total, null, distinct, cardinality)
- String stats (min/max/avg length)
- Top values (frequency analysis)
- Numeric stats (min, max, mean, stddev, skewness, kurtosis, cv)
- Percentiles
- Histograms

NOTE: Statistical quality models (BenfordAnalysis, OutlierDetection, StatisticalQualityResult,
StatisticalQualityMetrics) are in quality/models.py and quality/db_models.py - they represent
quality assessment, not statistics computation.
"""

from dataraum_context.analysis.statistics.db_models import StatisticalProfile
from dataraum_context.analysis.statistics.models import (
    ColumnProfile,
    DetectedPattern,
    HistogramBucket,
    NumericStats,
    StatisticsProfileResult,
    StringStats,
    ValueCount,
)
from dataraum_context.analysis.statistics.processor import profile_statistics

__all__ = [
    # Main entry point
    "profile_statistics",
    # DB Models
    "StatisticalProfile",
    # Pydantic Models
    "ColumnProfile",
    "NumericStats",
    "StringStats",
    "HistogramBucket",
    "ValueCount",
    "DetectedPattern",
    "StatisticsProfileResult",
]
