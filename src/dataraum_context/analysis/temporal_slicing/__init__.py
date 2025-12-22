"""Temporal slicing analysis module.

Provides temporal analysis capabilities for slice data including:
- Period completeness analysis (data cutoffs)
- Distribution drift detection (JS divergence)
- Cross-slice temporal comparison
- Volume anomaly detection
"""

from dataraum_context.analysis.temporal_slicing.analyzer import (
    TemporalSliceAnalyzer,
    TemporalSliceContext,
    aggregate_temporal_data,
    analyze_temporal_slices,
)
from dataraum_context.analysis.temporal_slicing.db_models import (
    SliceTimeMatrixEntry,
    TemporalDriftAnalysis,
    TemporalSliceAnalysis,
    TemporalSliceRun,
)
from dataraum_context.analysis.temporal_slicing.models import (
    AggregatedTemporalData,
    CompletenessResult,
    DistributionDriftResult,
    PeriodMetrics,
    SliceTimeCell,
    SliceTimeMatrix,
    TemporalAnalysisResult,
    TemporalSliceConfig,
    TimeGrain,
    VolumeAnomalyResult,
)

__all__ = [
    # Main entry points
    "analyze_temporal_slices",
    "aggregate_temporal_data",
    "TemporalSliceAnalyzer",
    "TemporalSliceContext",
    # Config
    "TemporalSliceConfig",
    "TimeGrain",
    # Models
    "PeriodMetrics",
    "CompletenessResult",
    "DistributionDriftResult",
    "SliceTimeCell",
    "SliceTimeMatrix",
    "VolumeAnomalyResult",
    "TemporalAnalysisResult",
    "AggregatedTemporalData",
    # DB Models
    "TemporalSliceAnalysis",
    "TemporalSliceRun",
    "TemporalDriftAnalysis",
    "SliceTimeMatrixEntry",
]
