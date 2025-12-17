"""Temporal analysis module.

Provides comprehensive temporal analysis for time columns including:
- Basic detection: granularity, gaps, completeness
- Pattern analysis: seasonality, trends, change points
- Quality assessment: staleness, distribution stability
- Fiscal calendar detection

Main Entry Points:
- profile_temporal(table_id, ...): Profile all temporal columns in a table
- analyze_temporal(): Single column analysis (legacy)

Example:
    from dataraum_context.analysis.temporal import profile_temporal

    result = await profile_temporal(table_id, duckdb_conn, session)
    if result.success:
        profile_result = result.unwrap()
        for profile in profile_result.column_profiles:
            print(f"{profile.column_name}: {profile.detected_granularity}")
"""

# Main entry points
# DB Models
from dataraum_context.analysis.temporal.db_models import (
    TemporalColumnProfile,
)
from dataraum_context.analysis.temporal.db_models import (
    TemporalTableSummary as DBTemporalTableSummary,
)

# Detection functions
from dataraum_context.analysis.temporal.detection import (
    analyze_basic_temporal,
    calculate_expected_periods,
    infer_granularity,
)

# Models
from dataraum_context.analysis.temporal.models import (
    ChangePointResult,
    DistributionShiftResult,
    DistributionStabilityAnalysis,
    FiscalCalendarAnalysis,
    SeasonalDecompositionResult,
    SeasonalityAnalysis,
    TemporalAnalysisResult,
    TemporalCompletenessAnalysis,
    TemporalEnrichmentResult,
    TemporalGapInfo,
    TemporalProfileResult,
    TemporalQualityIssue,
    TemporalTableSummary,
    TrendAnalysis,
    UpdateFrequencyAnalysis,
)

# Pattern analysis functions
from dataraum_context.analysis.temporal.patterns import (
    analyze_distribution_stability,
    analyze_seasonality,
    analyze_trend,
    analyze_update_frequency,
    detect_change_points,
    detect_fiscal_calendar,
)
from dataraum_context.analysis.temporal.processor import (
    analyze_temporal,  # Legacy, single column
    profile_temporal,  # New, table-level
)

# Legacy aliases for backwards compatibility
TemporalAnalysisMetrics = TemporalColumnProfile
TemporalTableSummaryMetrics = DBTemporalTableSummary

__all__ = [
    # Main entry points
    "profile_temporal",  # New primary entry point
    "analyze_temporal",  # Legacy single column
    # Detection
    "infer_granularity",
    "calculate_expected_periods",
    "analyze_basic_temporal",
    # Pattern analysis
    "analyze_seasonality",
    "analyze_trend",
    "detect_change_points",
    "analyze_update_frequency",
    "detect_fiscal_calendar",
    "analyze_distribution_stability",
    # Models
    "TemporalGapInfo",
    "TemporalCompletenessAnalysis",
    "SeasonalDecompositionResult",
    "SeasonalityAnalysis",
    "TrendAnalysis",
    "ChangePointResult",
    "UpdateFrequencyAnalysis",
    "FiscalCalendarAnalysis",
    "DistributionShiftResult",
    "DistributionStabilityAnalysis",
    "TemporalQualityIssue",
    "TemporalAnalysisResult",
    "TemporalTableSummary",
    "TemporalProfileResult",
    "TemporalEnrichmentResult",  # Deprecated
    # DB Models
    "TemporalColumnProfile",
    "DBTemporalTableSummary",
    # Legacy aliases
    "TemporalAnalysisMetrics",  # Alias for TemporalColumnProfile
    "TemporalTableSummaryMetrics",  # Alias for DBTemporalTableSummary
]
