"""Quality formatting utilities.

Provides reusable utilities for formatting quality metrics and issues
into structured, interpretable context for LLM consumption.
"""

from dataraum_context.quality.formatting.base import (
    THRESHOLDS,
    CommonThresholds,
    SeverityLevel,
    ThresholdConfig,
    format_list_with_overflow,
    generate_interpretation,
    generate_recommendation,
    map_to_severity,
    severity_emoji,
)

__all__ = [
    "CommonThresholds",
    "SeverityLevel",
    "ThresholdConfig",
    "THRESHOLDS",
    "format_list_with_overflow",
    "generate_interpretation",
    "generate_recommendation",
    "map_to_severity",
    "severity_emoji",
]
