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
from dataraum_context.quality.formatting.config import (
    FormatterConfig,
    MetricGroupConfig,
    get_default_config,
    load_formatter_config,
)
from dataraum_context.quality.formatting.multicollinearity import (
    format_cross_table_multicollinearity_for_llm,
    format_multicollinearity_for_llm,
)

__all__ = [
    # Base utilities
    "CommonThresholds",
    "SeverityLevel",
    "ThresholdConfig",
    "THRESHOLDS",
    "format_list_with_overflow",
    "generate_interpretation",
    "generate_recommendation",
    "map_to_severity",
    "severity_emoji",
    # Configuration
    "FormatterConfig",
    "MetricGroupConfig",
    "get_default_config",
    "load_formatter_config",
    # Multicollinearity formatters
    "format_multicollinearity_for_llm",
    "format_cross_table_multicollinearity_for_llm",
]
