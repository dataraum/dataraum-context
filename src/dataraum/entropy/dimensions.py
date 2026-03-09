"""Typed enums for entropy dimensions, analysis keys, and fix actions.

Provides compile-time safety for the string keys that couple pipeline
phases to entropy detectors.  All enums inherit from ``str`` so they
are drop-in replacements for bare strings in dict keys, comparisons,
and serialisation.
"""

from __future__ import annotations

from enum import Enum


class AnalysisKey(str, Enum):
    """Analysis outputs that phases produce and detectors consume."""

    TYPING = "typing"
    STATISTICS = "statistics"
    SEMANTIC = "semantic"
    RELATIONSHIPS = "relationships"
    CORRELATION = "correlation"
    DRIFT_SUMMARIES = "drift_summaries"
    SLICE_VARIANCE = "slice_variance"
    COLUMN_QUALITY_REPORTS = "column_quality_reports"
    ENRICHED_VIEW = "enriched_view"


class SubDimension(str, Enum):
    """Entropy sub-dimensions measured by detectors."""

    TYPE_FIDELITY = "type_fidelity"
    JOIN_PATH_DETERMINISM = "join_path_determinism"
    RELATIONSHIP_QUALITY = "relationship_quality"
    NULL_RATIO = "null_ratio"
    OUTLIER_RATE = "outlier_rate"
    BENFORD_COMPLIANCE = "benford_compliance"
    TEMPORAL_DRIFT = "temporal_drift"
    FORMULA_MATCH = "formula_match"
    NAMING_CLARITY = "naming_clarity"
    UNIT_DECLARATION = "unit_declaration"
    TIME_ROLE = "time_role"
    CROSS_COLUMN_PATTERNS = "cross_column_patterns"
    COLUMN_QUALITY = "column_quality"
    DIMENSION_COVERAGE = "dimension_coverage"


class FixAction(str, Enum):
    """Actions that detectors can flag as fixable."""

    TRANSFORM_EXCLUDE_OUTLIERS = "transform_exclude_outliers"
