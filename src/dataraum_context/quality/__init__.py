"""Quality assessment framework (Pillar 5).

.. deprecated::
    This module is deprecated. Quality checks have been migrated to the graphs module:

    - Filter graphs: config/graphs/filters/rules/
        - role_based.yaml (key, timestamp, measure, foreign_key checks)
        - type_based.yaml (DOUBLE, DATE, VARCHAR checks)
        - pattern_based.yaml (email, url, phone checks)
        - consistency.yaml (cross-column checks)

    - Quality metrics: config/graphs/metrics/quality/
        - data_completeness.yaml
        - anomaly_rate.yaml
        - data_freshness.yaml

    - Context building: dataraum_context.graphs.context
        - build_execution_context() replaces quality/context.py
        - format_context_for_prompt() replaces quality/formatting/

    Migration guide:
        # Old way
        from dataraum_context.quality import synthesis
        from dataraum_context.quality.context import format_dataset_quality_context

        # New way
        from dataraum_context.graphs import (
            build_execution_context,
            format_context_for_prompt,
            GraphLoader,
        )

        loader = GraphLoader()
        loader.load_all()
        quality_metrics = loader.get_graphs_by_category("quality")

The quality module synthesizes quality metrics from all other pillars:
- Statistical quality (Pillar 1) - in analysis/statistics
- Topological quality (Pillar 2) - in analysis/topology
- Semantic quality (Pillar 3)
- Temporal quality (Pillar 4) - in analysis/temporal

Architecture:
    analysis/
      statistics/      # Statistical quality metrics (Benford, outliers)
      temporal/        # Temporal quality metrics
      topology/        # Topological quality metrics
      cycles/          # Business cycle detection (LLM agent)
      validation/      # Generic validation checks (LLM-powered)

    quality/           # DEPRECATED - see graphs/ module
      synthesis.py     # Aggregate quality from all pillars
      formatting/      # Format quality data for LLM context

Usage:
    # Statistical quality
    from dataraum_context.analysis.statistics import (
        assess_statistical_quality,
        BenfordAnalysis,
        StatisticalQualityResult,
    )

    # Topological quality
    from dataraum_context.analysis.topology import (
        analyze_topological_quality,
        BettiNumbers,
    )

    # Temporal quality
    from dataraum_context.analysis.temporal import profile_temporal

    # Business cycle detection
    from dataraum_context.analysis.cycles import (
        BusinessCycleAgent,
        BusinessCycleAnalysis,
    )

    # Generic validation (replaces domain-specific checks)
    from dataraum_context.analysis.validation import (
        ValidationAgent,
        load_all_validation_specs,
    )

    # Multi-pillar quality synthesis (DEPRECATED)
    from dataraum_context.quality import synthesis
"""

from __future__ import annotations

import warnings as _warnings

from dataraum_context.quality import synthesis

_warnings.warn(
    "The quality module is deprecated. Use dataraum_context.graphs for quality checks and metrics.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "synthesis",
]
