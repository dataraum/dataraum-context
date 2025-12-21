"""Quality assessment framework (Pillar 5).

The quality module synthesizes quality metrics from all other pillars:
- Statistical quality (Pillar 1) - in analysis/statistics
- Topological quality (Pillar 2) - in analysis/topology
- Semantic quality (Pillar 3)
- Temporal quality (Pillar 4) - in analysis/temporal

Plus domain-specific quality rules in dataraum_context.domains.

Architecture:
    analysis/
      statistics/      # Statistical quality metrics (Benford, outliers)
      temporal/        # Temporal quality metrics
      topology/        # Topological quality metrics
      cycles/          # Business cycle detection (LLM agent)

    domains/
      financial/       # Financial accounting quality rules

    quality/
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

    # Domain-specific quality (financial)
    from dataraum_context.domains.financial import (
        analyze_financial_quality,
    )

    # Multi-pillar quality synthesis
    from dataraum_context.quality import synthesis
"""

from dataraum_context.quality import synthesis

__all__ = [
    "synthesis",
]
