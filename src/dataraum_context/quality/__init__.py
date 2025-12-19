"""Quality assessment framework (Pillar 5).

The quality module synthesizes quality metrics from all other pillars:
- Statistical quality (Pillar 1) - now in analysis/statistics
- Topological quality (Pillar 2) - now in analysis/topology
- Semantic quality (Pillar 3)
- Temporal quality (Pillar 4) - now in analysis/temporal

Plus domain-specific quality rules in dataraum_context.domains.

Architecture:
    analysis/
      statistics/      # Statistical quality metrics (Benford, outliers)
      temporal/        # Temporal quality metrics
      topology/        # Topological quality metrics

    domains/
      financial/       # Financial accounting quality rules

    quality/
      topological.py   # Re-export shim (use analysis.topology directly)
      synthesis.py     # Aggregate quality from all pillars (Phase 5)

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

    # Domain-specific quality (financial)
    from dataraum_context.domains.financial import (
        analyze_financial_quality,
        analyze_complete_financial_quality,
    )

    # Multi-pillar quality synthesis (Phase 5)
    from dataraum_context.quality import synthesis
    quality_context = await synthesis.synthesize_quality_context(source_id, conn, session)
"""

from dataraum_context.quality import synthesis, topological

__all__ = [
    "synthesis",
    "topological",
]
