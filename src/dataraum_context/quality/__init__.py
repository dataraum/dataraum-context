"""Quality assessment framework (Pillar 5).

The quality module synthesizes quality metrics from all other pillars:
- Statistical quality (Pillar 1)
- Topological quality (Pillar 2)
- Semantic quality (Pillar 3)
- Temporal quality (Pillar 4)

Plus domain-specific quality rules for various business domains.

Architecture:
    quality/
      domains/          # Domain-specific quality rules
        financial.py    # Financial accounting
        marketing.py    # Marketing analytics
        ...
      base.py          # Base domain interface
      synthesis.py     # Aggregate quality from all pillars

Usage:
    # Domain-specific quality
    from dataraum_context.quality.domains import financial
    result = await financial.analyze_financial_quality(table_id, conn, session)

    # Multi-pillar quality synthesis (future)
    from dataraum_context.quality import synthesis
    quality_context = await synthesis.synthesize_quality_context(source_id, conn, session)
"""

from dataraum_context.quality import domains

__all__ = [
    "domains",
]
