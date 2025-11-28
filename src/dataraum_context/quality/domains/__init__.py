"""Domain-specific quality modules.

Each domain provides specialized quality checks and rules:
- financial: Accounting rules (double-entry, trial balance, sign conventions)
- marketing: Campaign metrics, conversion funnels, attribution
- manufacturing: Production quality, inventory accuracy, supply chain
- healthcare: Clinical data quality, regulatory compliance
- etc.

Usage:
    from dataraum_context.quality.domains import financial

    result = await financial.analyze_financial_quality(table_id, conn, session)
"""

from dataraum_context.quality.domains import financial

__all__ = [
    "financial",
]
