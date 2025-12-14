"""Domain-specific quality modules.

Each domain provides specialized quality checks and rules:

**Accounting Quality Checks:**
- financial: Double-entry, trial balance, sign conventions, fiscal periods

**Domain Rule Analysis (Layer 1.5):**
- financial_orchestrator: Fiscal stability, financial anomalies

**LLM-Enhanced Analysis:**
- financial_orchestrator: Complete flow with LLM cycle classification
- financial_llm: LLM classification and interpretation functions

Usage:
    # Accounting quality checks only
    from dataraum_context.quality.domains import financial
    result = await financial.analyze_financial_quality(table_id, conn, session)

    # LLM-enhanced analysis with domain rules (recommended)
    from dataraum_context.quality.domains.financial_orchestrator import (
        analyze_complete_financial_quality,
    )
    result = await analyze_complete_financial_quality(table_id, conn, session, llm_service)

    # Domain rule functions (deterministic, no LLM)
    from dataraum_context.quality.domains.financial_orchestrator import (
        assess_fiscal_stability,
        detect_financial_anomalies,
    )
"""

# Lazy imports to avoid circular dependency with llm module
# Import these functions directly from financial_orchestrator when needed
from typing import Any

from dataraum_context.quality.domains import financial


def __getattr__(name: str) -> Any:
    """Lazy import to avoid circular imports."""
    if name == "assess_fiscal_stability":
        from dataraum_context.quality.domains.financial_orchestrator import (
            assess_fiscal_stability,
        )

        return assess_fiscal_stability
    if name == "detect_financial_anomalies":
        from dataraum_context.quality.domains.financial_orchestrator import (
            detect_financial_anomalies,
        )

        return detect_financial_anomalies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "financial",
    "assess_fiscal_stability",
    "detect_financial_anomalies",
]
