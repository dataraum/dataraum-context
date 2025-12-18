"""Domain-specific quality models - backward compatibility re-exports.

DEPRECATED: Import directly from the domain's models module instead:
    from dataraum_context.domains.financial.models import FinancialQualityConfig

This file re-exports financial models for backward compatibility.
"""

# Re-export financial models for backward compatibility
from dataraum_context.domains.financial.models import (
    DoubleEntryResult,
    FinancialQualityConfig,
    FinancialQualityIssue,
    FinancialQualityResult,
    FiscalPeriodIntegrityCheck,
    SignConventionConfig,
    SignConventionViolation,
    TrialBalanceResult,
)

__all__ = [
    "DoubleEntryResult",
    "TrialBalanceResult",
    "SignConventionViolation",
    "FiscalPeriodIntegrityCheck",
    "FinancialQualityIssue",
    "FinancialQualityResult",
    "SignConventionConfig",
    "FinancialQualityConfig",
]
