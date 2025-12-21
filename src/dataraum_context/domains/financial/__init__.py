"""Financial domain quality module.

Provides financial accounting-specific quality checks:
- Double-entry balance validation
- Trial balance checks (Assets = Liabilities + Equity)
- Sign convention validation
- Fiscal period integrity

Usage:
    # Using the analyzer (recommended)
    from dataraum_context.domains.financial import FinancialDomainAnalyzer

    analyzer = FinancialDomainAnalyzer()
    result = await analyzer.analyze(table_id, conn, session)
    issues = analyzer.get_issues(result.unwrap())

    # Using the convenience function
    from dataraum_context.domains.financial import analyze_financial_quality

    result = await analyze_financial_quality(table_id, conn, session)

    # Using individual checks
    from dataraum_context.domains.financial.checks import (
        check_double_entry_balance,
        check_trial_balance,
    )

Note:
    Business cycle detection has been moved to dataraum_context.analysis.cycles
    which provides a generic, LLM-powered approach without hardcoded patterns.
"""

from dataraum_context.domains.financial.analyzer import (
    FinancialDomainAnalyzer,
    analyze_financial_quality,
)
from dataraum_context.domains.financial.checks import (
    check_double_entry_balance,
    check_fiscal_period_integrity,
    check_sign_conventions,
    check_trial_balance,
)
from dataraum_context.domains.financial.config import (
    clear_config_cache,
    load_financial_config,
)
from dataraum_context.domains.financial.db_models import (
    DoubleEntryCheck,
    FinancialQualityMetrics,
    FiscalPeriodIntegrity,
    SignConventionViolation,
    TrialBalanceCheck,
)
from dataraum_context.domains.financial.models import (
    DoubleEntryResult,
    FinancialQualityConfig,
    FinancialQualityIssue,
    FinancialQualityResult,
    FiscalPeriodIntegrityCheck,
    SignConventionConfig,
    TrialBalanceResult,
)
from dataraum_context.domains.financial.models import (
    SignConventionViolation as SignConventionViolationModel,
)

__all__ = [
    # Analyzer
    "FinancialDomainAnalyzer",
    "analyze_financial_quality",
    # Config
    "load_financial_config",
    "clear_config_cache",
    # Check functions
    "check_double_entry_balance",
    "check_trial_balance",
    "check_sign_conventions",
    "check_fiscal_period_integrity",
    # DB Models
    "FinancialQualityMetrics",
    "DoubleEntryCheck",
    "TrialBalanceCheck",
    "SignConventionViolation",
    "FiscalPeriodIntegrity",
    # Pydantic Models
    "FinancialQualityConfig",
    "SignConventionConfig",
    "DoubleEntryResult",
    "TrialBalanceResult",
    "SignConventionViolationModel",
    "FiscalPeriodIntegrityCheck",
    "FinancialQualityIssue",
    "FinancialQualityResult",
]
