"""Pluggable domain-specific modules.

Provides a registry-based system for domain analyzers:
- Base classes for creating domain analyzers
- Registry for domain lookup and discovery
- Financial domain implementation

Usage:
    # Get an analyzer by name
    from dataraum_context.domains import get_analyzer

    analyzer = get_analyzer("financial")
    if analyzer:
        result = await analyzer.analyze(table_id, conn, session)
        issues = analyzer.get_issues(result.unwrap())

    # List available domains
    from dataraum_context.domains import list_domains

    available = list_domains()  # ["financial", ...]

    # Use financial domain directly
    from dataraum_context.domains.financial import analyze_financial_quality

    result = await analyze_financial_quality(table_id, conn, session)

    # Create a custom domain analyzer
    from dataraum_context.domains import DomainAnalyzer, register_domain

    @register_domain("my_domain")
    class MyDomainAnalyzer(DomainAnalyzer):
        ...
"""

# Import financial to register the analyzer
from dataraum_context.domains import financial  # noqa: F401
from dataraum_context.domains.base import DomainAnalyzer, DomainConfig
from dataraum_context.domains.registry import (
    get_all_analyzers,
    get_analyzer,
    list_domains,
    register_domain,
)

__all__ = [
    # Base classes
    "DomainAnalyzer",
    "DomainConfig",
    # Registry functions
    "register_domain",
    "get_analyzer",
    "list_domains",
    "get_all_analyzers",
    # Domains (for backward compatibility)
    "financial",
]
