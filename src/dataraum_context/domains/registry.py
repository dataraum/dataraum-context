"""Domain analyzer registry.

Provides registration and lookup for domain analyzers.
Uses a decorator pattern for easy registration.

Usage:
    from dataraum_context.domains.registry import (
        register_domain,
        get_analyzer,
        list_domains,
    )

    # Register a domain (typically in the domain module)
    @register_domain("financial")
    class FinancialDomainAnalyzer(DomainAnalyzer):
        ...

    # Look up an analyzer
    analyzer = get_analyzer("financial")
    if analyzer:
        result = await analyzer.analyze(table_id, conn, session)

    # List available domains
    domains = list_domains()  # ["financial", ...]
"""

from collections.abc import Callable

from dataraum_context.domains.base import DomainAnalyzer

# Registry of domain analyzers (name -> class)
_DOMAIN_ANALYZERS: dict[str, type[DomainAnalyzer]] = {}


def register_domain(name: str) -> Callable[[type[DomainAnalyzer]], type[DomainAnalyzer]]:
    """Decorator to register a domain analyzer.

    Args:
        name: Unique domain identifier (e.g., "financial", "marketing")

    Returns:
        Decorator function that registers the class

    Example:
        @register_domain("financial")
        class FinancialDomainAnalyzer(DomainAnalyzer):
            ...
    """

    def decorator(cls: type[DomainAnalyzer]) -> type[DomainAnalyzer]:
        if name in _DOMAIN_ANALYZERS:
            raise ValueError(f"Domain '{name}' is already registered")
        _DOMAIN_ANALYZERS[name] = cls
        return cls

    return decorator


def get_analyzer(domain_name: str) -> DomainAnalyzer | None:
    """Get an analyzer instance for a domain.

    Args:
        domain_name: Domain identifier (e.g., "financial")

    Returns:
        DomainAnalyzer instance if registered, None otherwise
    """
    analyzer_cls = _DOMAIN_ANALYZERS.get(domain_name)
    if analyzer_cls:
        return analyzer_cls()
    return None


def list_domains() -> list[str]:
    """List all registered domain names.

    Returns:
        List of registered domain identifiers
    """
    return list(_DOMAIN_ANALYZERS.keys())


def get_all_analyzers() -> dict[str, DomainAnalyzer]:
    """Get instances of all registered analyzers.

    Returns:
        Dict mapping domain name to analyzer instance
    """
    return {name: cls() for name, cls in _DOMAIN_ANALYZERS.items()}
