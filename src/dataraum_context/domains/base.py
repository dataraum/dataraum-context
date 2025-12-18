"""Base classes for pluggable domain analyzers.

Defines the interface that all domain analyzers must implement.
This enables a strategy pattern for domain-specific quality checks.

Usage:
    from dataraum_context.domains.base import DomainAnalyzer
    from dataraum_context.domains.registry import register_domain

    @register_domain("my_domain")
    class MyDomainAnalyzer(DomainAnalyzer):
        @property
        def domain_name(self) -> str:
            return "my_domain"

        async def analyze(self, table_id, duckdb_conn, session, config=None):
            # ... domain-specific analysis ...
            return Result.ok(metrics)

        def get_issues(self, metrics):
            # ... extract issues from metrics ...
            return issues
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import duckdb
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result


class DomainConfig(Protocol):
    """Protocol for domain configuration.

    Each domain can define its own config class that follows this protocol.
    """

    @property
    def domain_name(self) -> str:
        """Domain identifier."""
        ...


class DomainAnalyzer(ABC):
    """Base class for domain-specific quality analyzers.

    Each domain (financial, marketing, healthcare, etc.) implements this interface.
    The analyzer receives pre-computed statistics and returns domain-specific
    quality assessments.

    The registry pattern allows domains to be added without modifying core code.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique identifier for this domain (e.g., 'financial', 'marketing').

        This name is used for:
        - Registry lookup
        - Database storage (domain field)
        - Configuration file naming
        """
        ...

    @abstractmethod
    async def analyze(
        self,
        table_id: str,
        duckdb_conn: duckdb.DuckDBPyConnection,
        session: AsyncSession,
        config: DomainConfig | None = None,
    ) -> Result[dict[str, Any]]:
        """Run domain-specific quality analysis.

        Args:
            table_id: Table to analyze
            duckdb_conn: DuckDB connection for data queries
            session: SQLAlchemy session for metadata access
            config: Optional domain-specific configuration

        Returns:
            Result containing domain-specific metrics dict.
            The dict should include at minimum:
            - "domain": str (domain name)
            - "table_id": str
            - "computed_at": str (ISO timestamp)
        """
        ...

    @abstractmethod
    def get_issues(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract quality issues from computed metrics.

        Analyzes the metrics dict returned by analyze() and extracts
        any quality issues that should be reported.

        Args:
            metrics: Computed metrics from analyze()

        Returns:
            List of issue dicts, each containing:
            - "type": str (issue type identifier)
            - "severity": str ("minor", "moderate", "severe", "critical")
            - "description": str (human-readable description)
            - "recommendation": str (optional, suggested fix)
        """
        ...

    def format_for_context(self, metrics: dict[str, Any]) -> str:
        """Format metrics for LLM context (optional override).

        Default implementation returns a simple summary.
        Domains can override for custom formatting.

        Args:
            metrics: Computed metrics from analyze()

        Returns:
            String representation suitable for LLM context
        """
        issues = self.get_issues(metrics)
        if not issues:
            return f"{self.domain_name.title()} domain: No quality issues detected."

        lines = [f"{self.domain_name.title()} domain: {len(issues)} issue(s) detected:"]
        for issue in issues[:5]:  # Limit to 5 for context
            lines.append(f"  - [{issue.get('severity', 'unknown')}] {issue.get('description', '')}")

        return "\n".join(lines)
