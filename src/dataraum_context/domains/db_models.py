"""Generic domain quality metrics database model.

SQLAlchemy model for storing domain-specific quality metrics.
Domain-specific models (financial, marketing, etc.) are in their respective subfolders.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dataraum_context.storage import Base

if TYPE_CHECKING:
    from dataraum_context.storage import Table


class DomainQualityMetrics(Base):
    """Generic domain-specific quality metrics for a table.

    Stores quality metrics specific to a business domain (financial, marketing, etc.).
    The metrics field is a flexible JSONB storage for domain-specific data.

    For domain-specific structured storage, use the domain's db_models:
    - financial: quality.domains.financial.db_models.FinancialQualityMetrics
    """

    __tablename__ = "domain_quality_metrics"

    metric_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    table_id: Mapped[str] = mapped_column(String, ForeignKey("tables.table_id"))
    domain: Mapped[str] = mapped_column(String)  # 'financial', 'marketing', etc.
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Generic storage for domain-specific metrics
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Overall domain compliance
    domain_compliance_score: Mapped[float] = mapped_column(Float)
    violations: Mapped[list[dict[str, Any]]] = mapped_column(JSON)

    # Relationships
    table: Mapped[Table] = relationship(back_populates="domain_quality_metrics")
