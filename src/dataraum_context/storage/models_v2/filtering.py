"""Filtering persistence models.

SQLAlchemy models for persisting filtering recommendations and results
to enable audit trails and reproducibility.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dataraum_context.storage.models_v2.base import Base


class FilteringRecommendationRecord(Base):
    """Persisted filtering recommendation from LLM analysis.

    Stores the complete LLM-generated filtering recommendations including:
    - Scope filters (calculation row selection)
    - Quality filters (data cleaning)
    - Flags (issues that can't be filtered)
    - Calculation impacts
    - Business cycle context

    This provides audit trail for what filters were recommended and why.
    """

    __tablename__ = "filtering_recommendations"

    recommendation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Source tracking
    source: Mapped[str] = mapped_column(String, nullable=False)  # "llm", "merged", "user_rule"
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)

    # Confidence and acknowledgment
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    requires_acknowledgment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String, nullable=True)

    # Business cycle context
    business_cycles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Structured filters (JSONB)
    scope_filters: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    quality_filters: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    flags: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    calculation_impacts: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # Legacy format (backward compatibility)
    clean_view_filters: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    quarantine_criteria: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    column_exclusions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    rationale: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)


class FilteringExecutionRecord(Base):
    """Record of filter execution creating clean/quarantine artifacts.

    Stores the result of executing filtering recommendations:
    - What views/tables were created
    - Row counts (clean vs quarantine)
    - Applied filters
    - Execution timestamp

    This provides audit trail for what data was filtered and when.
    """

    __tablename__ = "filtering_executions"

    execution_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    recommendation_id: Mapped[str] = mapped_column(
        ForeignKey("filtering_recommendations.recommendation_id", ondelete="CASCADE"),
        nullable=False,
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )
    executed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Created artifacts
    clean_view_name: Mapped[str] = mapped_column(String, nullable=False)
    quarantine_table_name: Mapped[str] = mapped_column(String, nullable=False)

    # Row counts
    original_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    clean_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    quarantine_row_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Applied filters (may differ from recommendation if user rules merged)
    applied_filters: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    quarantine_reasons: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)

    # Execution status
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
