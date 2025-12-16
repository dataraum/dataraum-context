"""Quality Database Models.

SQLAlchemy models for quality framework:
- Statistical quality metrics (Benford's Law, outlier detection)
- Quality rules (LLM-generated or manual)
- Quality results (rule execution results)

Note: QualityScore was pruned (unused).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dataraum_context.storage import Base

if TYPE_CHECKING:
    from dataraum_context.storage import Column, Table


# =============================================================================
# Statistical Quality Models
# =============================================================================


class StatisticalQualityMetrics(Base):
    """Statistical quality assessment for a column.

    HYBRID STORAGE APPROACH:
    - Structured fields: Queryable quality indicators (flags, scores, key ratios)
    - JSONB field: Full quality analysis results for flexibility

    Advanced quality metrics that may be expensive to compute:
    - Benford's Law compliance (fraud detection for financial amounts)
    - Outlier detection (Isolation Forest + IQR method)

    Note: Distribution stability (KS test) is handled by temporal quality module.
    """

    __tablename__ = "statistical_quality_metrics"

    metric_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # STRUCTURED: Queryable quality indicators
    # Flags for filtering (fast queries)
    benford_compliant: Mapped[bool | None] = mapped_column(Integer)
    has_outliers: Mapped[bool | None] = mapped_column(Integer)

    # Key metrics for sorting/filtering
    iqr_outlier_ratio: Mapped[float | None] = mapped_column(Float)
    isolation_forest_anomaly_ratio: Mapped[float | None] = mapped_column(Float)

    # JSONB: Full quality analysis results
    # Stores: Benford analysis, outlier details, quality issues
    quality_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Relationships
    column: Mapped[Column] = relationship(back_populates="statistical_quality_metrics")


Index(
    "idx_statistical_quality_column",
    StatisticalQualityMetrics.column_id,
    StatisticalQualityMetrics.computed_at.desc(),
)


# =============================================================================
# Rule-Based Quality Models
# =============================================================================


class QualityRule(Base):
    """Quality rules (LLM-generated or manual).

    Defines data quality rules that can be executed against tables/columns.
    """

    __tablename__ = "quality_rules"

    rule_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))

    # Scope
    table_id: Mapped[str] = mapped_column(ForeignKey("tables.table_id"), nullable=False)
    column_id: Mapped[str | None] = mapped_column(
        ForeignKey("columns.column_id")
    )  # NULL = table-level rule

    # Rule definition
    rule_name: Mapped[str] = mapped_column(String, nullable=False)
    rule_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'range', 'regex', 'uniqueness', 'referential', 'custom'
    rule_expression: Mapped[str | None] = mapped_column(
        Text
    )  # SQL WHERE clause or other expression
    rule_parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Metadata
    severity: Mapped[str] = mapped_column(
        String, default="warning"
    )  # 'info', 'warning', 'error', 'critical'
    source: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'llm', 'manual', 'config', 'heuristic'
    description: Mapped[str | None] = mapped_column(Text)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    created_by: Mapped[str | None] = mapped_column(String)

    # Relationships
    table: Mapped[Table] = relationship(back_populates="quality_rules")
    column: Mapped[Column | None] = relationship(back_populates="quality_rules")
    results: Mapped[list[QualityResult]] = relationship(
        back_populates="rule", cascade="all, delete-orphan"
    )


Index("idx_quality_rules_table", QualityRule.table_id)
Index("idx_quality_rules_column", QualityRule.column_id)


class QualityResult(Base):
    """Quality rule execution results.

    Records the outcome of executing a quality rule.
    """

    __tablename__ = "quality_results"

    result_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("quality_rules.rule_id", ondelete="CASCADE"), nullable=False
    )

    executed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Results
    total_records: Mapped[int | None] = mapped_column(Integer)
    passed_records: Mapped[int | None] = mapped_column(Integer)
    failed_records: Mapped[int | None] = mapped_column(Integer)
    pass_rate: Mapped[float | None] = mapped_column(Float)

    # Failure details
    failure_samples: Mapped[dict[str, Any] | None] = mapped_column(
        JSON
    )  # Sample of failing records for investigation

    # Trend
    previous_pass_rate: Mapped[float | None] = mapped_column(Float)
    trend_direction: Mapped[str | None] = mapped_column(
        String
    )  # 'improving', 'stable', 'degrading'

    # Relationships
    rule: Mapped[QualityRule] = relationship(back_populates="results")
