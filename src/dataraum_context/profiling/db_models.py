"""Profiling Database Models.

SQLAlchemy models for profiling-related persistence:
- Statistical profiles and quality metrics
- Type inference (candidates and decisions)
- Correlation analysis (numeric, categorical, functional dependencies)
- Multicollinearity analysis
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dataraum_context.storage.models_v2.base import Base

if TYPE_CHECKING:
    from dataraum_context.storage.models_v2.core import Column


# =============================================================================
# Statistical Context Models (Pillar 1)
# =============================================================================


class StatisticalProfile(Base):
    """Statistical profile of a column.

    HYBRID STORAGE APPROACH:
    - Structured fields: Queryable core dimensions (counts, ratios, flags)
    - JSONB field: Full Pydantic ColumnProfile model for flexibility

    This allows:
    - Fast queries on core metrics (null_ratio, cardinality_ratio)
    - Schema flexibility for experimentation
    - Zero mapping code (Pydantic handles serialization)
    """

    __tablename__ = "statistical_profiles"

    profile_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )
    profiled_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Layer indicator: "raw" or "typed"
    # Determines which stage produced this profile
    layer: Mapped[str] = mapped_column(String, nullable=False, default="raw")

    # STRUCTURED: Queryable core dimensions
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    null_count: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_count: Mapped[int | None] = mapped_column(Integer)
    null_ratio: Mapped[float | None] = mapped_column(Float)
    cardinality_ratio: Mapped[float | None] = mapped_column(Float)

    # Flags for filtering (fast queries)
    is_unique: Mapped[bool | None] = mapped_column(Integer)  # All values unique (potential PK)
    is_numeric: Mapped[bool | None] = mapped_column(Integer)  # Has numeric stats

    # JSONB: Full Pydantic ColumnProfile model
    # Stores: numeric_stats, string_stats, histogram, top_values, detected_patterns
    profile_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Relationships
    column: Mapped[Column] = relationship(back_populates="statistical_profiles")


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


# =============================================================================
# Type Inference Models
# =============================================================================


class TypeCandidate(Base):
    """Type candidates from pattern detection.

    Each column may have multiple type candidates with different
    confidence scores based on pattern matching and parsing success.
    """

    __tablename__ = "type_candidates"

    candidate_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Type candidate
    data_type: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    parse_success_rate: Mapped[float | None] = mapped_column(Float)
    failed_examples: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Pattern info
    detected_pattern: Mapped[str | None] = mapped_column(String)
    pattern_match_rate: Mapped[float | None] = mapped_column(Float)

    # Unit detection (from Pint)
    detected_unit: Mapped[str | None] = mapped_column(String)
    unit_confidence: Mapped[float | None] = mapped_column(Float)

    # Relationships
    column: Mapped[Column] = relationship(back_populates="type_candidates")


class TypeDecision(Base):
    """Type decisions (human-reviewable).

    Final type decision for a column after inference and optional human review.
    One decision per column.
    """

    __tablename__ = "type_decisions"
    __table_args__ = (UniqueConstraint("column_id", name="uq_column_type_decision"),)

    decision_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )

    decided_type: Mapped[str] = mapped_column(String, nullable=False)
    decision_source: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'automatic', 'manual', 'override'
    decided_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    decided_by: Mapped[str | None] = mapped_column(String)

    # Audit trail
    previous_type: Mapped[str | None] = mapped_column(String)
    decision_reason: Mapped[str | None] = mapped_column(String)

    # Relationships
    column: Mapped[Column] = relationship(back_populates="type_decision")


# =============================================================================
# Correlation and Dependency Models
# =============================================================================


class ColumnCorrelation(Base):
    """Correlation between two numeric columns.

    Stores both Pearson (linear) and Spearman (monotonic) correlations.
    Only stores correlations above a configurable threshold to avoid noise.
    """

    __tablename__ = "column_correlations"

    correlation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )

    # Column pair (order doesn't matter, but we store both for query convenience)
    column1_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )
    column2_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )

    # Pearson correlation (linear relationship)
    pearson_r: Mapped[float | None] = mapped_column(Float)
    pearson_p_value: Mapped[float | None] = mapped_column(Float)

    # Spearman correlation (monotonic relationship)
    spearman_rho: Mapped[float | None] = mapped_column(Float)
    spearman_p_value: Mapped[float | None] = mapped_column(Float)

    # Metadata
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Interpretation
    correlation_strength: Mapped[str | None] = mapped_column(
        String
    )  # 'none', 'weak', 'moderate', 'strong', 'very_strong'
    is_significant: Mapped[bool | None] = mapped_column(Integer)  # p_value < 0.05


class CategoricalAssociation(Base):
    """Association between two categorical columns using Cramér's V.

    Cramér's V is based on chi-square test and measures association strength
    between categorical variables (0 = no association, 1 = perfect association).
    """

    __tablename__ = "categorical_associations"

    association_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )

    # Column pair
    column1_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )
    column2_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )

    # Cramér's V statistic (0 to 1)
    cramers_v: Mapped[float] = mapped_column(Float, nullable=False)

    # Chi-square test results
    chi_square: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    degrees_of_freedom: Mapped[int] = mapped_column(Integer, nullable=False)

    # Metadata
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Interpretation
    association_strength: Mapped[str | None] = mapped_column(
        String
    )  # 'none', 'weak', 'moderate', 'strong'
    is_significant: Mapped[bool | None] = mapped_column(Integer)  # p_value < 0.05


class FunctionalDependency(Base):
    """Functional dependency: determinant → dependent.

    A functional dependency A → B means that for each value of A,
    there is exactly one value of B (or very close to exact with confidence).

    Can be single-column (A → B) or multi-column ((A, B) → C).
    """

    __tablename__ = "functional_dependencies"

    dependency_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )

    # Determinant (left side) - list of column IDs
    determinant_column_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    # Dependent (right side) - single column
    dependent_column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )

    # Confidence (1.0 = exact FD, < 1.0 = approximate FD)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Evidence
    unique_determinant_values: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # Count of unique determinant combinations
    violation_count: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # How many determinants map to multiple dependents

    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Example of the dependency
    example: Mapped[dict[str, Any] | None] = mapped_column(
        JSON
    )  # {determinant_values: [...], dependent_value: ...}


class DerivedColumn(Base):
    """Detected derived column (computed from other columns).

    Examples:
    - col3 = col1 + col2 (arithmetic)
    - col2 = UPPER(col1) (transformation)
    - col3 = CONCAT(col1, col2) (string operation)
    """

    __tablename__ = "derived_columns"

    derived_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )

    # The derived column
    derived_column_id: Mapped[str] = mapped_column(
        ForeignKey("columns.column_id", ondelete="CASCADE"), nullable=False
    )

    # Source columns (list of column IDs)
    source_column_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    # Derivation type
    derivation_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'sum', 'difference', 'product', 'ratio', 'concat', 'upper', 'lower', 'substr', etc.

    # Formula (human-readable)
    formula: Mapped[str] = mapped_column(
        String, nullable=False
    )  # e.g., "col_a + col_b", "UPPER(col_a)"

    # Match rate (how often the formula holds)
    match_rate: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Evidence
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    matching_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    mismatch_examples: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSON
    )  # Sample of rows where formula doesn't hold


# =============================================================================
# Multicollinearity Models
# =============================================================================


class MulticollinearityMetrics(Base):
    """Multicollinearity analysis results (hybrid storage).

    HYBRID STORAGE APPROACH:
    - Structured fields: Queryable quality indicators (flags, scores, key metrics)
    - JSONB field: Full MulticollinearityAnalysis Pydantic model for flexibility

    Multicollinearity measures redundancy between columns:
    - VIF (Variance Inflation Factor): Column-level metric
    - Tolerance (1/VIF): Simpler interpretation
    - Condition Index: Table-level severity via eigenvalue analysis

    This is a structural property of the data (computed during profiling),
    not a quality issue per se, though it informs quality assessment.
    """

    __tablename__ = "multicollinearity_metrics"

    metric_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    table_id: Mapped[str] = mapped_column(
        ForeignKey("tables.table_id", ondelete="CASCADE"), nullable=False
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # STRUCTURED: Queryable flags for filtering/sorting
    has_severe_multicollinearity: Mapped[bool | None] = mapped_column(Integer)
    num_problematic_columns: Mapped[int | None] = mapped_column(Integer)
    condition_index: Mapped[float | None] = mapped_column(Float)
    max_vif: Mapped[float | None] = mapped_column(Float)

    # JSONB: Full MulticollinearityAnalysis Pydantic model
    # Stores: column_vifs (list[ColumnVIF]), condition_index details, quality_issues
    analysis_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class CrossTableMulticollinearityMetrics(Base):
    """Cross-table multicollinearity analysis results (hybrid storage).

    HYBRID STORAGE APPROACH:
    - Structured fields: Queryable dimensions for filtering/sorting
    - JSONB field: Full CrossTableMulticollinearityAnalysis Pydantic model

    Extends single-table multicollinearity to detect dependencies across
    related tables using a unified correlation matrix spanning all involved
    columns from semantic + topology enrichment.

    Per-dataset storage: One record represents analysis across multiple tables.
    """

    __tablename__ = "cross_table_multicollinearity_metrics"

    metric_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Scope (multiple tables) - stored as JSON array
    table_ids: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # {"table_ids": [...]}

    # STRUCTURED: Queryable dimensions for filtering
    overall_condition_index: Mapped[float | None] = mapped_column(Float)
    num_cross_table_groups: Mapped[int | None] = mapped_column(Integer)
    num_total_groups: Mapped[int | None] = mapped_column(Integer)
    has_severe_cross_table_dependencies: Mapped[bool | None] = mapped_column(Boolean)
    total_columns_analyzed: Mapped[int | None] = mapped_column(Integer)
    total_relationships_used: Mapped[int | None] = mapped_column(Integer)

    # JSONB: Full CrossTableMulticollinearityAnalysis Pydantic model
    # Stores: dependency_groups, cross_table_groups, join_paths, quality_issues
    analysis_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


# =============================================================================
# Indexes for efficient queries
# =============================================================================

# Statistical
Index(
    "idx_statistical_profiles_column",
    StatisticalProfile.column_id,
    StatisticalProfile.profiled_at.desc(),
)
Index(
    "idx_statistical_quality_column",
    StatisticalQualityMetrics.column_id,
    StatisticalQualityMetrics.computed_at.desc(),
)

# Type inference
Index("idx_type_candidates_column", TypeCandidate.column_id)

# Correlations
Index("idx_correlations_table", ColumnCorrelation.table_id)
Index("idx_correlations_col1", ColumnCorrelation.column1_id)
Index("idx_correlations_col2", ColumnCorrelation.column2_id)
Index("idx_associations_table", CategoricalAssociation.table_id)
Index("idx_associations_col1", CategoricalAssociation.column1_id)
Index("idx_associations_col2", CategoricalAssociation.column2_id)
Index("idx_dependencies_table", FunctionalDependency.table_id)
Index("idx_dependencies_dependent", FunctionalDependency.dependent_column_id)
Index("idx_derived_table", DerivedColumn.table_id)
Index("idx_derived_column", DerivedColumn.derived_column_id)

# Multicollinearity
Index(
    "idx_multicollinearity_severity",
    MulticollinearityMetrics.has_severe_multicollinearity,
    MulticollinearityMetrics.num_problematic_columns,
)
Index(
    "idx_multicollinearity_table",
    MulticollinearityMetrics.table_id,
    MulticollinearityMetrics.computed_at.desc(),
)
Index(
    "idx_cross_multicollinearity_severity",
    CrossTableMulticollinearityMetrics.has_severe_cross_table_dependencies,
    CrossTableMulticollinearityMetrics.num_cross_table_groups,
)
