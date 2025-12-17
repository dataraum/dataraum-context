"""Correlation analysis module.

Analyzes relationships between columns:
- Per-table: Numeric correlations, categorical associations, functional dependencies, derived columns

Cross-table relationship evaluation is in analysis/relationships/evaluator.py.
Quality-focused cross-table analysis (VDP) will be rebuilt in cross_table.py.
"""

# Algorithms (pure computation)
from dataraum_context.analysis.correlation.algorithms import (
    AssociationResult,
    CorrelationResult,
    DependencyGroupResult,
    MulticollinearityResult,
    compute_cramers_v,
    compute_multicollinearity,
    compute_pairwise_correlations,
)

# Per-table runners
from dataraum_context.analysis.correlation.categorical import compute_categorical_associations

# NOTE: Cross-table runner disabled - being replaced with per-relationship evaluation
# See analysis/relationships/evaluator.py
# DB Models
from dataraum_context.analysis.correlation.db_models import (
    CategoricalAssociation as DBCategoricalAssociation,
)
from dataraum_context.analysis.correlation.db_models import (
    ColumnCorrelation,
)
from dataraum_context.analysis.correlation.db_models import (
    DerivedColumn as DBDerivedColumn,
)
from dataraum_context.analysis.correlation.db_models import (
    FunctionalDependency as DBFunctionalDependency,
)
from dataraum_context.analysis.correlation.derived_columns import detect_derived_columns
from dataraum_context.analysis.correlation.functional_dependency import (
    detect_functional_dependencies,
)

# Pydantic Models
from dataraum_context.analysis.correlation.models import (
    CategoricalAssociation,
    CorrelationAnalysisResult,
    DerivedColumn,
    FunctionalDependency,
    NumericCorrelation,
)
from dataraum_context.analysis.correlation.numeric import compute_numeric_correlations
from dataraum_context.analysis.correlation.processor import analyze_correlations

__all__ = [
    # Main entry points
    "analyze_correlations",  # Per-table
    # NOTE: Cross-table functions disabled - see analysis/relationships/evaluator.py
    # Per-table analysis functions
    "compute_numeric_correlations",
    "compute_categorical_associations",
    "detect_functional_dependencies",
    "detect_derived_columns",
    # Algorithms (pure computation)
    "compute_pairwise_correlations",
    "compute_cramers_v",
    "compute_multicollinearity",
    "CorrelationResult",
    "AssociationResult",
    "DependencyGroupResult",
    "MulticollinearityResult",
    # DB Models
    "ColumnCorrelation",
    "DBCategoricalAssociation",
    "DBFunctionalDependency",
    "DBDerivedColumn",
    # Pydantic Models
    "NumericCorrelation",
    "CategoricalAssociation",
    "FunctionalDependency",
    "DerivedColumn",
    "CorrelationAnalysisResult",
]
