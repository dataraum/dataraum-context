"""Correlation analysis module.

Analyzes relationships between columns:
- Per-table: Numeric correlations, categorical associations, functional dependencies, derived columns
- Post-confirmation: Cross-table quality analysis (VDP, redundant/derived columns)

Main entry points:
- analyze_correlations: Within-table correlation analysis
- analyze_cross_table_quality: Cross-table quality analysis on confirmed relationships
"""

# Processors (main entry points)
# DB Models - Within-table
from dataraum_context.analysis.correlation.db_models import (
    CategoricalAssociation as DBCategoricalAssociation,
)

# DB Models - Cross-table quality
from dataraum_context.analysis.correlation.db_models import (
    ColumnCorrelation,
    CorrelationAnalysisRun,
    CrossTableCorrelationDB,
    MulticollinearityGroup,
    QualityIssueDB,
)
from dataraum_context.analysis.correlation.db_models import (
    DerivedColumn as DBDerivedColumn,
)
from dataraum_context.analysis.correlation.db_models import (
    FunctionalDependency as DBFunctionalDependency,
)

# Pydantic Models - Within-table
# Pydantic Models - Cross-table quality
from dataraum_context.analysis.correlation.models import (
    CategoricalAssociation,
    CorrelationAnalysisResult,
    CrossTableCorrelation,
    CrossTableQualityResult,
    DependencyGroup,
    DerivedColumn,
    DerivedColumnCandidate,
    EnrichedRelationship,
    FunctionalDependency,
    NumericCorrelation,
    QualityIssue,
    RedundantColumnPair,
)
from dataraum_context.analysis.correlation.processor import (
    analyze_correlations,
    analyze_cross_table_quality,
)

__all__ = [
    # Processors (main entry points)
    "analyze_correlations",
    "analyze_cross_table_quality",
    # DB Models - Within-table
    "ColumnCorrelation",
    "DBCategoricalAssociation",
    "DBFunctionalDependency",
    "DBDerivedColumn",
    # DB Models - Cross-table quality
    "CorrelationAnalysisRun",
    "CrossTableCorrelationDB",
    "MulticollinearityGroup",
    "QualityIssueDB",
    # Pydantic Models - Within-table
    "NumericCorrelation",
    "CategoricalAssociation",
    "FunctionalDependency",
    "DerivedColumn",
    "CorrelationAnalysisResult",
    # Pydantic Models - Cross-table quality
    "CrossTableQualityResult",
    "CrossTableCorrelation",
    "RedundantColumnPair",
    "DerivedColumnCandidate",
    "DependencyGroup",
    "QualityIssue",
    "EnrichedRelationship",
]
