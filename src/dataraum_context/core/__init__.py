"""Core module - configuration, connections, and shared models."""

from dataraum_context.core.config import Settings, get_settings
from dataraum_context.core.models import (
    Cardinality,
    ColumnContext,
    ColumnProfile,
    ColumnRef,
    ContextDocument,
    DataType,
    DecisionSource,
    MetricDefinition,
    QualitySeverity,
    Relationship,
    RelationshipType,
    Result,
    SemanticRole,
    TableContext,
    TypeCandidate,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Models - enums
    "Cardinality",
    "DataType",
    "DecisionSource",
    "QualitySeverity",
    "RelationshipType",
    "SemanticRole",
    # Models - data structures
    "ColumnContext",
    "ColumnProfile",
    "ColumnRef",
    "ContextDocument",
    "MetricDefinition",
    "Relationship",
    "Result",
    "TableContext",
    "TypeCandidate",
]
