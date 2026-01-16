"""Core module - configuration, connections, and shared models."""

from dataraum_context.core.config import Settings, get_settings
from dataraum_context.core.connections import (
    ConnectionConfig,
    ConnectionManager,
    close_default_manager,
    get_connection_manager,
)
from dataraum_context.core.models.base import (
    Cardinality,
    ColumnRef,
    DataType,
    DecisionSource,
    QualitySeverity,
    RelationshipType,
    Result,
    SemanticRole,
    SourceConfig,
    TableRef,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Connections
    "ConnectionConfig",
    "ConnectionManager",
    "close_default_manager",
    "get_connection_manager",
    # Models - enums
    "Cardinality",
    "DataType",
    "DecisionSource",
    "QualitySeverity",
    "RelationshipType",
    "SemanticRole",
    # Models - base data structures
    "ColumnRef",
    "Result",
    "SourceConfig",
    "TableRef",
]
