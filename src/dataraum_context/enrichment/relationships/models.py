"""Relationship models.

Shared Pydantic models for relationship operations.
"""

from typing import Any

from pydantic import BaseModel

from dataraum_context.core.models.base import Cardinality, RelationshipType


class EnrichedRelationship(BaseModel):
    """Relationship enriched with column and table metadata for join construction.

    This model extends the basic relationship with human-readable names
    and additional metadata needed for building SQL joins and analysis.
    """

    relationship_id: str
    from_table: str
    from_column: str
    from_column_id: str
    from_table_id: str
    to_table: str
    to_column: str
    to_column_id: str
    to_table_id: str
    relationship_type: RelationshipType
    cardinality: Cardinality | None = None
    confidence: float
    detection_method: str
    evidence: dict[str, Any] = {}


class GraphAnalysisResult(BaseModel):
    """Result of relationship graph analysis.

    Contains cycle detection and connectivity metrics.
    """

    cycles: list[list[str]]
    """List of cycles, each cycle is a list of table IDs."""

    betti_0: int
    """Number of connected components (Betti-0 number)."""

    cycle_count: int
    """Total number of cycles detected."""

    node_count: int
    """Number of nodes (tables) in the graph."""

    edge_count: int
    """Number of edges (relationships) in the graph."""
