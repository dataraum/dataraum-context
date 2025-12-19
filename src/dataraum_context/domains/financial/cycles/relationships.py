"""Financial relationship gathering and graph analysis.

Load relationships from the database and analyze the graph structure
to detect business cycles in financial data.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import networkx as nx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.analysis.relationships.db_models import Relationship as RelationshipModel
from dataraum_context.core.models.base import Cardinality, RelationshipType
from dataraum_context.storage import Column, Table

# =============================================================================
# Models
# =============================================================================


class EnrichedRelationship(BaseModel):
    """Relationship with resolved table/column names for display."""

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
    cardinality: Cardinality | None
    confidence: float
    detection_method: str
    evidence: dict[str, Any]


class GraphAnalysisResult(BaseModel):
    """Result of relationship graph analysis."""

    cycles: list[list[str]]
    """List of cycles, each cycle is a list of table IDs."""

    betti_0: int
    """Number of connected components (Betti-0 number)."""

    cycle_count: int
    """Total number of cycles detected."""

    node_count: int = 0
    """Number of nodes (tables) in the graph."""

    edge_count: int = 0
    """Number of edges (relationships) in the graph."""


class RelationshipLike(Protocol):
    """Protocol for relationship objects (duck typing)."""

    from_table_id: str
    to_table_id: str
    confidence: float
    relationship_type: str | Any
    cardinality: str | Any | None


# =============================================================================
# Confidence thresholds for financial analysis
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "foreign_key": 0.7,  # Stricter (high reliability expected)
    "semantic": 0.6,  # Medium (LLM-based, good but not perfect)
    "correlation": 0.5,  # More permissive (TDA-detected)
    "hierarchy": 0.6,  # Medium (similar to semantic)
}


# =============================================================================
# Relationship Gathering
# =============================================================================


async def gather_relationships(
    table_ids: list[str],
    session: AsyncSession,
    *,
    detection_method: str = "llm",
) -> list[EnrichedRelationship]:
    """Gather relationships confirmed by LLM semantic enrichment.

    Strategy:
    - Query relationships table for all combinations of input tables
    - Filter by detection_method (default: "llm" for LLM-confirmed relationships)
    - Filter by differentiated confidence thresholds (FK: 0.7, Semantic: 0.6, etc.)
    - Merge/dedupe relationships from multiple sources
    - Resolve conflicts (prefer higher confidence, prefer FK over correlation)

    Args:
        table_ids: List of table IDs to analyze
        session: Async database session
        detection_method: Filter by detection method (default: "llm" for LLM-confirmed)

    Returns:
        List of enriched relationships with metadata
    """
    # Build query for relationships between any pair of input tables
    # Filter by detection_method to get only LLM-confirmed relationships
    stmt = (
        select(RelationshipModel)
        .where(
            (RelationshipModel.from_table_id.in_(table_ids))
            & (RelationshipModel.to_table_id.in_(table_ids))
            & (RelationshipModel.detection_method == detection_method)
        )
        .order_by(RelationshipModel.confidence.desc())
    )

    result = await session.execute(stmt)
    db_relationships = result.scalars().all()

    # Convert to enriched format with additional metadata
    enriched = []
    seen_pairs: set[tuple[str, str]] = set()

    for db_rel in db_relationships:
        # Apply differentiated confidence threshold
        threshold = CONFIDENCE_THRESHOLDS.get(db_rel.relationship_type, 0.5)
        if db_rel.confidence < threshold:
            continue

        # Check BOTH directions for duplicates
        pair_forward = (db_rel.from_column_id, db_rel.to_column_id)
        pair_reverse = (db_rel.to_column_id, db_rel.from_column_id)

        if pair_forward in seen_pairs or pair_reverse in seen_pairs:
            continue

        # Load column metadata for join construction
        from_col = await session.get(Column, db_rel.from_column_id)
        to_col = await session.get(Column, db_rel.to_column_id)
        from_table = await session.get(Table, db_rel.from_table_id)
        to_table = await session.get(Table, db_rel.to_table_id)

        if from_col is None or to_col is None or from_table is None or to_table is None:
            continue

        enriched.append(
            EnrichedRelationship(
                relationship_id=db_rel.relationship_id,
                from_table=from_table.table_name,
                from_column=from_col.column_name,
                from_column_id=db_rel.from_column_id,
                from_table_id=db_rel.from_table_id,
                to_table=to_table.table_name,
                to_column=to_col.column_name,
                to_column_id=db_rel.to_column_id,
                to_table_id=db_rel.to_table_id,
                relationship_type=RelationshipType(db_rel.relationship_type),
                cardinality=Cardinality(db_rel.cardinality) if db_rel.cardinality else None,
                confidence=db_rel.confidence,
                detection_method=db_rel.detection_method,
                evidence=db_rel.evidence or {},
            )
        )

        seen_pairs.add(pair_forward)
        seen_pairs.add(pair_reverse)

    return enriched


# =============================================================================
# Graph Analysis
# =============================================================================


def build_relationship_graph(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> nx.DiGraph:  # type: ignore[type-arg]
    """Build a directed graph from table relationships."""
    G: nx.DiGraph = nx.DiGraph()  # type: ignore[type-arg]

    for table_id in table_ids:
        G.add_node(table_id)

    for rel in relationships:
        G.add_edge(
            rel.from_table_id,
            rel.to_table_id,
            confidence=rel.confidence,
            relationship_type=rel.relationship_type,
            cardinality=rel.cardinality,
        )

    return G


def analyze_relationship_graph(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> dict[str, list[list[str]] | int]:
    """Analyze the graph of table relationships to detect cycles.

    Uses NetworkX to find cycles in the directed graph of tables.
    These cycles represent business process flows (e.g., AR cycle, AP cycle).

    Args:
        table_ids: List of table IDs
        relationships: List of relationship objects (duck-typed)

    Returns:
        Dict containing:
        - cycles: List of cycles (each cycle is a list of table_ids)
        - betti_0: Number of connected components in undirected graph
        - cycle_count: Total number of cycles detected
    """
    G = build_relationship_graph(table_ids, relationships)

    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        cycles = []

    G_undirected = G.to_undirected()
    betti_0 = nx.number_connected_components(G_undirected)

    return {
        "cycles": cycles,
        "betti_0": betti_0,
        "cycle_count": len(cycles),
    }


def analyze_relationship_graph_detailed(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> GraphAnalysisResult:
    """Analyze relationship graph with detailed results."""
    G = build_relationship_graph(table_ids, relationships)

    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        cycles = []

    G_undirected = G.to_undirected()
    betti_0 = nx.number_connected_components(G_undirected)

    return GraphAnalysisResult(
        cycles=cycles,
        betti_0=betti_0,
        cycle_count=len(cycles),
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
    )
